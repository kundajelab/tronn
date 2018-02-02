import os
from sys import argv
import glob
import subprocess
import h5py
import logging
import pandas as pd
import commands
import numpy as np

import tronn.preprocess as trope
from tronn.util.parallelize import setup_multiprocessing_queue
from tronn.util.parallelize import run_in_parallel


src_file, work_dir, prefix = argv[1:]
print("parameter checking--------------")
print(src_file)
print(work_dir)
print(prefix)

if('hBreast' in src_file):
	ref_fasta = "/mnt/data/annotations/by_release/hg20.GRCh38/GRCh38.genome.fa"
else:
	ref_fasta = "/mnt/data/annotations/by_organism/mouse/mm10/GRCm38.genome.fa"


def generate_labels(
        interval_file,
        h5_ml_file,
	prefix): 
    	"""Generate labels
    
    	Args:
      		interval_file: interval_file
      		h5_ml_file: where to store the labels
		prefix: prefix for output files
    	Returns:
      		None
    	"""
	result = commands.getoutput("zcat " + interval_file + " | cut -f 1-3 --complement ").split('\n')
	bin_count = len(result)
	num_tasks = len(np.float32(result[0].split('\t')))
	print(interval_file, bin_count, num_tasks)
	label_set_names = ['V'+str(i+1) for i in range(num_tasks)]
    	# then for each, generate labels
    	with h5py.File(h5_ml_file, 'a') as hf:

        	if not 'labels' in hf:
            		all_labels = hf.create_dataset('labels', (bin_count, num_tasks))
        	else:
            		all_labels = hf['labels']

		if not 'label_metadata' in hf:
                        label_names = hf.create_dataset('label_metadata', (num_tasks,), dtype='S1000')
		hf['label_metadata'][:,] = label_set_names
		
		"""
        	if not 'label_metadata' in hf:
            		label_names = hf.create_dataset('label_metadata', (num_tasks,), dtype='S1000')

        	hf['label_metadata'][:,] = label_set_names
		""" 
                # now store all values at once
        	print "storing in file"
        	for i in range(len(result)):
			single_result = np.float32(result[i].split('\t'))
            		if i % 10000 == 0:
				print(i)
                		#print(i, single_result, type(single_result), len(single_result))
            		all_labels[i,:] = single_result
	
	print("done labeling....", interval_file)
    	return None


def generate_labels_chrom(
        interval_dir,
        fasta_dir,
        out_dir,
	prefix,
        parallel=12):
    	"""Utilize func_worker to run on chromosomes
    	Args:
      		interval_dir: directory to store extended bin files
      		fasta_dir: directory of tab delim FASTA sequence examples
      		out_dir: h5 file directory
      		prefix: output prefix
      		parallel: nunmber to run in parallel
    	Returns: 
      		None
    	"""
    	label_queue = setup_multiprocessing_queue()

    	# First find chromosome files
    	chrom_files = glob.glob('{0}/*.bed.gz'.format(interval_dir))
    	print 'Found {} chromosome files'.format(len(chrom_files))

    	# Then for each file, give the function and run
    	for chrom_file in chrom_files:
        	print chrom_file

        	chrom_prefix = chrom_file.split('/')[-1].split('.bed.gz')[0]
        	h5_ml_file = '{0}/{1}.h5'.format(out_dir,
                                            chrom_prefix)
		#generate_labels(chrom_file, h5_ml_file, prefix)
		#"""
        	labels_args = [chrom_file, h5_ml_file, prefix]

        	label_queue.put([generate_labels, labels_args])
		#"""
		
    	run_in_parallel(label_queue, parallel=parallel, wait=True)

    	return None


def generate_examples(
        interval_file,
        fasta_sequences,
        examples_file,
        ref_fasta_file,
        reverse_complemented):
    	"""Generate one hot encoded sequence examples from binned regions
    	Args:
      		interval_file: interval file 
      		fasta_sequences: TAB delim output file of sequence strings for each bin
      		examples_file: h5 output file where final examples are stored
      		ref_fasta_file: reference FASTA file (to get sequences)
      		reverse_complemented: Boolean for whether to generate RC sequences too
    	Returns:
      		None
    	"""
    	#logging.info("extending bins for {}...".format(binned_file))
    	"""
	with gzip.open(binned_extended_file, 'w') as out:
        	with gzip.open(binned_file, 'rb') as fp:
            		for line in fp:
                		[chrom, start, stop, metadata] = line.strip().split('\t')
                if int(start) - extend_length < 0:
                    new_start = 0
                    new_stop = final_length
                else:
                    new_start = int(start) - extend_length
                    new_stop = int(stop) + extend_length

                if reverse_complemented:
                    out.write('{0}\t{1}\t{2}\tfeatures={0}:{1}-{2}(+);{3}\t1\t+\n'.format(
                        chrom, new_start, new_stop, metadata))
                    out.write('{0}\t{1}\t{2}\tfeatures={0}:{1}-{2}(-);{3}\t1\t-\n'.format(
                        chrom, new_start, new_stop, metadata))
                else:
                    out.write('{0}\t{1}\t{2}\tfeatures={0}:{1}-{2};{3}\n'.format(
                        chrom, new_start, new_stop, metadata))
	"""
	
    	# Then run get fasta to get sequences
    	print("getting fasta sequences...", interval_file)
    	get_sequence = ("bedtools getfasta -s -fo {0} -tab -name "
                    	"-fi {1} "
                    	"-bed {2}").format(fasta_sequences,
                                       ref_fasta_file,
                                       interval_file)
    	print get_sequence
    	os.system(get_sequence)

    	num_bins = int(subprocess.check_output(
        	['wc','-l', fasta_sequences]).strip().split()[0])
    
	# Then one hot encode and save into hdf5 file
        print("one hot encoding...", interval_file, num_bins)	
	final_length = 1000 #manual but can be caculated from 1st entry of any interval file
    	with h5py.File(examples_file, 'a') as hf:
        	features_hf = hf.create_dataset('features', (num_bins, 1, final_length, 4))
        	metadata_hf = hf.create_dataset('example_metadata', (num_bins,), dtype='S1000')
        
        	counter = 0
        	with open(fasta_sequences, 'rb') as fp:
            		for line in fp:
                
                		if counter % 50000 == 0:
                    			print counter

                		sequence = line.strip().split()[1].upper()
                		metadata = line.strip().split()[0]
                
                		# convert each sequence into one hot encoding
                		# and load into hdf5 file
                		features_hf[counter,:,:,:] = trope.one_hot_encode(sequence)

                		# track the region name too: could be garbage as I did not named the sequences
                		metadata_hf[counter] = metadata
               	 
                		counter += 1

    	# TO DO now gzip the file
    	os.system('gzip {}'.format(fasta_sequences))
	print("saved examples....", interval_file)
    	return None


def generate_intervals_chrom(
	interval_dir,
        fasta_dir,
        out_dir,
        prefix,
        ref_fasta_file,
        reverse_complemented,
        parallel=12):
    	"""Utilize func_workder to run on chromosomes
    	Args:
      		interval_dir: directory with binned BED files
      		fasta_dir: directory to store FASTA sequences for each bin
      		out_dir: directory to store h5 files
      		prefix: prefix to append to output files
      		ref_fasta_file: reference FASTA file
      		reverse_complemented: boolean on whether to produce RC examples
      		parallel: number of parallel processes
    	Returns:
      		None
    	"""
	example_queue = setup_multiprocessing_queue()

    	# First find chromosome files
    	chrom_files = glob.glob('{0}/*.bed.gz'.format(interval_dir))
    	print 'Found {} chromosome files'.format(len(chrom_files))
	
	
    	# Then for each file, give the function and run
    	for chrom_file in chrom_files:
        	chrom_prefix = chrom_file.split('/')[-1].split('.bed')[0]
		#interval_file = '{0}/{1}.bed.gz'.format(interval_dir, chrom_prefix)
        	regions_fasta_file = '{0}/{1}.fa'.format(fasta_dir, chrom_prefix)
        	examples_file = '{0}/{1}.h5'.format(out_dir, chrom_prefix)
		#if not os.path.isfile(regions_fasta_file+'.gz'):
		#generate_examples(chrom_file, regions_fasta_file, examples_file, ref_fasta_file, reverse_complemented)
		#"""
        	examples_args = [chrom_file, regions_fasta_file,
                         examples_file, ref_fasta_file,
                         reverse_complemented]

        	if not os.path.isfile(examples_file):
        		example_queue.put([generate_examples, examples_args])
		#"""
		#break
    	run_in_parallel(example_queue, parallel=parallel, wait=True)
	
    	return None



def generate_interval_dataset():
	#TODO: check whether the work_dir exits: if not make a directory: if exists: give warning
	#make a sorted file in destination directory
	intervalfile = os.path.join(work_dir, src_file.split("/")[-1])
	print(intervalfile)
	if(not os.path.exists(intervalfile)):
		print(intervalfile + 'does not exists...')
		print("Need to copy the sorted file and gip it!")
		sorting_command = "zcat " + src_file + " | sort -k1,1 -k2,2n | gzip -c > " + intervalfile
		os.system(sorting_command)
	
	tmp_dir = "{}/tmp".format(work_dir)
    	os.system("mkdir -p {}".format(tmp_dir))	
	
	# split into chromosomes, everything below done by chromosome
    	chrom_interval_dir = '{}/interval_by_chrom'.format(tmp_dir)
    	if not os.path.isfile('{0}/{1}.chrY.bed.gz'.format(chrom_interval_dir, prefix)):
        	os.system('mkdir -p {}'.format(chrom_interval_dir))
		print("Splitting the intervals into chromosomes....")
        	trope.split_bed_to_chrom_bed(chrom_interval_dir, intervalfile, prefix)

	# generate one-hot encoding sequence files (examples) and then labels
    	#intersect_dir = '{}/intersect'.format(tmp_dir)
	regions_fasta_dir = '{}/regions_fasta'.format(tmp_dir)
    	chrom_hdf5_dir = '{}/h5'.format(work_dir)

	# now run example generation and label generation
    	if not os.path.isfile('{0}/{1}.chrY.h5'.format(chrom_hdf5_dir, prefix)):
        	os.system('mkdir -p {}'.format(chrom_hdf5_dir))
        	os.system('mkdir -p {}'.format(regions_fasta_dir))
        	#os.system('mkdir -p {}'.format(bin_ext_dir))
        	generate_intervals_chrom(
            		chrom_interval_dir,
            		regions_fasta_dir,
            		chrom_hdf5_dir,
            		prefix,
            		ref_fasta,
            		reverse_complemented=True)
        	#os.system('mkdir -p {}'.format(intersect_dir))
        	generate_labels_chrom(
            		chrom_interval_dir,
            		regions_fasta_dir,
            		chrom_hdf5_dir,
			prefix)
        	#os.system("rm -r {}".format(intersect_dir))

	return '{}/h5'.format(work_dir)


print("before calling--------------")
print(src_file)
print(work_dir)
print(prefix)
	
whatever = generate_interval_dataset()

print(whatever)



