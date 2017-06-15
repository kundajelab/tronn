"""Description: code to analyze motif matrices
"""


from scipy.signal import fftconvolve
from scipy.signal import convolve2d


from tronn.models import pwm_convolve
from tronn.visualization import plot_weights



class PWM(object):
    def __init__(self, weights, name=None, threshold=None):
        self.weights = weights
        self.name = name
        self.threshold = threshold

    @staticmethod
    def from_homer_motif(motif_file):
        with open(motif_file) as fp:
            header = fp.readline().strip().split('\t')
            name = header[1]
            threshold = float(header[2])
            weights = np.loadtxt(fp)

        return PWM(weights, name, threshold)

    @staticmethod
    def get_encode_pwms(motif_file):
        pwms = []

        with open(motif_file) as fp:
            line = fp.readline().strip()
            while True:
                if line == '':
                    break

                header = line.strip('>').strip()
                weights = []
                while True:
                    line = fp.readline()
                    if line == '' or line[0] == '>':
                        break
                    weights.append(map(float, line.split()))
                pwms.append(PWM(np.array(weights).transpose(1,0), header))

        return pwms

    @staticmethod
    def from_cisbp_motif(motif_file):
        name = os.path.basename(motif_file)
        with open(motif_file) as fp:
            _ = fp.readline()
            weights = np.loadtxt(fp)[:, 1:]
        return PWM(weights, name)



def run_pwm_convolution(data_loader,
                        importance_h5,
                        out_h5,
                        batch_size,
                        pwm_file,
                        task_num):
    '''
    Wrapper function where, given an importance matrix, can convert everything
    into a motif matrix
    '''

    importance_key = 'importances_task{}'.format(task_num)
    
    # get basic key stats (to set up output h5 file)
    pwm_list = PWM.get_encode_pwms(pwm_file)
    num_pwms = len(pwm_list)
    with h5py.File(importance_h5, 'r') as hf:
        num_examples = hf[importance_key].shape[0]
        num_tasks = hf['labels'].shape[1]

    # First set up graph and convolutions model
    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader([importance_h5],
                                                 batch_size,
                                                 features_key=importance_key)

        # load the model
        motif_tensor, load_pwm_update = pwm_convolve(features, pwm_list)

        # run the model (set up sessions, etc)
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Run update to load the PWMs
        _ = sess.run(load_pwm_update)

        # set up hdf5 file for saving sequences
        with h5py.File(out_h5, 'w') as out_hf:
            motif_mat = out_hf.create_dataset('motif_scores',
                                              [num_examples, num_pwms])
            labels_mat = out_hf.create_dataset('labels',
                                               [num_examples, num_tasks])
            regions_mat = out_hf.create_dataset('regions',
                                                [num_examples, 1],
                                                dtype='S100')
            motif_names_mat = out_hf.create_dataset('motif_names',
                                                    [num_pwms, 1],
                                                    dtype='S100')

            # save out the motif names
            for i in range(len(pwm_list)):
                motif_names_mat[i] = pwm_list[i].name


            # run through batches worth of sequence
            for batch_idx in range(num_examples / batch_size + 1):

                print batch_idx * batch_size

                batch_motif_mat, batch_regions, batch_labels = sess.run([motif_tensor,
                                                                         metadata,
                                                                         labels])

                batch_start = batch_idx * batch_size
                batch_stop = batch_start + batch_size

                # TODO save out to hdf5 file
                if batch_stop < num_examples:
                    motif_mat[batch_start:batch_stop,:] = batch_motif_mat
                    labels_mat[batch_start:batch_stop,:] = batch_labels
                    regions_mat[batch_start:batch_stop] = batch_regions.astype('S100')
                else:
                    motif_mat[batch_start:num_examples,:] = batch_motif_mat[0:num_examples-batch_start,:]
                    labels_mat[batch_start:num_examples,:] = batch_labels[0:num_examples-batch_start]
                    regions_mat[batch_start:num_examples] = batch_regions[0:num_examples-batch_start].astype('S100')

        coord.request_stop()
        coord.join(threads)

    return None

def run_pwm_convolution_multiple(data_loader,
                        importance_h5,
                        out_h5,
                        batch_size,
                        num_tasks,
                        pwm_file):
    '''
    Wrapper function where, given an importance matrix, can convert everything
    into a motif matrix. Does this across multiple tasks
    '''

    # get basic key stats (to set up output h5 file)
    pwm_list = PWM.get_encode_pwms(pwm_file)
    num_pwms = len(pwm_list)
    with h5py.File(importance_h5, 'r') as hf:
        num_examples = hf['importances_task0'].shape[0]

    # set up hdf5 file for saving sequences
    with h5py.File(out_h5, 'w') as out_hf:
        motif_mat = out_hf.create_dataset('motif_scores',
                                          [num_examples, num_pwms, num_tasks])
        labels_mat = out_hf.create_dataset('labels',
                                           [num_examples, num_tasks])
        regions_mat = out_hf.create_dataset('regions',
                                            [num_examples, 1],
                                            dtype='S100')
        motif_names_mat = out_hf.create_dataset('motif_names',
                                                [num_pwms, 1],
                                                dtype='S100')

        # save out the motif names
        for i in range(len(pwm_list)):
            motif_names_mat[i] = pwm_list[i].name

        # for each task
        for task_num in range(num_tasks):

            # First set up graph and convolutions model
            with tf.Graph().as_default() as g:

                # data loader
                features, labels, metadata = data_loader([importance_h5],
                                                         batch_size,
                                                         'importances_task{}'.format(task_num))

                # load the model
                motif_tensor, load_pwm_update = models.pwm_convolve(features, pwm_list)

                # run the model (set up sessions, etc)
                sess = tf.Session()

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # start queue runners
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # Run update to load the PWMs
                _ = sess.run(load_pwm_update)

                # run through batches worth of sequence
                for batch_idx in range(num_examples / batch_size + 1):

                    print batch_idx * batch_size

                    batch_motif_mat, batch_regions, batch_labels = sess.run([motif_tensor,
                                                                             metadata,
                                                                             labels])

                    batch_start = batch_idx * batch_size
                    batch_stop = batch_start + batch_size

                    # TODO save out to hdf5 file
                    if batch_stop < num_examples:
                        motif_mat[batch_start:batch_stop,:,task_num] = batch_motif_mat
                        labels_mat[batch_start:batch_stop,:] = batch_labels[:,0:num_tasks]
                        regions_mat[batch_start:batch_stop] = batch_regions.astype('S100')
                    else:
                        motif_mat[batch_start:num_examples,:,task_num] = batch_motif_mat[0:num_examples-batch_start,:]
                        labels_mat[batch_start:num_examples,:] = batch_labels[0:num_examples-batch_start,0:num_tasks]
                        regions_mat[batch_start:num_examples] = batch_regions[0:num_examples-batch_start].astype('S100')

                coord.request_stop()
                coord.join(threads)

    return None


def run_motif_distance_extraction(data_loader,
                        importance_h5,
                        out_h5,
                        batch_size,
                        pwm_file,
                        task_num,
                        top_k_val=2):
    '''
    Wrapper function where, given an importance matrix, can convert everything
    into motif scores and motif distances for the top k hits
    Only take positive sequences to build grammars!
    '''

    importance_key = 'importances_task{}'.format(task_num)
    print importance_key
    
    # get basic key stats (to set up output h5 file)
    pwm_list = PWM.get_encode_pwms(pwm_file)
    num_pwms = len(pwm_list)
    with h5py.File(importance_h5, 'r') as hf:
        num_examples = hf[importance_key].shape[0]
        num_tasks = hf['labels'].shape[1]

    # First set up graph and convolutions model
    with tf.Graph().as_default() as g:

        # data loader
        features, labels, metadata = data_loader([importance_h5],
                                                 batch_size,
                                                 importance_key)

        # load the model
        motif_scores, motif_distances, load_pwm_update = models.top_motifs_w_distances(features, pwm_list, top_k_val)

        # run the model (set up sessions, etc)
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Run update to load the PWMs
        _ = sess.run(load_pwm_update)

        # set up hdf5 file for saving sequences
        # TODO edit these datasets
        with h5py.File(out_h5, 'w') as out_hf:
            motif_score_mat = out_hf.create_dataset('motif_scores',
                                              [num_examples, num_pwms, num_pwms, top_k_val ** 2])
            motif_dist_mat = out_hf.create_dataset('motif_dists',
                [num_examples, num_pwms, num_pwms, top_k_val ** 2])
            labels_mat = out_hf.create_dataset('labels',
                                               [num_examples, num_tasks])
            regions_mat = out_hf.create_dataset('regions',
                                                [num_examples, 1],
                                                dtype='S100')
            motif_names_mat = out_hf.create_dataset('motif_names',
                                                    [num_pwms, 1],
                                                    dtype='S100')

            # save out the motif names
            for i in range(len(pwm_list)):
                motif_names_mat[i] = pwm_list[i].name

            # run through batches worth of sequence
            for batch_idx in range(num_examples / batch_size + 1):

                print batch_idx * batch_size

                batch_motif_scores, batch_motif_dists, batch_regions, batch_labels = sess.run([motif_scores,
                    motif_distances,
                                                                         metadata,
                                                                         labels])

                batch_start = batch_idx * batch_size
                batch_stop = batch_start + batch_size

                # TODO save out to hdf5 file
                if batch_stop < num_examples:
                    motif_score_mat[batch_start:batch_stop,:,:,:] = batch_motif_scores
                    motif_dist_mat[batch_start:batch_stop,:,:,:] = batch_motif_dists
                    labels_mat[batch_start:batch_stop,:] = batch_labels
                    regions_mat[batch_start:batch_stop] = batch_regions.astype('S100')
                else:
                    motif_score_mat[batch_start:num_examples,:,:,:] = batch_motif_scores[0:num_examples-batch_start,:,:,:]
                    motif_dist_mat[batch_start:num_examples,:,:,:] = batch_motif_dists[0:num_examples-batch_start,:,:,:]
                    labels_mat[batch_start:num_examples,:] = batch_labels[0:num_examples-batch_start]
                    regions_mat[batch_start:num_examples] = batch_regions[0:num_examples-batch_start].astype('S100')

        coord.request_stop()
        coord.join(threads)

    return None


# =======================================================================
# Other useful helper functions
# =======================================================================

def extract_positives_from_motif_mat(h5_file, out_file, task_num):
    '''
    Extract positive set from h5 file to handle in R
    remember to keep track of index positions
    '''

    with h5py.File(h5_file, 'r') as hf:

        # better to pull it all into memory to slice fast
        labels = hf['labels'][:,task_num]
        motif_scores = hf['motif_scores'][:] 
        motif_names = list(hf['motif_names'][:,0])
        regions = list(hf['regions'][:,0])

        pos_array = motif_scores[labels > 0,:]
        pos_regions = list(hf['regions'][labels > 0,0])

        motif_df = pd.DataFrame(data=pos_array[:],
                                index=pos_regions,
                                columns=motif_names)

        # also save out indices
        pos_indices = np.where(labels > 0)
        motif_df['indices'] = pos_indices[0]
        motif_df.to_csv(out_file, sep='\t', compression='gzip')

    return None


def extract_positives_from_motif_topk_mat(h5_file, out_file):
    '''
    Extract positive set from h5 file to handle in R
    remember to keep track of index positions
    '''

    with h5py.File(h5_file, 'r') as hf:

        # better to pull it all into memory to slice fast
        labels = hf['labels'][:,0]
        motif_scores = hf['motif_scores'][:] 
        motif_names = list(hf['motif_names'][:,0])
        regions = list(hf['regions'][:,0])

        pos_array = motif_scores[labels > 0,:]
        pos_regions = list(hf['regions'][labels > 0,0])

        motif_df = pd.DataFrame(data=pos_array[:],
                                index=pos_regions,
                                columns=motif_names)

        # also save out indices
        pos_indices = np.where(labels > 0)
        motif_df['indices'] = pos_indices[0]
        motif_df.to_csv(out_file, sep='\t', compression='gzip')

    return None




def bootstrap_fdr(motif_mat_h5, out_prefix, task_num, region_set=None,
                  bootstrap_num=9999, fdr=0.005, zscore_cutoff=1.0):
    '''
    Given a motif matrix and labels, calculate a bootstrap FDR
    '''

    # set up numpy array
    with h5py.File(motif_mat_h5, 'r') as hf:

        # better to pull it all into memory to slice fast
        labels = hf['labels'][:,task_num] 
        motif_scores = hf['motif_scores'][:] # TODO only take sequences that are from skin sequences
        motif_names = list(hf['motif_names'][:,0])

        # first calculate the scores for positive set
        if region_set != None:
            # TODO allow passing in an index set which represents your subset of positives
            pos_indices = np.loadtxt(region_set, dtype=int)
            pos_indices_sorted = np.sort(pos_indices)
            pos_array = motif_scores[pos_indices_sorted,:]
            
        else:
            pos_array = motif_scores[labels > 0,:]
        pos_array_z = scipy.stats.mstats.zscore(pos_array, axis=1)
        pos_vector = np.mean(pos_array_z, axis=0) # normalization

        # TODO save out the mean column of positives
        motif_z_avg_df = pd.DataFrame(data=pos_vector, index=motif_names)
        motif_z_avg_df.to_csv('{}.zscores.txt'.format(out_prefix), sep='\t')

        num_pos_examples = pos_array.shape[0]
        num_motifs = pos_array.shape[1]

        # set up results array
        bootstraps = np.zeros((bootstrap_num+1, num_motifs))
        bootstraps[0,:] = pos_vector

        # Now only select bootstraps from regions that are open in skin
        #motif_scores = motif_scores[np.sum(hf['labels'], axis=1) > 0,:]

        for i in range(bootstrap_num):

            if i % 1000 == 0:
                print i

            # randomly select examples 
            bootstrap_indices = np.random.choice(motif_scores.shape[0],
                                                 num_pos_examples,
                                                 replace=False)
            bootstrap_indices.sort()
            bootstrap_array = motif_scores[bootstrap_indices,:]

            bootstrap_array_z = scipy.stats.mstats.zscore(bootstrap_array, axis=1)

            # calculate sum
            bootstrap_sum = np.mean(bootstrap_array_z, axis=0)

            # save into vector
            bootstraps[i+1,:] = bootstrap_sum

    # convert to ranks and save out
    bootstrap_df = pd.DataFrame(data=bootstraps, columns=motif_names)
    bootstrap_ranks_df = bootstrap_df.rank(ascending=False, pct=True)
    pos_fdr = bootstrap_ranks_df.iloc[0,:]
    pos_fdr.to_csv('{}.bootstrap_fdr.txt'.format(out_prefix), sep='\t')

    # also save out a list of those that passed the FDR cutoff
    fdr_cutoff = pos_fdr.ix[pos_fdr < 0.05]
    fdr_cutoff.to_csv('{}.bootstrap_fdr.cutoff.txt'.format(out_prefix), sep='\t')

    # also save out list that pass FDR and also zscore cutoff
    pos_fdr_t = pd.DataFrame(data=pos_fdr, index=pos_fdr.index)
    fdr_w_zscore = motif_z_avg_df.merge(pos_fdr_t, left_index=True, right_index=True)
    fdr_w_zscore.columns = ['zscore', 'FDR']
    fdr_w_zscore_cutoffs = fdr_w_zscore[(fdr_w_zscore['FDR'] < 0.005) & (fdr_w_zscore['zscore'] > zscore_cutoff)]
    fdr_w_zscore_cutoffs_sorted = fdr_w_zscore_cutoffs.sort_values('zscore', ascending=False)
    fdr_w_zscore_cutoffs_sorted.to_csv('{}.fdr_cutoff.zscore_cutoff.txt'.format(out_prefix), sep='\t')
    
    
    return None


def generate_motif_x_motif_mat(motif_mat_h5, out_prefix, region_set=None, score_type='spearman'):
    '''
    With a sequences x motif mat, filter for region set and then get
    correlations of motif scores with other motif scores
    '''

    with h5py.File(motif_mat_h5, 'r') as hf:

        # better to pull it all into memory to slice fast
        labels = hf['labels'][:,0]
        motif_scores = hf['motif_scores'][:] 
        motif_names = list(hf['motif_names'][:,0])

        # select region set if exists, if not just positives
        if region_set != None:
            # TODO allow passing in an index set which represents your subset of positives
            pos_indices = np.loadtxt(region_set, dtype=int)
            pos_indices_sorted = np.sort(pos_indices)
            pos_array = motif_scores[pos_indices_sorted,:]
            
        else:
            pos_array = motif_scores[labels > 0,:]

        pos_array_z = scipy.stats.mstats.zscore(pos_array, axis=1)


        # Now for each motif, calculate the correlation (spearman)
        num_motifs = len(motif_names)
        motif_x_motif_array = np.zeros((num_motifs, num_motifs))

        for i in range(num_motifs):
            if i % 50 == 0:
                print i
            for j in range(num_motifs):
                if score_type == 'spearman':
                    score, pval = scipy.stats.spearmanr(pos_array_z[:,i], pos_array_z[:,j])
                elif score_type == 'mean_score':
                    score = np.mean(pos_array_z[:,i] * pos_array_z[:,j])
                elif score_type == 'mean_x_spearman':
                    rho, pval = scipy.stats.spearmanr(pos_array_z[:,i], pos_array_z[:,j])
                    score = rho * np.mean(pos_array_z[:,i] * pos_array_z[:,j])
                else:
                    score, pval = scipy.stats.spearmanr(pos_array_z[:,i], pos_array_z[:,j])
                motif_x_motif_array[i,j] = score

        motif_x_motif_df = pd.DataFrame(data=motif_x_motif_array, columns=motif_names, index=motif_names)
        motif_x_motif_df.to_csv('{0}.motif_x_motif.{1}.txt'.format(out_prefix, score_type), sep='\t')
    

    return None




def group_motifs_by_sim(motif_list, motif_dist_mat, out_file, cutoff=0.7):
    '''
    Given a motif list and a distance matrix, form
    groups of motifs and put out list
    '''

    # Load the scores into a dictionary
    motif_dist_df = pd.read_table(motif_dist_mat, index_col=0)
    motif_dist_dict = {}
    print motif_dist_df.shape
    motif_names = list(motif_dist_df.index)
    for i in range(motif_dist_df.shape[0]):
        motif_dist_dict[motif_names[i]] = {}
        for j in range(motif_dist_df.shape[1]):
            motif_dist_dict[motif_names[i]][motif_names[j]] = motif_dist_df.iloc[i, j]

    # if first motif, put into motif group dict as seed
    motif_groups = []

    with gzip.open(motif_list, 'r') as fp:
        for line in fp:
            current_motif = line.strip()
            print current_motif
            current_motif_matched = 0

            if len(motif_groups) == 0:
                motif_groups.append([current_motif])
                continue

            for i in range(len(motif_groups)):
                # compare to each motif in group. if at least 1 is above cutoff, join group
                motif_group = list(motif_groups[i])
                for motif in motif_group:
                    similarity = motif_dist_dict[motif][current_motif]
                    if similarity >= cutoff:
                        motif_groups[i].append(current_motif)
                        current_motif_matched = 1

                motif_groups[i] = list(set(motif_groups[i]))

            if current_motif_matched == 0:
                motif_groups.append([current_motif])


    with gzip.open(out_file, 'w') as out:
        for motif_group in motif_groups:
            out.write('#\n')
            for motif in motif_group:
                out.write('{}\n'.format(motif))

    return None


def get_motif_similarities(motif_list, motif_dist_mat, out_file, cutoff=0.5):
    '''
    Given a motif list and a distance matrix, form
    groups of motifs and put out list
    '''

    # Load the scores into a dictionary
    motif_dist_df = pd.read_table(motif_dist_mat, index_col=0)
    motif_dist_dict = {}
    print motif_dist_df.shape
    motif_names = list(motif_dist_df.index)
    for i in range(motif_dist_df.shape[0]):
        motif_dist_dict[motif_names[i]] = {}
        for j in range(motif_dist_df.shape[1]):
            motif_dist_dict[motif_names[i]][motif_names[j]] = motif_dist_df.iloc[i, j]


    # load in motifs
    important_motifs = pd.read_table(motif_list, index_col=0)
    important_motif_list = list(important_motifs.index)

    with open(out_file, 'w') as out:
        with open(motif_list, 'r') as fp:
            for line in fp:

                if 'zscore' in line:
                    continue
                
                current_motif = line.strip().split('\t')[0]
                print current_motif
                for motif in important_motif_list:
                    if motif == current_motif:
                        continue

                    similarity = motif_dist_dict[motif][current_motif]
                    if similarity >= cutoff:
                        out.write('{}\t{}\t{}\n'.format(current_motif, motif, similarity))

    return None


def choose_strongest_motif_from_group(zscore_file, motif_groups_file, out_file):
    '''
    Takes a motif groups file and zscores and chooses strongest one to output
    '''

    # read in zscore file to dictionary
    zscore_dict = {}
    with open(zscore_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')

            if fields[0] == '0':
                continue

            zscore_dict[fields[0]] = float(fields[1])

    # for each motif group, select strongest
    with gzip.open(motif_groups_file, 'r') as fp:
        with gzip.open(out_file, 'w') as out:
            motif = ''
            zscore = 0


            for line in fp:

                if line.startswith('#'):
                    if motif != '':
                        out.write('{0}\t{1}\n'.format(motif, zscore))

                    motif = ''
                    zscore = 0
                    continue

                current_motif = line.strip()
                current_zscore = zscore_dict[current_motif]

                if current_zscore > zscore:
                    motif = current_motif
                    zscore = current_zscore

    return None

def add_zscore(zscore_file, motif_file, out_file):
    '''
    Quick function to put zscore with motif
    '''

    # read in zscore file to dictionary
    zscore_dict = {}
    with open(zscore_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')

            if fields[0] == '0':
                continue

            zscore_dict[fields[0]] = float(fields[1])

    # for each motif add zscore
    with open(motif_file, 'r') as fp:
        with open(out_file, 'w') as out:
            for line in fp:

                motif = line.strip()
                zscore = zscore_dict[motif]
                out.write('{0}\t{1}\n'.format(motif, zscore))

    return None


def reduce_motif_redundancy_by_dist_overlap(motif_dists_mat_h5, motif_offsets_mat_file, motif_list_file):
    '''
    remove motifs if they overlap (ie, their average distance is 0)
    '''

    # read in motif list
    motif_list = []    
    with gzip.open(motif_list_file, 'r') as fp:
        for line in fp:
            fields = line.strip().split('\t')
            motif_list.append((fields[0], float(fields[1])))

    final_motif_list = []
    with h5py.File(motif_dists_mat_h5, 'r') as hf:


        # make a motif to index dict
        motif_names = list(hf['motif_names'][:,0])
        name_to_index = {}
        for i in range(len(motif_names)):
            name_to_index[motif_names[i]] = i


        for i in range(len(motif_list)):
            is_best_single_motif = 1
            motif_i = motif_list[i][0]
            motif_i_idx = name_to_index[motif_i]

            for j in range(len(motif_list)):
                motif_j = motif_list[j][0]
                motif_j_idx = name_to_index[motif_j]

                dists = hf['motif_dists'][:,motif_i_idx, motif_j_idx,:]
                dists_flat = dists.flatten()

                dists_mean = np.mean(dists_flat)

                print motif_i, motif_j, dists_mean

            # compare to all others. if no matches stronger than it, put into final list

            # if there is a match, but the other one is higher zscore, do not add




        # for each motif compared to each other motif,
        # check to see their average distance


    return None


def make_score_dist_plot(motif_a, motif_b, motif_dists_mat_h5, out_prefix):
    '''
    Helper function to make plot
    '''

    with h5py.File(motif_dists_mat_h5, 'r') as hf:

        # make a motif to index dict
        motif_names = list(hf['motif_names'][:,0])
        name_to_index = {}
        for i in range(len(motif_names)):
            name_to_index[motif_names[i]] = i

        motif_a_idx = name_to_index[motif_a]
        motif_b_idx = name_to_index[motif_b]

        scores = hf['motif_scores'][:,motif_a_idx,motif_b_idx,:]
        dists = hf['motif_dists'][:,motif_a_idx,motif_b_idx,:]

        # flatten
        scores_flat = scores.flatten()
        dists_flat = dists.flatten()

        # TODO adjust the dists
        
        # make a pandas df and save out to text
        out_table = '{}.scores_w_dists.txt.gz'.format(out_prefix)
        dists_w_scores = np.stack([dists_flat, scores_flat], axis=1)
        dists_w_scores_df = pd.DataFrame(data=dists_w_scores)
        dists_w_scores_df.to_csv(out_table, sep='\t', compression='gzip', header=False, index=False)

    # then plot in R
    plot_script = '/users/dskim89/git/tronn/scripts/make_score_dist_plot.R'
    os.system('Rscript {0} {1} {2}'.format(plot_script, out_table, out_prefix))


    return None


def plot_sig_pairs(motif_pair_file, motif_dists_mat_h5, cutoff=3):
    '''
    Go through sig file and plot sig pairs
    '''

    seen_pairs = []

    with open(motif_pair_file, 'r') as fp:
        for line in fp:

            [motif_a, motif_b, zscore] = line.strip().split('\t')

            if float(zscore) >= cutoff:
                motif_a_hgnc = motif_a.split('_')[0]
                motif_b_hgnc = motif_b.split('_')[0]

                pair = '{0}-{1}'.format(motif_a_hgnc, motif_b_hgnc)

                if pair not in seen_pairs:
                    out_prefix = '{0}.{1}-{2}'.format(motif_pair_file.split('.txt')[0], motif_a_hgnc, motif_b_hgnc)
                    make_score_dist_plot(motif_a, motif_b, motif_dists_mat_h5, out_prefix)

                    seen_pairs.append(pair)
                    seen_pairs.append('{0}-{1}'.format(motif_b_hgnc, motif_a_hgnc))

    return None


def get_significant_motif_pairs(motif_list, motif_x_motif_mat_file, out_file, manual=False, std_cutoff=3):
    '''
    With a motif list, compare all to all and check significance
    '''

    # first load in the motif x motif matrix
    motif_x_motif_df = pd.read_table(motif_x_motif_mat_file, index_col=0)
    motif_names = list(motif_x_motif_df.index)

    # get index dictionary
    motif_to_idx = {}
    for i in range(len(motif_names)):
        motif_to_idx[motif_names[i]] = i

    # calculate mean and std across all values in matrix
    mean = motif_x_motif_df.values.mean()
    std = motif_x_motif_df.values.std()

    print mean
    print std

    # for each motif, compare to each other one. only keep if above 2 std
    if manual:
        important_motifs = pd.read_table(motif_list, header=None)
        important_motif_list = list(important_motifs[0])
    else:
        important_motifs = pd.read_table(motif_list, index_col=0)
        important_motif_list = list(important_motifs.index)

    print important_motif_list

    already_seen = []
    
    with open(out_file, 'w') as out:

        for i in range(len(important_motif_list)):

            mean = motif_x_motif_df.values.mean(axis=0)[i]
            std = motif_x_motif_df.values.std(axis=0)[i]

            print mean, std


            for j in range(len(important_motif_list)):

                name_1 = important_motif_list[i]
                name_2 = important_motif_list[j]

                if name_1 == name_2:
                    continue

                idx_1 = motif_to_idx[name_1]
                idx_2 = motif_to_idx[name_2]

                score = motif_x_motif_df.iloc[idx_1, idx_2]

                if score >= (mean + std_cutoff * std):
                    print name_1, name_2, score
                    out_string = '{0}\t{1}\t{2}\n'.format(name_1, name_2, score)
                    if out_string in already_seen:
                        continue
                    else:
                        out.write(out_string)
                        already_seen.append('{1}\t{0}\t{2}\n'.format(name_1, name_2, score))

    return None
