#!/usr/bin/env Rscript

library('visNetwork')


# script to make network form of grammar




# need to have: node IDs, links (to and from), labels, node size (motif z score), links strength (correlation strength)

args <- commandArgs(trailingOnly=TRUE)
links_file <- args[1]
zscore_file <- args[2]
out_file <- args[3]

node_size_factor <- 15
edge_width_factor <- 1

print(out_file)

links <- read.table(links_file, header=FALSE, sep='\t', stringsAsFactors=FALSE)
colnames(links) <- c('to', 'from', 'correlation')
links$width <- edge_width_factor*links$correlation


print(head(links))

nodes <- read.table(zscore_file, header=TRUE, sep='\t', stringsAsFactors=FALSE)
print(nodes)
colnames(nodes) <- c('id', 'size', 'fdr')

nodes$size <- node_size_factor*nodes$size

#nodes <- data.frame(id=unique(c(links$to, links$from)))
nodes$label <- sub("_.*", "", nodes$id)
print(head(nodes))

net <- visNetwork(nodes, links)

#print(dirname(out_file))
#out_dir <- paste(getwd(), dirname(out_file), sep='/')
setwd(dirname(out_file))

#visSave(net, file=basename(out_file), selfcontained=TRUE) # requires pandoc
visSave(net, file=basename(out_file), selfcontained=FALSE)



q()

png('test_network.png')
visNetwork(nodes, links)
dev.off()









