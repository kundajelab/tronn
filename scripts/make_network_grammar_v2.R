#!/usr/bin/env Rscript

library('igraph')
library(reshape2)

# script to make network form of grammar
# need to have: node IDs, links (to and from), labels, node size (motif z score), links strength (correlation strength)

args <- commandArgs(trailingOnly=TRUE)
zscore_file <- args[1]
sim_file <- args[2]
out_file <- args[3]

node_size_factor <- 3
edge_width_factor <- 3

# set up nodes
nodes <- read.table(zscore_file, header=TRUE, sep='\t', stringsAsFactors=FALSE)
nodes_reduced <- nodes[,c('motif_id', 'hgnc_symbol', 'zscore')]
colnames(nodes_reduced) <- c('id', 'label', 'size')
nodes_reduced$size <- node_size_factor*nodes_reduced$size

nodes <- nodes_reduced[!duplicated(nodes_reduced$id),]

print(head(nodes))


# set up links
links <- read.table(sim_file, header=TRUE, row.names=NULL, sep='\t')
links$X <- NULL
links_dedup <- links[!(colnames(links)=="HOXD3_ENSG00000128652_D.1"), !(colnames(links)=="HOXD3_ENSG00000128652_D.1")]
rownames(links_dedup) <- colnames(links_dedup)

# remove duplicates
links_dedup_mat <- as.matrix(links_dedup)
links_dedup_mat[lower.tri(links_dedup_mat, diag=TRUE)] <- 0
links_dedup <- data.frame(links_dedup_mat)

links_dedup$id <- rownames(links_dedup)
links_melted <- melt(links_dedup)
colnames(links_melted) <- c('to', 'from', 'correlation')
links_melted <- links_melted[links_melted$correlation > 0.5,]
links_melted$width <- edge_width_factor*links_melted$correlation

# and remove links that are not in nodes
links <- links_melted[(links_melted$to %in% nodes$id),]
links <- links[(links$from %in% nodes$id),]

print(links)

# build the net
net <- graph_from_data_frame(d=links, vertices=nodes, directed=FALSE)

# call communities to figure out how to order
#clp <- cluster_label_prop(net)
clp <- cluster_edge_betweenness(net)
clp_membership <- data.frame(membership=clp$membership, v_idx=seq(1:length(clp$membership)))
#print(clp_membership)

# sort by membership. then add new column. then resort. then put into permue
clp_membership_ordered <- clp_membership[order(clp_membership$membership),]
clp_membership_ordered$final_dest <- seq(1:nrow(clp_membership_ordered))
clp_membership_ordered <- clp_membership_ordered[order(clp_membership_ordered$v_idx),]

#print(clp_membership_ordered)

net <- permute(net, clp_membership_ordered$final_dest)

l <- layout_in_circle(net)
#l <- layout_with_fr(net)
#l <- norm_coords(l, ymin=-1, ymax=1, xmin=-1, xmax=1)

num_nodes <- nrow(nodes)
neg_side_num <- ceiling(num_nodes / 2) + 1
pos_side_num <- floor(num_nodes / 2) + 1

neg_side_degrees <- seq(0, -pi, length.out=neg_side_num)[1:neg_side_num-1]
pos_side_degrees <- seq(pi, 0, length.out=pos_side_num)[1:pos_side_num-1]

label_degree <- c(neg_side_degrees, pos_side_degrees)

#half_node_num <- floor((nrow(nodes)-1)/2)
    
#label_degree_neg <- seq(0, -pi, length.out=half_node_num)[2:half_node_num]
#label_degree_pos <- seq(0,)

#label_degree <- seq(-pi, pi, length.out=(nrow(nodes)+1))[2:(nrow(nodes)+1)]

print(label_degree)

print(clp_membership_ordered)

# some extra work for community coloring
clp_membership_ordered <- clp_membership[order(clp_membership$membership),]
index_groups <- split(seq(1:nrow(clp_membership_ordered)), clp_membership_ordered$membership)
print(index_groups)


pdf(out_file)
#pdf('test.pdf')
plot(net,
     vertex.label.color='black',
     vertex.label.family='Helvetica',
     #vertex.label.font=2,
     vertex.label.cex=1.5,
     #vertex.label.degree=-pi/2,
     vertex.label.degree=label_degree,
     vertex.label.dist=2,
     vertex.color='tomato',
     edge.curved=0.1,
     #rescale=F,
     #layout=l*0.6,
     layout=l*0.2,
     margin=c(0.2, 0.2, 0.2, 0.2),
     mark.groups=index_groups,
     mark.border=NA
     )
dev.off()








