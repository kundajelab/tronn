#!/usr/bin/env/ Rscript


# description: module for easy management of
# clustering vectors, hard (vector) or soft
# (array)




# fn for separating out clusters
get_cluster_data <- function(data, clusters, cluster_id) {

    # check if soft (2d) or hard (1D)
    if (length(dim(clusters)) > 1) {
        # soft
        cluster_data <- data[clusters[,cluster_id] > 0,]
        
    } else {
        # hard
        cluster_data <- data[clusters == cluster_id,]
        
    }

    return(cluster_data)

}


# fn for ordering clusters
order_by_clusters <- function(data, clusters) {

    # check if soft (2d) or hard (1D)
    if (length(dim(clusters)) > 1) {
        # soft
        data <- apply(data, 2, sort, decreasing=FALSE)
        
    } else {
        # hard
        if (length(dim(data)) == 3) {
            data <- data[order(clusters),,]
            data <- data[clusters != -1,,]
        } else {
            data <- data[order(clusters),]
            data <- data[clusters != -1,]
        }
        
    }

    return(data)

}


