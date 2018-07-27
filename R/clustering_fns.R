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


# fn for ordering by clusters, but only those in cluster ids
order_by_masked_clusters <- function(data, clusters, cluster_ids) {

    # check if soft (2d) or hard (1D)
    if (length(dim(clusters)) > 1) {
        # soft
        data <- apply(data, 2, sort, decreasing=FALSE)

        # filter
        present_clusters <- clusters[,cluster_ids]
        in_any_present_cluster <- apply(present_clusters, 1, any)
        data <- data[in_any_present_cluster,]
        
    } else {
        # hard
        if (length(dim(data)) == 3) {
            data <- data[order(clusters),,]
            data <- data[clusters != -1,,]

            # and then only keep those in clusters
            data <- data[clusters %in% cluster_ids,,]
            
        } else {
            data <- data[order(clusters),]
            data <- data[clusters != -1,]

            # and then only keep those in clusters
            data <- data[clusters %in% cluster_ids,]
        }
        
    }

    return(data)

}
