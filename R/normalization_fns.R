#!/usr/bin/env/ Rscript


# description: module for easy management of
# possible normalizations


normalize_rows <- function(data) {
    rowmax <- apply(data, 1, function(x) max(x))
    data_norm <- data / rowmax
    data_norm[is.na(data_norm)] <- 0
    return(data_norm)

}



