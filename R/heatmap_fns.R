#!/usr/bin/env/ Rscript


library(gplots)
library(RColorBrewer)
library(reshape2)

# description: module for easy management of
# heatmaps


get_ordered_sample <- function(data) {
    plottable_nrow <- 1500 #2000 # important - useRaster in heatmap!
    if (nrow(data) > plottable_nrow) {
        skip_int <- as.integer(nrow(data) / plottable_nrow)
        ordered_sample_indices <- seq(1, nrow(data), skip_int)

        if (length(dim(data)) == 3) {
            data <- data[ordered_sample_indices,,]
        } else {
            data <- data[ordered_sample_indices,]
        }
    }
    
    return(data)
}


make_heatmap <- function(
    data,
    out_pdf_file,
    cluster_columns,
    is_signal,
    large_view,
    use_raster) {
    
    # color palette
    my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)

    # if signal, adjust the color scale
    if (is_signal) {
        color_granularity <- 50
        data_melted <- melt(data)
        my_breaks <- seq(
            quantile(data_melted$value, 0.01),
            quantile(data_melted$value, 0.90),
            length.out=color_granularity)
    } else {
        my_breaks <- NULL
    }
    
    # grid
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(2,6,2)
    mylhei = c(0.5,12,1.5)

    # adjust width and colnames
    if (large_view) {
        labCol <- colnames(data)
        width <- 30
    } else {
        labCol <- ""
        width <- 10
    }
    
    # plot
    pdf(out_pdf_file, height=18, width=width)
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=cluster_columns,
        dendrogram="none",
        trace="none",
        density.info="none",
        
        labRow="",
        labCol=labCol,
        srtCol=45,
        cexCol=0.75,
        
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(9.1,1,2.1,1),
            mgp=c(3,2,0),
            cex.axis=2.0,
            font.axis=2),
        key.xtickfun=function() {
            breaks <- pretty(parent.frame()$breaks)
            #breaks <- breaks[c(1,length(breaks))]
            list(at = parent.frame()$scale01(breaks),
                 labels = breaks)},
        
        margins=c(3,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        
        col=my_palette,
        breaks=my_breaks,
        useRaster=use_raster)
    dev.off()

}



make_agg_heatmap <- function(
    data,
    out_pdf_file,
    cluster_columns) {
    
    # color palette
    #my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)

    # grid
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(0.25,1.5,0.5)
    mylhei = c(0.25,4,0.75) # 0.5

    # plot
    pdf(out_pdf_file, height=7, width=3, family="ArialMT")
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=cluster_columns,
        dendrogram="none",
        trace="none",
        density.info="none",
        colsep=1:ncol(data),
        rowsep=1:nrow(data),
        sepcolor="black",
        sepwidth=c(0.01,0.01),
        cexCol=1.25,
        cexRow=1.25,
        srtCol=45,
        
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(3.1,1,3.1,1),
            mgp=c(3,1,0),
            cex.axis=1.0,
            font.axis=2),
        key.xtickfun=function() {
            breaks <- pretty(parent.frame()$breaks)
            #breaks <- breaks[c(1,length(breaks))]
            list(at = parent.frame()$scale01(breaks),
                 labels = breaks)},
        
        margins=c(1,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        col=my_palette,
        useRaster=FALSE)
    dev.off()

}
