#!/usr/bin/env Rscript

# description: plot all grammar information together
library(gplots)
library(RColorBrewer)
library(reshape2)
library(ggplot2)

library(grid)
library(gridGraphics)
library(gridExtra)

library(extrafont)

args <- commandArgs(trailingOnly=TRUE)
motif_presence_file <- args[1]
atac_file <- args[2]
rna_file <- args[3]
go_file <- args[4]
plot_file <- args[5]

my_hclust <- function(data) {
    hc <- hclust(data, method="ward.D2")
    return(hc)
}

make_heatmap <- function(data, colv, my_palette) {

    
    # color palette
    #my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    #my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)
    #my_palette <- colorRampPalette(brewer.pal(9, "Blues"))(49)
    #my_palette <- rev(colorRampPalette(brewer.pal(9, "RdBu"))(49))
    
    # breaks
    if (FALSE) {
        color_granularity <- 50
        data_melted <- melt(data)
        my_breaks <- seq(
            min(data_melted$value),
                                        #0,
                                        #quantile(data_melted$value, 0.01),
            max(data_melted$value),
                                        #quantile(data_melted$value, 0.90),
            length.out=color_granularity)
    }
    
    # grid
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(1.0,7,1.0) # 0.5
    mylhei = c(0.25,4,0.75) # 0.5

    # plot
    heatmap.2(
        as.matrix(data),
        hclustfun=my_hclust,
        Rowv=FALSE,
        Colv=colv,
        dendrogram="none",
        trace="none",
        density.info="none",
        colsep=1:ncol(data),
        rowsep=1:nrow(data),
        sepcolor="black",
        sepwidth=c(0.01,0.01),

        labRow=rep("", nrow(data)),
        #labCol=labCol,
        cexCol=1.25,
        cexRow=1,
        srtCol=60,

        key=TRUE,
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
        #breaks=my_breaks,
        useRaster=FALSE,

        #RowSideColors=rowsidecolors
        )

}


make_grammar_plot <- function(data) {
    # use ggplot2 to make grammar plot
    data$tfs <- rownames(data)
    data_melted <- melt(data)
    data_melted <- data_melted[data_melted$value != 0,]
    data_melted$tfs <- factor(data_melted$tfs, levels=rev(rownames(data)))
    data_melted$variable <- factor(data_melted$variable, levels=colnames(data))
    
    p <- ggplot(data_melted, aes(x=variable, y=tfs)) + geom_point(size=3) +
        geom_line(aes(group=tfs)) +
        labs(x="Motif presence", y="Grammars") + 
        theme_bw() +
            theme(
                axis.text.x=element_text(size=14, angle=60, hjust=1),
                axis.text.y=element_text(size=10))        
    return(p)
    
}

make_go_plot <- function(data) {
    data$ids <- rownames(data)
    data_melted <- melt(data)
    data_melted <- data_melted[data_melted$value != 0,]
    data_melted$ids <- factor(data_melted$ids, levels=rev(rownames(data)))
    go_term_levels <- gsub(".", " ", colnames(data), fixed=TRUE)
    go_terms <- gsub(".", " ", data_melted$variable, fixed=TRUE)
    data_melted$variable <- factor(go_terms, levels=go_term_levels)
    
    p <- ggplot(data_melted, aes(x=variable, y=ids)) +
        geom_point(aes(size=value)) +
        labs(y=NULL, x="GO terms") +
        theme_bw() +
        theme(
            axis.text.x=element_text(size=14, angle=60, hjust=1),
            axis.text.y=element_blank())       
    return(p)



}

# grab grob fn
grab_grob <- function(fn) {
    #grid.echo(fn)
    grid.grabExpr(grid.echo(fn))
    #grid.grab()
}


# read in data
motif_presence <- read.table(motif_presence_file, header=TRUE, row.names=1)
atac <- read.table(atac_file, header=TRUE, row.names=1)
rna <- read.table(rna_file, header=TRUE, row.names=1)
go <- read.table(go_file, header=TRUE, row.names=1, sep="\t")


# choose new ordering, by ATAC
if (TRUE) {
    my_dist <- dist(cbind(motif_presence, atac, rna))
    hc <- my_hclust(my_dist)
    hc_dend <- as.dendrogram(hc)
    hc_dend[[2]] <- rev(hc_dend[[2]])
    ordering <- order.dendrogram(hc_dend)
    #ordering <- hc$order
} else {
    library(cluster)
    hc <- diana(cbind(motif_presence, atac, rna))
    #hc <- diana(atac)
    ordering <- hc$order
}


# reorder
if (TRUE) {
    motif_presence <- motif_presence[ordering,]
    atac <- atac[ordering,]
    rna <- rna[ordering,]
    go <- go[ordering,]
}

# normalize
atac <- atac - apply(atac, 1, min)
colnames(atac) <- c("d0.0", "d1.0", "d1.5", "d2.0", "d2.5", "d3.0", "d4.5", "d5.0", "d6.0")

rna <- rna  - apply(rna, 1, min)
colnames(rna) <- c("d0.0", "d1.0", "d1.5", "d2.0", "d2.5", "d3.0", "d4.5", "d5.0", "d6.0")

# set up plotting fns for grob
blue_palette <- colorRampPalette(brewer.pal(9, "Blues"))(49)
#fn1 <- function() make_heatmap(motif_presence, FALSE, blue_palette)
fn1 <- function() make_grammar_plot(motif_presence)
#rdbu_palette <- rev(colorRampPalette(brewer.pal(9, "RdBu"))(49))
fn2 <- function() make_heatmap(atac, FALSE, blue_palette)
purples_palette <- colorRampPalette(brewer.pal(9, "Purples"))(49)
fn3 <- function() make_heatmap(rna, FALSE, purples_palette)
#fn4 <- function() make_go_plot(go)

grob_list <- list(
    "1"=make_grammar_plot(motif_presence),
    #"1"=grab_grob(fn1),
    "2"=grab_grob(fn2),
    "3"=grab_grob(fn3),
    "4"=make_go_plot(go))

test_plot_file <- "test.pdf"
pdf(
    file=test_plot_file,
    height=12, width=21, onefile=FALSE, family="ArialMT", useDingbats=FALSE)
grid.newpage()
grid.arrange(grobs=grob_list, nrow=1, ncol=4, heights=c(10), widths=c(10, 1.5, 1.5, 8), clip=FALSE)
dev.off()

embed_fonts(test_plot_file, outfile=plot_file)
