#!/usr/bin/env Rscript

# description: diagnostic plot of motif x position
# provides a view into where motifs are marked
# useful for checking motif redundancy/overlap

library(gplots)
library(RColorBrewer)

library(grid)
library(gridGraphics)

grab_grob <- function(fn) {
    grid.echo(fn)
    grid.grab()
}



args <- commandArgs(trailingOnly=TRUE)
heatmap1_file <- args[1] # motif x pos
heatmap2_file <- args[2] # max motif val

# read in data
heatmap1_data <- read.table(heatmap1_file, header=TRUE, row.names=1)
heatmap2_data <- read.table(heatmap2_file, header=TRUE, row.names=1)

# adjust as needed
colnames(heatmap2_data) <- "val1"
heatmap2_data$val2 <- heatmap2_data$val1
heatmap2_data$index <- seq(1,nrow(heatmap2_data))

# set up ordering
heatmap2_data <- heatmap2_data[order(-heatmap2_data$val1),]
ordering <- heatmap2_data$index
heatmap2_data$index <- NULL

heatmap1_data <- heatmap1_data[ordering,]

# add in breaks, use max across values for color scale
max_val <- max(abs(heatmap1_data))
break_point <- 0.1 * max_val
step <- 0.01 * max_val
my_palette <- rev(colorRampPalette(brewer.pal(9, "RdBu"))(49))
breaks <- c(seq(-max_val, -break_point, length=10),
            seq(-break_point+step, break_point-step, length=30),
            seq(break_point, max_val, length=10))

# set up plot grid


# plot heatmap
plot_heatmap1 <- function(data) {
    # set up sizing
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(1,16,1)
    mylhei = c(0.5,8,1.0)

    # heatmap
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=FALSE,
        dendrogram="none",
        trace="none",
        density.info="none",
        labCol="",
        cexRow=0.5,
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(4.1,0,0.1,0),
            mgp=c(3,2,0),
            cex.axis=2.0,
            font.axis=2),
        margins=c(3,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        col=my_palette,
        breaks=breaks)
}

plot_heatmap2 <- function(data) {
    # set up sizing
    mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
    mylwid = c(0.1,0.1,2)
    mylhei = c(0.5,8,1.0)

    # heatmap
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=FALSE,
        dendrogram="none",
        trace="none",
        density.info="none",
        labCol="",
        cexRow=0.5,
        key=FALSE,
        margins=c(3,0),
        lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
        col=my_palette,
        breaks=breaks)
}

# now set up viewports etc

fn <- function() plot_heatmap1(heatmap1_data)
g1 <- grab_grob(fn)

fn <- function() plot_heatmap2(heatmap2_data)
g2 <- grab_grob(fn)

heatmap_file <- sub("txt", "pdf", heatmap1_file)
pdf(heatmap_file, height=16, width=32)

grid.newpage()
lay <- grid.layout(nrow=1, ncol=2, widths=c(16, 1))
pushViewport(viewport(layout=lay))

grid.draw(editGrob(
    g1,
    vp=viewport(
        layout.pos.row=1,
        layout.pos.col=1, clip=FALSE)))

grid.draw(editGrob(
    g2,
    vp=viewport(
        layout.pos.row=1,
        layout.pos.col=2, clip=FALSE)))

upViewport()

dev.off()
