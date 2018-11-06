

# description: code to plot out PWM scores and RNA scores
library(gplots)
library(RColorBrewer)
library(reshape2)

library(grid)
library(gridGraphics)
library(gridExtra)

library(rhdf5)

args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]

make_heatmap <- function(data, data_max) {

    # for the rowside colors
    # make sure the values lie between 0-1
    pal <- colorRamp(brewer.pal(9, "RdBu"))
    #pal <- colorRamp(c("white", "black"))
    data_max <- 1 - data_max
    rowsidecolors <- rgb(pal(data_max), maxColorVal=256)
    
    # color palette
    #my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
    my_palette <- colorRampPalette(brewer.pal(9, "YlGnBu"))(49)
    my_palette <- colorRampPalette(brewer.pal(9, "Blues"))(49)
    
    # breaks
    color_granularity <- 50
    data_melted <- melt(data)
    my_breaks <- seq(
        min(data_melted$value),
        #0,
        #quantile(data_melted$value, 0.01),
        max(data_melted$value),
        #quantile(data_melted$value, 0.90),
        length.out=color_granularity)

    
    # grid
    mylmat = rbind(c(0,3,0),c(1,2,0),c(4,5,0))
    mylwid = c(0.10,1.5,0.5)
    mylhei = c(0.25,4,0.75) # 0.5

    # plot
    heatmap.2(
        as.matrix(data),
        Rowv=FALSE,
        Colv=FALSE,
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
        #breaks=my_breaks,
        useRaster=FALSE,

        RowSideColors=rowsidecolors
        )

}

# grab grob fn
grab_grob <- function(fn) {
    grid.echo(fn)
    #grid.grabExpr(grid.echo(fn))
    grid.grab()
}



main_group <- "cor_filt"

# find groups
metadata <- h5ls(h5_file)
groups <- metadata$name[metadata$group == paste("/", main_group, sep="")]

# run for each group

for (group_i in 1:length(groups)) {

    group <- paste("/", main_group, "/", groups[group_i], sep="")
    print(group)

    # extract relevant data
    group_data <- h5read(h5_file, group, read.attributes=TRUE)

    # plot 3 heatmaps

    # TODO here can select a correlation cutoff as desired
    
    
    # ===================
    # 1) pwm scores (w max vector)
    # ===================
    pwm_scores <- aperm(group_data$pwm_patterns)
    rownames(pwm_scores) <- attr(group_data, "pwm_names")

    # min/max norm
    pwm_scores_max <- apply(pwm_scores, 1, max)
    pwm_scores_min <- apply(pwm_scores, 1, min)
    pwm_scores <- (pwm_scores - pwm_scores_min) / (pwm_scores_max - pwm_scores_min)

    # and adjust the max vector
    pwm_unit_max <- pwm_scores_max / max(pwm_scores_max)
    pwm_unit_max <- pwm_unit_max / 2 + 0.5

    
    pdf(paste(groups[group_i], ".pwm_scores.pdf", sep=""))
    make_heatmap(pwm_scores, pwm_unit_max)
    dev.off()

    fn1 <- function() make_heatmap(pwm_scores, pwm_unit_max)
    
    # ===================
    # 2) RNA scores (w max vector)
    # ===================
    rna_scores <- aperm(group_data$rna_patterns)
    rownames(rna_scores) <- attr(group_data, "hgnc_ids")

    # min/max norm
    rna_scores_max <- apply(rna_scores, 1, max)
    rna_scores_min <- apply(rna_scores, 1, min)
    rna_scores <- (rna_scores - rna_scores_min) / (rna_scores_max - rna_scores_min)

    # and adjust the max vector
    rna_unit_max <- rna_scores_max / max(rna_scores_max)
    rna_unit_max <- rna_unit_max / 2 + 0.5
    
    pdf(paste(groups[group_i], ".rna_scores.pdf", sep=""))
    make_heatmap(rna_scores, rna_unit_max)
    dev.off()

    fn2 <- function() make_heatmap(rna_scores, rna_unit_max)

    # ===================
    # 3) correlation plot
    # ===================
    correlation_scores <- aperm(group_data$correlations)
    rownames(correlation_scores) <- attr(group_data, "hgnc_ids")

    correlation_scores <- (correlation_scores + 1) / 2
    
    pdf(paste(groups[group_i], ".pwm_rna_cor.pdf", sep=""))
    make_heatmap(rna_scores, correlation_scores)
    dev.off()

    fn3 <- function() make_heatmap(rna_scores, correlation_scores)

    # now try plot all together
    grob_list <- list(
        "1"=grab_grob(fn1),
        "2"=grab_grob(fn3),
        "3"=grab_grob(fn2))
    
    pdf(
        file=paste(groups[group_i], ".all.pdf", sep=""),
        height=7, width=21, onefile=FALSE)
    grid.newpage()
    grid.arrange(grobs=grob_list, nrow=1, ncol=3, clip=FALSE)
    dev.off()
    
}











