#!/usr/bin/env Rscript

# description: plot all adjacency matrices for overlaps
# in a consistent way
library(ggplot2)
library(reshape2)


my_hclust <- function(data) {
    #hc <- hclust(data, method="ward.D2")
    hc <- hclust(data)
    return(hc)
}


# args
args <- commandArgs(trailingOnly=TRUE)
prefix <- args[1]
go_filter <- args[2]
summary_files <- args[3:length(args)]

# go through summary files
for (summary_idx in 1:length(summary_files)) {
    summary_file <- summary_files[summary_idx]
    
    # read in file
    summary_prefix <- strsplit(summary_file, "/")[[1]]
    summary_prefix <- summary_prefix[8]
    summary_data <- read.table(summary_file, sep="\t", row.names=1, header=TRUE)

    # filter for GO terms
    if (go_filter == "TRUE") {
        summary_data <- summary_data[summary_data$GO_terms == 1,]
    }
    
    # simplify down
    summary_data <- summary_data[,c("nodes", "region_num")]

    # make sure non-redundant
    summary_data <- aggregate(
        summary_data$region_num,
        by=list(nodes=summary_data$nodes),
        FUN=sum, na.rm=TRUE)

    # split node list to make two columns
    summary_tmp <- strsplit(as.character(summary_data$nodes), ",")
    summary_tmp <- data.frame(do.call(rbind, summary_tmp))
    colnames(summary_tmp) <- c("pwm1", "pwm2")
    summary_tmp$region_num <- summary_data$x
    summary_data <- summary_tmp
    
    # make symmetric
    symmetric_tmp <- data.frame(summary_data)
    symmetric_tmp$pwm_tmp <- symmetric_tmp$pwm1
    symmetric_tmp$pwm1 <- symmetric_tmp$pwm2
    symmetric_tmp$pwm2 <- symmetric_tmp$pwm_tmp
    symmetric_tmp$pwm_tmp <- NULL
    summary_data <- rbind(summary_data, symmetric_tmp)
    
    # and make non-redundant again
    summary_data <- aggregate(
        summary_data$region_num,
        by=list(pwm1=summary_data$pwm1, pwm2=summary_data$pwm2),
        FUN=sum, na.rm=TRUE)
    summary_data$region_num <- summary_data$x
    summary_data$x <- NULL
    
    # set up ordering 
    summary_unmelted <- dcast(
        summary_data, formula = pwm1 ~ pwm2, fun.aggregate=sum, value.var="region_num")
    rownames(summary_unmelted) <- summary_unmelted$pwm1
    summary_unmelted$pwm1 <- NULL
    hc <- my_hclust(dist(summary_unmelted))
    ordering <- hc$order
    ordering <- colnames(summary_unmelted)[ordering]
    summary_data$pwm1 <- factor(summary_data$pwm1, levels=ordering)
    summary_data$pwm2 <- factor(summary_data$pwm2, levels=rev(ordering))

    # plot
    plot_file <- paste(prefix, summary_prefix, "pdf", sep=".")
    print(plot_file)
    ggplot(summary_data, aes(x=pwm1, y=pwm2)) +
        geom_point(shape=21, stroke=0.115, aes(size=region_num)) +
        labs(x="Motif", y="Motif", title="Significant overlaps") + 
        theme_bw() +
        theme(
            aspect.ratio=1,
            text=element_text(family="ArialMT"),
            plot.title=element_text(size=8, margin=margin(b=0)),
            plot.margin=margin(5,1,1,1),
            panel.background=element_blank(),
            panel.border=element_rect(size=0.115),
            panel.grid=element_blank(),
            axis.title=element_text(size=6),
            axis.title.x=element_text(margin=margin(0,0,0,0)),
            axis.title.y=element_text(margin=margin(0,0,0,0)),
            axis.text.y=element_text(size=6),
            axis.text.x=element_text(size=6, angle=90),
            axis.line=element_line(color="black", size=0.115, lineend="square"),
            axis.ticks=element_line(size=0.115),
            axis.ticks.length=unit(0.01, "in"),
            legend.background=element_blank(),
            legend.box.background=element_blank(),
            legend.margin=margin(0,0,0,0),
            legend.key.size=unit(0.05, "in"),
            legend.box.margin=margin(0,0,0,0),
            legend.box.spacing=unit(0.05, "in"),
            legend.title=element_blank(),
            legend.text=element_text(size=6)) +
        scale_size_continuous(range=c(0,2)) +
    
    ggsave(plot_file, height=3.25, width=3.25, units="in", useDingbats=FALSE)
    
}


