#!/usr/bin/env Rscript

# description - code to plot out synergy scores
library(rhdf5)
library(reshape2)
library(ggplot2)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
dist_key <- args[2]
logits_key <- args[3]
group_key <- args[4]
avgs_key <- args[5]
out_prefix <- args[6]

# read in data
distances <- h5read(h5_file, dist_key)
logits <- h5read(h5_file, logits_key)
groups <- h5read(h5_file, group_key)
avgs <- h5read(h5_file, avgs_key, read.attributes=TRUE)
num_tasks <- dim(logits)[1]

print(levels(factor(groups)))

# run for every tasks
for (task_idx in 1:num_tasks) {

    next
    
    # set up task data
    task_prefix <- paste(out_prefix, ".taskidx-", task_idx-1, sep="")
    print(task_prefix)
    task_data <- data.frame(
        dists=distances,
        logits=aperm(logits)[,task_idx],
        groups=aperm(groups)[,1])

    # get the avg data
    task_avgs <- data.frame(aperm(avgs)[,,task_idx])
    colnames(task_avgs) <- attr(avgs, "dists")
    task_avgs$groups <- attr(avgs, "grammar.string")
    task_avgs <- melt(task_avgs, id.vars="groups")
    task_avgs$variable <- as.numeric(as.character(task_avgs$variable))
    
    # seprate odds (embedded) from evens (non-embedded) and subtract
    task_data_embed <- task_data[seq(1,nrow(task_data)-1,2),]
    task_data_orig <- task_data[seq(2,nrow(task_data),2),]
    task_data_embed$diff <- task_data_embed$logits - task_data_orig$logits
    task_data <- task_data_embed
    #task_data <- task_data_orig

    # debug
    #task_data <- task_data[task_data$groups == "HCLUST-110_GRHL2.UNK.0.A+;HCLUST-105_ATF4.UNK.0.A+",]
    #task_avgs <- task_avgs[task_avgs$groups == "HCLUST-110_GRHL2.UNK.0.A+;HCLUST-105_ATF4.UNK.0.A+",]
    
    # plot with distance
    plot_file <- paste(task_prefix, ".dist_x_logFC.pdf", sep="")
    p <- ggplot(task_data, aes(x=dists, y=diff, colour=factor(groups))) +
        geom_point(alpha=0.5, size=0.5) +
        geom_line(data=task_avgs, aes(x=variable, y=value, colour=factor(groups))) +
        #geom_point(position="jitter", aes(colour=factor(groups)), alpha=0.5) +
        #geom_smooth(se=FALSE, aes(group=factor(groups), colour=factor(groups))) +
        #stat_summary(aes(y=logits, group=factor(groups), colour=factor(groups)), fun.y=mean, geom="line") + 
        labs(y="logit", x="PWM distance (bp)") +
        theme_bw() +
        theme(
            axis.title=element_text(size=32),
            axis.text.y=element_text(size=16),
            axis.text.x=element_text(size=12, angle=30, hjust=1),
            legend.title=element_text(size=32),
            legend.text=element_text(size=12))
            #legend.position="none")
    ggsave(plot_file, height=7, width=14)

}

# run for every syntax
dist_window <- 10
syntaxes <- levels(factor(groups))
num_syntaxes <- length(syntaxes)
print(num_syntaxes)
avgs <- h5read(h5_file, "simul.calcs/simul.scores.smooth.high", read.attributes=TRUE)
for (syntax_idx in 1:num_syntaxes) {

    # set up syntax data
    syntax_prefix <- paste(out_prefix, ".syntaxidx-", syntax_idx-1, sep="")
    group <- syntaxes[syntax_idx]
    print(group)

    # select distance
    syntax_avgs <- data.frame(aperm(avgs)[syntax_idx,,])
    rownames(syntax_avgs) <- attr(avgs, "dists")
    syntax_avgs$max <- apply(syntax_avgs, 1, max)
    dist_idx <- which.max(syntax_avgs$max)
    dist <- as.numeric(rownames(syntax_avgs)[dist_idx])
    dist_min <- max(dist - dist_window, 0)
    dist_max <- min(
        dist + dist_window,
        as.numeric(rownames(syntax_avgs)[nrow(syntax_avgs)]))
    print(c(dist_min, dist_max))
    
    # get the data points
    raw_data <- data.frame(logits=aperm(logits))
    raw_data <- raw_data[,c(1,2,3,4,5,6,7,10,11,13)] # GGR specific!
    raw_data$dists <- distances
    raw_data$syntax <- aperm(groups)[,1]
    #raw_data <- raw_data[raw_data$dists == dist,]
    raw_data <- raw_data[raw_data$dists <= dist_max,]
    raw_data <- raw_data[raw_data$dists >= dist_min,]
    raw_data <- raw_data[raw_data$syntax == group,]
    raw_data$syntax <- NULL
    #raw_data$dists <- NULL
    raw_data$group <- rep(1, nrow(raw_data))
    raw_data$group[seq(1,nrow(raw_data)) %% 2 == 0] <- 0
    raw_data$idx <- 1:nrow(raw_data)
    raw_data$dists_alpha <- dist_max - raw_data$dists # invert for color purposes

    # melt
    raw_data_melt <- melt(raw_data, id.vars=c("group","idx", "dists", "dists_alpha"))
    raw_data_melt$viol <- paste(raw_data_melt$variable, raw_data_melt$group, sep=".")
    raw_data_melt$group <- as.character(raw_data_melt$group)
    raw_data_melt$group[raw_data_melt$group == "0"] <- "Background"
    raw_data_melt$group[raw_data_melt$group != "Background"] <- "Embedded"
    
    # plot 
    plot_file <- paste(syntax_prefix, ".timepoint_x_logit.pdf", sep="")
    p <- ggplot(raw_data_melt, aes(x=variable, y=value, group=idx, colour=factor(group))) +
        geom_line(alpha=0.1) +
        geom_violin(aes(group=viol), alpha=1.0) + 
        geom_point(aes(group=viol), position=position_jitterdodge(), alpha=0.5) +
        labs(y="logit", x="task") +
        theme_bw() +
        theme(
            axis.title=element_text(size=32),
            axis.text.y=element_text(size=16),
            axis.text.x=element_text(size=12, angle=30, hjust=1),
            legend.title=element_text(size=32),
            legend.text=element_text(size=12))
    ggsave(plot_file, height=7, width=14)
    
    # do again but the logFC
    raw_data_melt_embed <- raw_data_melt[raw_data_melt$group == "Embedded",]
    raw_data_melt_background <- raw_data_melt[raw_data_melt$group == "Background",]
    raw_data_melt_embed$diff <- raw_data_melt_embed$value - raw_data_melt_background$value

    # plot
    plot_file <- paste(syntax_prefix, ".timepoint_x_logFC.pdf", sep="")
    p <- ggplot(raw_data_melt_embed, aes(x=variable, y=diff, group=idx, colour=factor(group), alpha=dists_alpha)) +
        geom_line(alpha=0.1) +
        geom_violin(aes(group=viol), alpha=1.0) + 
        geom_point(aes(group=viol), position=position_jitterdodge()) +
        labs(y="logit", x="task") +
        theme_bw() +
        theme(
            axis.title=element_text(size=32),
            axis.text.y=element_text(size=16),
            axis.text.x=element_text(size=12, angle=30, hjust=1),
            legend.title=element_text(size=32),
            legend.text=element_text(size=12))
    ggsave(plot_file, height=7, width=14)

}

summary(warnings())
