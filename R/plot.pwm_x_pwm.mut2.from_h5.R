#!/usr/bin/env Rscript

# description: plot summary of grammar and mutagenesis
# NOTE: rhdf5 transposes axes!

library(rhdf5)
library(ggplot2)
library(reshape)
library(RColorBrewer)
library(scales)

# helper functions
se <- function(x) sd(x) / sqrt(length(x))


#h5ls(h5_file)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
logits_key <- args[2]
pwm_scores_key <- args[3]
logits_mut_key <- args[4]
pwm_scores_mut_key <- args[5]
mutation_names_key <- args[6]
out_prefix <- args[7]
task_indices <- as.numeric(args[8:length(args)])


get_pwm_scores_melted_mean_and_se <- function(pwm_data, axes, col_names, row_names, horiz_facet, x_shift, normalize) {
    # calculate the mean and se of dataset
    # melt the data and return
    #pwm_data <- aperm(pwm_data, c(3, 2, 1))

    # row normalize
    if (normalize) {
        rowmax_vals <- apply(pwm_data, 3, max)
        pwm_data <- sweep(pwm_data, 3, rowmax_vals, "/")
        pwm_data[is.nan(pwm_data)] <- 0
    } else {
        pwm_data <- pwm_data
    }
    
    # get means
    means <- apply(pwm_data, axes, mean)
    means <- t(data.frame(
        motifs=means,
        row.names=col_names))
    rownames(means) <- row_names
    
    # get standard errors
    standard_errors <- apply(pwm_data, axes, se)
    standard_errors <- t(data.frame(
        motifs=standard_errors,
        row.names=col_names))
    rownames(standard_errors) <- row_names

    # melt
    means_melted <- melt(means)
    colnames(means_melted) <- c("target", "response", "mean")
    means_melted$id <- paste(means_melted$target, means_melted$response, sep="_to_")
    
    se_melted <- melt(standard_errors)
    colnames(se_melted) <- c("target", "response", "se")
    se_melted$id <- paste(se_melted$target, se_melted$response, sep="_to_")
    se_melted$target <- NULL
    se_melted$response <- NULL
    
    # and merge
    summary_melted <- merge(means_melted, se_melted, by="id")

    # add in extra id info (for facet_grid)
    summary_melted$horiz_facet <- rep(horiz_facet, nrow(summary_melted))

    # also pass out an ordering
    ordering_vals <- order(colSums(means))
    map <- setNames(
        1:ncol(means)+x_shift,
        colnames(means)[ordering_vals])
    
    return(list(data_melted=summary_melted, ordering=map))
    
}

get_logits_melted_mean_and_se <- function(logits_data, axes, col_names, row_names, horiz_facet) {
    # calculate the mean and se for logit data
    # melt and return

    # get means
    means <- apply(logits_data, axes, mean)
    means <- t(data.frame(
        logits=means,
        row.names=row_names))
    rownames(means) <- col_names
    
    # get standard errors
    standard_errors <- apply(logits_data, axes, se)
    standard_errors <- t(data.frame(
        logits=standard_errors,
        row.names=row_names))
    rownames(standard_errors) <- col_names

    # melt
    means_melted <- melt(means)
    colnames(means_melted) <- c("response", "target", "mean")
    means_melted$id <- paste(means_melted$target, means_melted$response, sep="_to_")
    
    se_melted <- melt(standard_errors)
    colnames(se_melted) <- c("response", "target", "se")
    se_melted$id <- paste(se_melted$target, se_melted$response, sep="_to_")
    se_melted$target <- NULL
    se_melted$response <- NULL
    
    # and merge
    summary_melted <- merge(means_melted, se_melted, by="id")

    # add in extra id info (for facet_grid)
    summary_melted$horiz_facet <- rep(horiz_facet, nrow(summary_melted))
    
    return(summary_melted)
    
}


ggplot_single_state_map <- function(data_melted, col_names, out_file) {
    # ggplot the map
    p <- ggplot(data_melted, aes(x=response_ordered, y=mean)) +
        facet_grid(target ~ horiz_facet, scales="free", space="free_x", switch="y") +
            
        geom_col( # bar plot for original motif scores
            data=subset(data_melted, horiz_facet=="pwms" & target=="original"),
            aes(y=mean, fill=-mean)) +
        geom_point( # point plot for mutation effects
            data=subset(data_melted, horiz_facet=="pwms" & target!="original"),
            aes(y=mean, color=mean, size=-mean)) +
        geom_errorbar( # and SE bars
            data=subset(data_melted, horiz_facet=="pwms"),
            aes(ymin=mean-2*se, ymax=mean+2*se, color=mean), width=0.05) +
        geom_point( # set up logit spread
            data=subset(data_melted, horiz_facet=="logits"),
            alpha=0.00,
            aes(x=-5, y=0, color=0)) + #x=10*delta, y=0
        geom_point( # set up logit spread
            data=subset(data_melted, horiz_facet=="logits"),
            alpha=0.00,
            aes(x=5, y=0, color=0)) + #x=10*delta, y=0
        geom_point(
            data=subset(data_melted, horiz_facet=="logits"),
            aes(y=0)) + #x=10*delta, y=0
        geom_segment(
            data=subset(data_melted, horiz_facet=="logits"),
            aes(x=0, y=0, xend=response_ordered, yend=0)) + #x=10*delta, y=0
            
        scale_y_continuous(position="right") +
        scale_fill_gradient(high="white", low="steelblue") +

        theme_bw() +
        scale_x_continuous(
            breaks=c(seq(-5, 5, length.out=9), 1:length(col_names)+10),
            labels=c(seq(-5, 5, length.out=9), col_names),
            expand=c(0,0)) +
        theme(
            panel.grid.major.x=element_blank(),
            panel.grid.minor=element_blank(),
            axis.text.x=element_text(size=5, angle=60, hjust=1),
            axis.text.y=element_text(size=5),
            legend.text=element_text(size=5),
            legend.key.size=unit(0.5, "line"),
            legend.spacing=unit(0.5, "line"),
            strip.background=element_blank())
     
    ggsave(out_file, height=1*length(levels(data_melted$target)), width=16)

}


ggplot_summary_map <- function(data_melted, col_names, out_file) {
    # ggplot the map
    fill_max <- max(abs(data_melted$mean))
    fill_min <- min(min(-abs(data_melted$mean)), -1.1)
    
    p <- ggplot(data_melted, aes(x=response_ordered, y=task)) +
        facet_grid(target ~ horiz_facet, scales="free", space="free_x") +

        geom_tile(
            data=subset(data_melted, horiz_facet=="logits"),
            aes(fill=mean),
            colour="white") +
        geom_tile(
            data=subset(data_melted, horiz_facet=="pwms" & target=="original"),
            aes(fill=-mean),
            colour="white") +
        #geom_point(
        #    data=subset(data_melted, horiz_facet=="pwms" & target=="original"),
        #    aes(size=-mean)) +
        geom_point(
            data=subset(data_melted, horiz_facet=="pwms" & target!="original"),
            aes(colour=mean)) +

        #scale_fill_gradient2(low="steelblue", mid="white", high="red", midpoint=0) +
        scale_colour_gradient(low="steelblue", high="white") +    
        scale_fill_gradientn(
            colours=c("steelblue3", "steelblue2", "white", "red"),
            limits=c(fill_min, fill_max),
            values=rescale(c(fill_min, -0.5, 0, fill_max))) +

        theme_bw() +
        scale_x_continuous(
            breaks=c(0, 1:length(col_names)+10),
            labels=c("logits", col_names),
            expand=c(0,0)) +
        theme(
            panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(),
            axis.text.x=element_text(size=5, angle=60, hjust=1),
            axis.text.y=element_text(size=5),
            legend.text=element_text(size=5),
            legend.key.size=unit(0.5, "line"),
            legend.spacing=unit(0.5, "line"),
            strip.background=element_blank())
    
    ggsave(out_file, height=1*length(levels(data_melted$target)), width=16)

}


# set up function to plot individual cell state maps
# returns the melted data - easy to append to final total sum
# i is the ADJUSTED index (1-start)
plot_single_state_map <- function(args, i, out_file, ordering_map) {
    
    # params
    x_shift <- 10
    
    # set up keys
    task_indices <- as.numeric(args[8:length(args)])
    task_idx <- task_indices[i]
    h5_file <- args[1]
    logits_key <- args[2]
    pwm_scores_key <- paste(args[3], ".taskidx-", task_idx, sep="")
    logits_mut_key <- args[4]
    pwm_scores_mut_key <- paste(args[5], ".taskidx-", task_idx, sep="")
    mutation_names_key <- args[6]
    
    # pull out appropriate datasets
    logits <- h5read(h5_file, logits_key)[task_idx+1,,drop=FALSE]
    pwm_scores <- h5read(h5_file, pwm_scores_key, read.attributes=TRUE)
    logits_mut <- h5read(h5_file, logits_mut_key)[,i,]
    pwm_scores_mut <- h5read(h5_file, pwm_scores_mut_key, read.attributes=TRUE)
    mut_names <- attr(pwm_scores_mut, "pwm_mut_names")

    # adjust dims
    dim(pwm_scores) <- c(dim(pwm_scores)[1], 1, dim(pwm_scores)[2])

    # TODO fix this later?
    col_names <- attr(pwm_scores, "pwm_names")
    
    # get means/standard error for weighted pwm scores
    results <- get_pwm_scores_melted_mean_and_se(
        pwm_scores,
        c(1, 2),
        col_names,
        c("original"),
        "pwms",
        x_shift,
        TRUE)
    pwm_scores_melted <- results$data_melted
    #ordering_map <- results$ordering

    # get means/standard error for mutated pwm scores
    results <- get_pwm_scores_melted_mean_and_se(
        pwm_scores_mut,
        c(1, 2),
        col_names,
        mut_names,
        "pwms",
        x_shift,
        FALSE)
    pwm_scores_mut_melted <- results$data_melted
    if (is.null(ordering_map)) {
        ordering_map <- results$ordering
    }
    
    # get means/standard error for logits
    logits_melted <- get_logits_melted_mean_and_se(
        logits,
        c(1),
        c("logits"),
        c("original"),
        "logits")
    
    # get means/standard error for mutated logits
    logits_mut_melted <- get_logits_melted_mean_and_se(
        logits_mut,
        c(1),
        c("logits"),
        mut_names,
        "logits")
    
    # first rbind the pwms together and set up ordering
    pwm_all_melted <- rbind(
        pwm_scores_melted,
        pwm_scores_mut_melted)
    pwm_all_melted$response_ordered <- ordering_map[as.character(pwm_all_melted$response)]
    
    # and rbind the logits, adjust the deltas
    logits_all_melted <- rbind(
        logits_melted,
        logits_mut_melted)
    logits_all_melted$response_ordered <- logits_all_melted$mean
    logits_all_melted$response_ordered[logits_all_melted$target != "original"] <- logits_all_melted$mean[logits_all_melted$target != "original"] +
        logits_all_melted$mean[logits_all_melted$target == "original"]
    logits_all_melted$mean[logits_all_melted$target != "original"] <- logits_all_melted$mean[logits_all_melted$target != "original"] +
        logits_all_melted$mean[logits_all_melted$target == "original"]
    
    # then rbind all together
    task_all_melted <- rbind(
        pwm_all_melted,
        logits_all_melted)

    # adjust factors as needed
    task_all_melted$target <- factor(
        task_all_melted$target,
        levels=c("original", mut_names))

    # and plot
    ordered_col_names <- names(sort(ordering_map[col_names]))
    ggplot_single_state_map(task_all_melted, ordered_col_names, out_file)

    # and return the data with an ordering
    return(list(data_melted=task_all_melted, ordering_map=ordering_map, ordered_col_names=ordered_col_names))

}


for (i in 1:length(task_indices)) {
    
    print(task_indices[i])
    
    # plot single state map
    out_plot <- paste(out_prefix, ".taskidx-", task_indices[i], ".pdf", sep="")
    
    # add to larger melted data with extra timepoint column
    if (i == 1) {
        results <- plot_single_state_map(args, i, out_plot, NULL)
        task_all_melted <- results$data_melted
        task_all_melted$task <- rep(task_indices[i], nrow(task_all_melted))

        data_all_melted <- task_all_melted
        ordering_map <- results$ordering_map
        ordered_col_names <- results$ordered_col_names
    } else {
        results <- plot_single_state_map(args, i, out_plot, ordering_map)
        task_all_melted <- results$data_melted
        task_all_melted$task <- rep(task_indices[i], nrow(task_all_melted))
        
        data_all_melted <- rbind(data_all_melted, task_all_melted)
    }
    
}

# adjust the ordering appropriately
data_all_melted$response_ordered <- ordering_map[as.character(data_all_melted$response)]
data_all_melted$response_ordered[data_all_melted$response == "logits"] <- 0
data_all_melted$response <- factor(data_all_melted$response, levels=c("logits", ordered_col_names), ordered=TRUE)

# adjust tasks to drop missing tasks
data_all_melted$task <- factor(data_all_melted$task, levels=rev(sort(unique(data_all_melted$task))))

# and plot full summary state map
summary_plot_file <- paste(out_prefix, ".summary.pdf", sep="")
ggplot_summary_map(data_all_melted, ordered_col_names, summary_plot_file)


quit()






















mut_names <- c("SMAD3", "TFAP2B")








# TODO - combine all this into a function
# then can call multiple times and also append things properly
# finally take the final output and make the plot


# keys needed: logits, dmim-scores, pwm-scores, delta logits, mutation names
# with corresponding task indices, adjusted for R (1-start not 0-start)

# read in data
data <- h5read(h5_file, dataset_key, read.attributes=TRUE)#[,2,]
pwm_data <- h5read(h5_file, "pwm-scores.taskidx-0", read.attributes=TRUE)
logits_data <- h5read(h5_file, "delta_logits", read.attributes=TRUE)[,1,]

logits_orig_data <- h5read(h5_file, "logits", read.attributes=TRUE)

print(dim(logits_orig_data))

quit()

print(dim(logits_data))

# organize the logits, add fake data in
logits_means <- apply(logits_data, 1, mean)
logits_means <- t(data.frame(
    logits=logits_means,
    row.names=mut_names))

rownames(logits_means) <- "a"
logits_means <- rbind(logits_means, rep(-1, ncol(logits_means)))
rownames(logits_means)[nrow(logits_means)] <- "b"
logits_means <- rbind(logits_means, rep(1, ncol(logits_means)))
rownames(logits_means)[nrow(logits_means)] <- "c"

#print(logits_means)
#quit()

logits_se <- apply(logits_data, 1, se)
logits_se <- t(data.frame(
    logits=logits_se,
    row.names=mut_names))

rownames(logits_se) <- "a"
logits_se <- rbind(logits_se, rep(-1, ncol(logits_se)))
rownames(logits_se)[nrow(logits_se)] <- "b"
logits_se <- rbind(logits_se, rep(1, ncol(logits_se)))
rownames(logits_se)[nrow(logits_se)] <- "c"

# organize the pwm scores
print(dim(pwm_data))
pwm_scores_means <- apply(pwm_data, 1, mean)
pwm_scores_means <- t(data.frame(
    motifs=pwm_scores_means,
    row.names=attr(pwm_data, "pwm_names")))

pwm_scores_se <- apply(pwm_data, 1, se)
pwm_scores_se <- t(data.frame(
    motifs=pwm_scores_se,
    row.names=attr(pwm_data, "pwm_names")))

# organize the mutational response data
data <- aperm(data, c(3, 2, 1))

data_mut_means <- apply(data, c(2, 3), mean)
colnames(data_mut_means) <- attr(pwm_data, "pwm_names")
rownames(data_mut_means) <- c("SMAD3", "TFAP2B")

data_mut_sd <- apply(data, c(2, 3), se)
colnames(data_mut_sd) <- attr(pwm_data, "pwm_names")
rownames(data_mut_sd) <- c("SMAD3", "TFAP2B")

# ===============


# order by strongest effect
ordering_vals <- order(colSums(data_mut_means))
#data_mut_means <- rbind(data_mut_means, ordering_vals)
#rownames(data_mut_means)[nrow(data_mut_means)] <- "order"


map <- setNames(
    1:ncol(data_mut_means)+10,
    colnames(data_mut_means)[ordering_vals])


data_mut_means <- data_mut_means[,order(colSums(data_mut_means))]
print(colnames(data_mut_means))

print(ordering_vals)



# ==================

# ggplot2 version
library(ggplot2)
library(reshape)

data_melted <- melt(data_mut_means)
print(head(data_melted))

colnames(data_melted) <- c("target", "response", "delta")
data_melted$id <- paste(data_melted$target, data_melted$response, sep="_to_")

# set up correct x
data_melted$x <- map[as.character(data_melted$response)]


# melt the sd also and merge in
sd_melted <- melt(data_mut_sd)
colnames(sd_melted) <- c("target", "response", "sd")
sd_melted$id <- paste(sd_melted$target, sd_melted$response, sep="_to_")
sd_melted$target <- NULL
sd_melted$response <- NULL
data_melted <- merge(data_melted, sd_melted, by="id")

# melt the pwm scores and merge in
pwm_means_melted <- melt(pwm_scores_means)
colnames(pwm_means_melted) <- c("target", "response", "delta")
pwm_means_melted$id <- paste(pwm_means_melted$target, pwm_means_melted$response, sep="_to_")
pwm_means_melted$target <- NULL
pwm_means_melted$response <- NULL

pwm_se_melted <- melt(pwm_scores_se)
colnames(pwm_se_melted) <- c("target", "response", "sd")
pwm_se_melted$id <- paste(pwm_se_melted$target, pwm_se_melted$response, sep="_to_")

pwm_full_melted <- merge(pwm_means_melted, pwm_se_melted, by="id")
pwm_full_melted$x <- map[as.character(pwm_full_melted$response)]

# and rbind all together
data_melted <- rbind(data_melted, pwm_full_melted)

# melt the logits and merge in
logit_means_melted <- melt(logits_means)
colnames(logit_means_melted) <- c("response", "target", "delta")
logit_means_melted$id <- paste(logit_means_melted$target, logit_means_melted$response, sep="_to_")
logit_means_melted$target <- NULL
logit_means_melted$response <- NULL

logit_se_melted <- melt(logits_se)
colnames(logit_se_melted) <- c("response", "target", "sd")
logit_se_melted$id <- paste(logit_se_melted$target, logit_se_melted$response, sep="_to_")

logit_full_melted <- merge(logit_means_melted, logit_se_melted, by="id")
#logit_full_melted$response <- factor(
#    logit_full_melted$response, levels=c(1, 2, 3), ordered=TRUE)

print(logit_full_melted$response)

#logit_full_melted$response <- factor(
#    logit_full_melted$response, levels=rev(rownames(logits_means)), ordered=TRUE)

#print(logit_full_melted$response)

logit_full_melted$x <- logit_full_melted$delta


#logit_full_melted$tmp <- logit_full_melted$response
#logit_full_melted$response <- logit_full_melted$delta
#logit_full_melted$delta <- logit_full_melted$tmp
#logit_full_melted$delta <- rep(0, nrow(logit_full_melted))
#logit_full_melted$tmp <- NULL


# add in extra column as factor
data_melted$horiz_factor <- rep("motifs", nrow(data_melted))
logit_full_melted$horiz_factor <- rep("logits", nrow(logit_full_melted))

# adjust response column
#data_melted$response <- as.character(data_melted$response)


# and rbind all together
data_melted <- rbind(data_melted, logit_full_melted)
print(tail(data_melted, n=10L))


# fix order of respondents

#data_melted$response[data_melted$horiz_factor == "logits"] <- "0.0"

data_melted$response <- factor(
    data_melted$response,
    levels=c(rev(rownames(logits_means)), colnames(data_mut_means)),
    ordered=TRUE)
    #labels=c("0", "-1", colnames(data_mut_means)))

print(head(data_melted$response))

data_melted$target <- factor(
    data_melted$target,
    levels=c("motifs", "TFAP2B", "SMAD3"))

print(tail(data_melted))

#data_melted$target <- as.numeric(data_melted$target)

# plot
#p <- ggplot(data_melted, aes(x=response, y=target, fill=delta)) + geom_tile(color="white") +
#    scale_fill_gradient(high="white", low="steelblue") +
#        theme_bw() +
#            theme(
#                panel.grid.major=element_blank(),
#                axis.text.x=element_text(size=5, angle=60, hjust=1))
        
#ggsave("testing2.pdf", height=4, width=16)

#data_melted$x <- nrow(data_melted):1

print(tail(data_melted))

# TODO - at the end, will replace the geom_points with geom_tile?
# geom_tile with scale fill gradient (check out r calendar heatmap)
# the aes is x=response_motif, y=timepoint in the main ggplot call
# the fill is the delta
# the facet grid is the same, target ~ horiz factor

p <- ggplot(data_melted, aes(x=x, y=delta)) +
    facet_grid(target ~ horiz_factor, scales="free", space="free_x", switch="y") +  # space=free_x

        geom_col(
            data=subset(data_melted, target=="motifs"),
            aes(y=delta, fill=-delta)) +

            # factor this correctly, so that not duplicating code
            geom_errorbar(
                data=subset(data_melted, horiz_factor=="motifs"),
                aes(ymin=delta-2*sd, ymax=delta+2*sd, color=delta), width=0.05) +
            geom_point(
                data=subset(data_melted, horiz_factor=="motifs" & target=="TFAP2B"),
                aes(y=delta, color=delta, size=-delta)) +
            geom_point(
                data=subset(data_melted, horiz_factor=="motifs" & target=="SMAD3"),
                aes(y=delta, color=delta, size=-delta)) +

            # this gets the spread out on the left
            geom_point(
                data=subset(data_melted, horiz_factor=="logits"),
                alpha=0.00,
                aes(x=-5, y=0, color=0)) + #x=10*delta, y=0

            geom_point(
                data=subset(data_melted, horiz_factor=="logits"),
                alpha=0.00,
                aes(x=5, y=0, color=0)) + #x=10*delta, y=0

            # this gets the labels on
            #geom_point(
             #   data=subset(data_melted, horiz_factor=="logits"),
             #   alpha=0.00,
             #   aes(y=0, color=0)) + #x=10*delta, y=0
                
            geom_point(
                data=subset(data_melted, response=="a" & horiz_factor=="logits"),
                aes(x=5*delta, y=0)) + #x=10*delta, y=0

            geom_segment(
                data=subset(data_melted, response=="a" & horiz_factor=="logits"),
                aes(x=0, y=0, xend=5*delta, yend=0)) + #x=10*delta, y=0

            #geom_segment(
            #    data=subset(data_melted, response=="a" & horiz_factor=="logits"),
            #    aes(x=0, y=0, xend=0, yend=0)) + #x=10*delta, y=0

                #expand_limits(x=-1) +
    #scale_x_discrete(breaks=1:200) +

                scale_y_continuous(position="right") +
    scale_fill_gradient(high="white", low="steelblue") +
            #geom_bar(data=subset(data_melted, target=="motif"), stat="identity", aes(color=target)) +
        theme_bw() +
                              #geom_vline(xintercept=0, color="gray") +
                scale_x_continuous(
                    breaks=c(seq(-5, 5, length.out=9), 1:ncol(data_mut_means)+10),
                    labels=c(seq(-1, 1, length.out=9), colnames(data_mut_means)),
                    expand=c(0,0)) +

            
            theme(
                panel.grid.major.x=element_blank(),
                panel.grid.minor=element_blank(),
                axis.text.x=element_text(size=5, angle=60, hjust=1),
                axis.text.y=element_text(size=5),
                legend.text=element_text(size=5),
                legend.key.size=unit(0.5, "line"),
                legend.spacing=unit(0.5, "line"),
                strip.background=element_blank())
        
ggsave("testing2.pdf", height=4, width=16)


# to make the final one, need an extra column for the y position?


p2 <- ggplot(data_melted, aes(x=response, y=0)) +
    facet_grid(target ~ horiz_factor, scales="free", space="free_x", switch="y") +  # space=free_x

        geom_tile(
            data=subset(data_melted, target=="motifs"),
            aes(fill=delta)) +

            # factor this correctly, so that not duplicating code
            #geom_errorbar(
            #    data=subset(data_melted, horiz_factor=="motifs"),
            #    aes(ymin=delta-2*sd, ymax=delta+2*sd, color=delta), width=0.05) +
            geom_tile(
                data=subset(data_melted, horiz_factor=="motifs" & target=="TFAP2B"),
                aes(fill=delta)) +
            geom_tile(
                data=subset(data_melted, horiz_factor=="motifs" & target=="SMAD3"),
                aes(fill=delta)) +

            # this gets the spread out on the left
            #geom_tile(
            #    data=subset(data_melted, horiz_factor=="logits"),
            #    alpha=0.00,
            #    aes(x=-5, y=0, color=0)) + #x=10*delta, y=0

            #geom_point(
            #    data=subset(data_melted, horiz_factor=="logits"),
           #     alpha=0.00,
             #   aes(x=5, y=0, color=0)) + #x=10*delta, y=0

            # this gets the labels on
            #geom_point(
             #   data=subset(data_melted, horiz_factor=="logits"),
             #   alpha=0.00,
             #   aes(y=0, color=0)) + #x=10*delta, y=0
                
            geom_tile(
                data=subset(data_melted, response=="a" & horiz_factor=="logits"),
                aes(fill=delta)) + #x=10*delta, y=0

            #geom_segment(
            #    data=subset(data_melted, response=="a" & horiz_factor=="logits"),
            #    aes(x=0, y=0, xend=5*delta, yend=0)) + #x=10*delta, y=0

            #geom_segment(
            #    data=subset(data_melted, response=="a" & horiz_factor=="logits"),
            #    aes(x=0, y=0, xend=0, yend=0)) + #x=10*delta, y=0

                #expand_limits(x=-1) +
    #scale_x_discrete(breaks=1:200) +

                scale_y_continuous(position="right") +
    scale_fill_gradient(high="white", low="steelblue") +
            #geom_bar(data=subset(data_melted, target=="motif"), stat="identity", aes(color=target)) +
        theme_bw() +
                              #geom_vline(xintercept=0, color="gray") +
                #scale_x_continuous(
                #    breaks=c(seq(-5, 5, length.out=9), 1:ncol(data_mut_means)+10),
                #    labels=c(seq(-1, 1, length.out=9), colnames(data_mut_means)),
                #    expand=c(0,0)) +

            
            theme(
                panel.grid.major.x=element_blank(),
                panel.grid.minor=element_blank(),
                axis.text.x=element_text(size=5, angle=60, hjust=1),
                axis.text.y=element_text(size=5),
                legend.text=element_text(size=5),
                legend.key.size=unit(0.5, "line"),
                legend.spacing=unit(0.5, "line"),
                strip.background=element_blank())
        
ggsave("testing3.pdf", height=4, width=16)
