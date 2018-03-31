#!/usr/bin/env Rscript

# description: plot region x pwm from hdf5 file
# NOTE: rhdf5 transposes axes!

library(rhdf5)
library(gplots)
library(RColorBrewer)

# args
args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
dataset_key <- args[2]

mut_names <- c("SMAD3", "TFAP2B")

#h5ls(h5_file)

se <- function(x) sd(x) / sqrt(length(x))

# read in data
data <- h5read(h5_file, dataset_key, read.attributes=TRUE)#[,2,]
pwm_data <- h5read(h5_file, "pwm-scores.taskidx-0", read.attributes=TRUE)
logits_data <- h5read(h5_file, "delta_logits", read.attributes=TRUE)[,1,]

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

p <- ggplot(data_melted, aes(x=x, y=delta)) +
    facet_grid(target ~ horiz_factor, scales="free", space="free_x", switch="y") +  # space=free_x

        geom_col(
            data=subset(data_melted, target=="motifs"),
            aes(y=delta, fill=-delta)) +


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

