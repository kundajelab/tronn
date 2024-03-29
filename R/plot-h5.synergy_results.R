#!/usr/bin/env Rscript

# description - code to plot out synergy scores

library(rhdf5)
library(reshape2)
library(ggplot2)

args <- commandArgs(trailingOnly=TRUE)
h5_file <- args[1]
key <- args[2]
task_idx <- as.numeric(args[3])
out_file <- args[4]

# read in data
data <- h5read(h5_file, key, read.attributes=TRUE)
labels <- attr(data, "labels")
data <- data.frame(aperm(data)[,,task_idx])
colnames(data) <- labels



#data <- 2^data
data$diff <- data[,1] - data[,2]

probs <- c(0.01, 0.99)
thresh1 <- quantile(data[,1], probs=probs)
thresh2 <- quantile(data[,2], probs=probs)
threshdiff <- quantile(data$diff, probs=probs)
#threshdiff <- quantile(data$diff, probs=c(0.75, 0.99))

if (TRUE) {
    data <- data[data[,1] > thresh1[1],]
    data <- data[data[,1] < thresh1[2],]
    
    data <- data[data[,2] > thresh2[1],]
    data <- data[data[,2] < thresh2[2],]
    
    data <- data[data$diff > threshdiff[1],]
    data <- data[data$diff < threshdiff[2],]
}
#data <- data[data$diff > 0.5,]
#data <- data[data$diff < 1,]

print(head(data))

if (FALSE) {
    data <- data[sample(1:nrow(data), 30),]
}

# set up for ggplot
data_melted <- melt(data)
data_melted$Var1 <- NULL
colnames(data_melted)[1] <- "variable"
print(head(data_melted))

ggplot(data_melted, aes(x=variable, y=value, fill=variable)) +
    #geom_boxplot(alpha=0.6, position=position_dodge(width=0.9)) + # colour="black"
    geom_violin(alpha=0.6, position=position_dodge(width=0.9)) +
    geom_point(
        alpha=0.3,
        position=position_jitterdodge(dodge.width=0.9, jitter.width=0.50)) + # size=3
    labs(x=NULL, y="log2(FC)\n") +
    theme_bw() +
    theme(
        axis.title=element_text(size=32),
        axis.text.y=element_text(size=16),
        axis.text.x=element_text(size=12, angle=30, hjust=1),
        legend.title=element_text(size=32),
        legend.text=element_text(size=12)) +
                                        #scale_fill_hue(l=60, c=35) +
                                        #scale_color_hue(l=60, c=35)
     scale_fill_brewer(palette="Dark2") +
     scale_color_brewer(palette="Dark2")

ggsave(out_file, height=7, width=7)
