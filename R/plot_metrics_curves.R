#!/usr/bin/env Rscript

# Description:
# given a set of files containing (x,y) columns, plot on same graph
# use for AUROC/AUPRC plots

library(ggplot2)
library(RColorBrewer)

# reminders
print("Remember that the first piece of the filename (before a '.') is the task name")

args <- commandArgs(trailingOnly=TRUE)
plot_title <- args[1]
x_lab <- args[2]
y_lab <- args[3]
out_file <- args[4]
data_files <- args[5:length(args)]

# initialize main dataframe
all_data <- data.frame()

for (i in 1:length(data_files)) {

    # for each data file, get out x and y and also set up with distinguishing factor name
    data <- read.table(data_files[i], header=TRUE, sep='\t')
    data$task <- sapply(strsplit(rep(basename(data_files[i]), nrow(data)), "\\."), "[", 1)
    all_data <- rbind(all_data, data)

}

# plot
my_palette <- rev(colorRampPalette(brewer.pal(11, 'Spectral'))(length(unique(all_data$task))))
ggplot(data=all_data, aes(x=x, y=y, group=task)) +
    geom_line(aes(color=task)) + labs(x=x_lab, y=y_lab, title=plot_title) +
    scale_x_continuous(limits=c(0,1), expand=c(0,0)) +
    scale_y_continuous(limits=c(0,1), expand=c(0,0)) + 

    theme_bw() + coord_fixed() +
    theme(
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank()
        ) + 
    scale_colour_manual(values=my_palette)
ggsave(out_file)






