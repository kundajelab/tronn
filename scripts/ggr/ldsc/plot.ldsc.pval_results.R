#!/usr/bin/env Rscript

library(ggplot2)
library(qvalue)
library(ggsci)

# args
args <- commandArgs(trailingOnly=TRUE)
results_file <- args[1]
plot_file <- args[2]
plot_file2 <- args[3]

fdr_thresh <- 0.10

# read data, get qvalue
data <- read.table(results_file, sep="\t", header=TRUE)

# get phenotype string
phenotype <- strsplit(basename(results_file), ".", fixed=TRUE)
phenotype <- phenotype[[1]][3]
print(phenotype)

# adjust colors
data$category <- "0 Union DHS"
data$category[grepl("HepG2", data$Name)] <- "1 HepG2"
data$category[grepl("GGR_ALL", data$Name)] <- "2 Union ATAC (KC)"
data$category[grepl("ATAC-seq", data$Name)] <- "3 ATAC timepoints"
data$category[grepl("cluster", data$Name)] <- "4 Dynamic trajectories"
data$category[grepl("TRAJ_LABELS-0", data$Name)] <- "5 Early rule"
data$category[grepl("TRAJ_LABELS-8", data$Name)] <- "5 Early rule"
data$category[grepl("TRAJ_LABELS-9", data$Name)] <- "5 Early rule"
data$category[grepl("TRAJ_LABELS-1", data$Name, fixed=TRUE)] <- "6 Mid rule"
data$category[grepl("TRAJ_LABELS-2", data$Name)] <- "7 Late rule"
data$category[grepl("TRAJ_LABELS-3", data$Name)] <- "7 Late rule"
data$category[grepl("grouped", data$Name)] <- "8 Grouped rules"

# get qval and convert cutoff to pval
if (grepl("dermatitis", phenotype)) {
    lambda <- seq(0, 0.95, 0.05)
} else {
    lambda <- seq(0, 0.8, 0.2)
}
q_obj <- qvalue(data$Coefficient_P_value, lambda=lambda)
cutoff_idx <- min(which(q_obj$qvalues >= fdr_thresh))
if (cutoff_idx > 1) {
    pval_thresh <- -log10((q_obj$pvalues[cutoff_idx] + q_obj$pvalues[cutoff_idx-1])/2)
} else {
    pval_thresh <- -log10(q_obj$pvalues[cutoff_idx])
}

# conversions to -log10
data$log10pvals <- -log10(data$Coefficient_P_value)
data$size <- 0
data$size[data$log10pvals > pval_thresh] <- 1

# ordering
data <- data[order(data$category, data$Name),]
data$Name <- factor(data$Name, levels=data$Name, ordered=TRUE)
data$category <- gsub("^[[:digit:]] ", "", data$category)

# factorize for plotting
data$category <- factor(data$category, levels=unique(data$category), ordered=TRUE)
data$size <- factor(data$size, levels=c(0, 1), ordered=TRUE)

# plot
ggplot(data, aes(x=Name, y=log10pvals, colour=category)) +
    geom_hline(size=0.115, yintercept=pval_thresh, linetype="dashed") +
    geom_point(aes(size=size)) +
    labs(x="Annotation", y=expression(-log[10](italic(P))), title=phenotype) +
    theme_bw() +
    theme(
        text=element_text(size=6, family="ArialMT"),
        aspect.ratio=1,
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background=element_blank(),
        panel.border=element_blank(),
        axis.title=element_text(margin=margin(0,0,0,0)),
        axis.line=element_line(size=0.115, color="black", lineend="square"),
        axis.ticks.y=element_line(size=0.25),
        axis.ticks.x=element_blank(),
        axis.text.x=element_blank(),
        axis.text.y=element_text(size=6),
        legend.text=element_text(size=5),
        legend.key.size=unit(0.01, "in")) +
    scale_size_manual(values=c(0.03, 1), guide="none") +
    scale_y_continuous(limits=c(0,3), expand=c(0,0)) +
    scale_color_d3("category20")
        
ggsave(plot_file, height=2, width=3, useDingbats=FALSE)

# also plot out with conf intervals etc        
#data <- data[order(data$category, -data$Coefficient),]
#data$Name <- factor(data$Name, levels=data$Name, ordered=TRUE)
data$ymax <- data$Coefficient + data$Coefficient_std_error
data$ymin <- data$Coefficient - data$Coefficient_std_error
data$sig <- 0
data$sig[data$ymin > 0] <- 1
data$sig <- factor(data$sig, levels=c(0, 1), ordered=TRUE)

ggplot(data, aes(x=Name, y=Coefficient, colour=category)) +
    geom_hline(size=0.115, yintercept=0, linetype="dashed") +
    geom_point(aes(size=sig)) +
    geom_errorbar(aes(ymax=ymax, ymin=ymin), width=0.01, size=0.115) +
    labs(x="Annotation", y="LDSC Enrichment (h2/SNP)", title=phenotype) +
    theme_bw() +
    theme(
        text=element_text(size=6, family="ArialMT"),
        aspect.ratio=1,
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background=element_blank(),
        panel.border=element_blank(),
        axis.title=element_text(margin=margin(0,0,0,0)),
        axis.line=element_line(size=0.115, color="black", lineend="square"),
        axis.ticks.y=element_line(size=0.25),
        axis.ticks.x=element_blank(),
        axis.text.x=element_blank(),
        axis.text.y=element_text(size=6),
        legend.text=element_text(size=5),
        legend.key.size=unit(0.01, "in")) +
    scale_size_manual(values=c(0.03, 0.6), guide="none") +
    #scale_y_continuous(limits=c(0,3), expand=c(0,0)) +
    scale_color_d3("category20")
        
ggsave(plot_file2, height=2, width=3, useDingbats=FALSE)
