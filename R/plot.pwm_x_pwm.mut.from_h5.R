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

#h5ls(h5_file)

se <- function(x) sd(x) / sqrt(length(x))

# read in data
data <- h5read(h5_file, dataset_key, read.attributes=TRUE)#[,2,]
pwm_data <- h5read(h5_file, "pwm-scores.taskidx-0", read.attributes=TRUE)

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


# transpose (fastest changing axis is opposite order in R vs python)
data <- aperm(data, c(3, 2, 1))

# mean across example axis
data_mut_means <- apply(data, c(2, 3), mean)
colnames(data_mut_means) <- attr(pwm_data, "pwm_names")
rownames(data_mut_means) <- c("SMAD3", "TFAP2B")


# standard error
data_mut_sd <- apply(data, c(2, 3), se)
colnames(data_mut_sd) <- attr(pwm_data, "pwm_names")
rownames(data_mut_sd) <- c("SMAD3", "TFAP2B")

# order by strongest effect
data_mut_means <- data_mut_means[,order(colSums(data_mut_means))]


# color palette
#my_palette <- colorRampPalette(brewer.pal(9, "Blues"))(49)
my_palette <- rev(colorRampPalette(brewer.pal(9, "RdBu"))(49))

# grid
mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(2,10,2)
mylhei = c(0.5,2,0.5)

# heatmap
heatmap_file <- paste(
    paste(
        sub(".h5", "", h5_file),
        dataset_key,
        "1", sep="-"), # to match 0-START from python
    "pdf", sep=".")
heatmap_file <- "testing.pdf"
pdf(heatmap_file, height=18, width=20)
heatmap.2(
    as.matrix(data_mut_means),
    Rowv=FALSE,
    Colv=FALSE,
    #dendrogram="column",
    dendrogram="none",
    trace="none",
    density.info="none",
    #labRow="",
    cexCol=0.5,
    keysize=0.1,
    key.title=NA,
    key.xlab=NA,
    key.par=list(pin=c(4,0.1),
        mar=c(9.1,0,2.1,0),
        mgp=c(3,2,0),
        cex.axis=2.0,
        font.axis=2),
    margins=c(3,0),
    lmat=mylmat,
    lwid=mylwid,
    lhei=mylhei,
    col=my_palette)
dev.off()

# ggplot2 version
library(ggplot2)
library(reshape)

data_melted <- melt(data_mut_means)
colnames(data_melted) <- c("target", "response", "delta")
data_melted$id <- paste(data_melted$target, data_melted$response, sep="_to_")

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
#pwm_means_melted$target <- NULL
#pwm_means_melted$response <- NULL

pwm_full_melted <- merge(pwm_means_melted, pwm_se_melted, by="id")

# and rbind all together
data_melted <- rbind(data_melted, pwm_full_melted)
#data_melted <- merge(data_melted, pwm_full_melted, by="id")

print(head(pwm_full_melted))
print(dim(pwm_full_melted))

# fix order of respondents
data_melted$response <- factor(
    data_melted$response,
    levels=colnames(data_mut_means))

data_melted$target <- factor(
    data_melted$targe,
    levels=c("motifs", "TFAP2B", "SMAD3"))

data_melted$delta_2 <- data_melted$delta
print(head(data_melted))


#data_melted$target <- as.numeric(data_melted$target)

# plot
#p <- ggplot(data_melted, aes(x=response, y=target, fill=delta)) + geom_tile(color="white") +
#    scale_fill_gradient(high="white", low="steelblue") +
#        theme_bw() +
#            theme(
#                panel.grid.major=element_blank(),
#                axis.text.x=element_text(size=5, angle=60, hjust=1))
        
#ggsave("testing2.pdf", height=4, width=16)


p <- ggplot(data_melted, aes(x=response, y=delta)) +
    facet_grid(target ~ ., scales="free") +
        geom_col(data=subset(data_melted, target=="motifs"), aes(y=delta, fill=-delta)) +


            geom_errorbar(aes(ymin=delta-2*sd, ymax=delta+2*sd, color=delta), width=0.05) +
        geom_point(data=subset(data_melted, target=="TFAP2B"), aes(y=delta, color=delta, size=-delta)) +
            geom_point(data=subset(data_melted, target=="SMAD3"), aes(y=delta, color=delta, size=-delta)) +



    scale_fill_gradient(high="white", low="steelblue") +
            #geom_bar(data=subset(data_melted, target=="motif"), stat="identity", aes(color=target)) +
        theme_bw() +
            theme(
                panel.grid.major=element_blank(),
                axis.text.x=element_text(size=5, angle=60, hjust=1))
        
ggsave("testing2.pdf", height=4, width=16)



quit()





# asdfasdfasdf





pwm_data <- t(pwm_data)
print(dim(data))
print(dim(pwm_data))


quit()

# normalize
rowmax <- apply(pwm_data, 1, function(x) max(x))
pwm_norm <- pwm_data / rowmax
pwm_norm[is.na(pwm_norm)] <- 0
pwm_data <- pwm_norm

data_norm <- data

# get mean and standard dev
data_stds <- apply(data_norm, 2, sd)
data_means <- apply(data_norm, 2, mean)

print(data_means)
print(sort(data_means))

quit()

# draw ordered sample
plottable_nrow <- 1500
if (nrow(data_norm) > plottable_nrow) {
    skip_int <- as.integer(nrow(data_norm) / plottable_nrow)
    ordered_sample_indices <- seq(1, nrow(data_norm), skip_int)
    data_ordered <- data_norm[ordered_sample_indices,]
    pwm_data <- pwm_data[ordered_sample_indices,]
} else {
    data_ordered <- data_norm
}

# order out here
hc <- hclust(dist(data_ordered))
data_ordered <- data_ordered[hc$order,]
pwm_data <- pwm_data[hc$order,]

# color palette
my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
my_palette <- rev(colorRampPalette(brewer.pal(9, "RdBu"))(49))

# grid
mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(2,6,2)
mylhei = c(0.5,12,1.5)

# heatmap
heatmap_file <- paste(
    paste(
        sub(".h5", "", h5_file),
        dataset_key,
        "1", sep="-"), # to match 0-START from python
    "pdf", sep=".")
pdf(heatmap_file, height=18, width=20)
heatmap.2(
    as.matrix(data_ordered),
    Rowv=FALSE,
    Colv=TRUE,
    dendrogram="column",
    trace="none",
    density.info="none",
    labRow="",
    cexCol=0.5,
    keysize=0.1,
    key.title=NA,
    key.xlab=NA,
    key.par=list(pin=c(4,0.1),
        mar=c(9.1,0,2.1,0),
        mgp=c(3,2,0),
        cex.axis=2.0,
        font.axis=2),
    margins=c(3,0),
    lmat=mylmat,
    lwid=mylwid,
    lhei=mylhei,
    col=my_palette)
dev.off()


# heatmap
my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)
heatmap_file <- paste(
    paste(
        sub(".h5", "", h5_file),
        dataset_key,
        "0", sep="-"), # to match 0-START from python
    "example_x_pwm",
    "pdf", sep=".")
pdf(heatmap_file, height=18, width=20)
heatmap.2(
    as.matrix(pwm_data),
    Rowv=FALSE,
    Colv=TRUE,
    dendrogram="column",
    trace="none",
    density.info="none",
    labRow="",
    cexCol=0.5,
    keysize=0.1,
    key.title=NA,
    key.xlab=NA,
    key.par=list(pin=c(4,0.1),
        mar=c(9.1,0,2.1,0),
        mgp=c(3,2,0),
        cex.axis=2.0,
        font.axis=2),
    margins=c(3,0),
    lmat=mylmat,
    lwid=mylwid,
    lhei=mylhei,
    col=my_palette)
dev.off()
