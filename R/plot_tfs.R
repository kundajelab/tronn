#!/usr/bin/env Rscript

# quick plot

library(gplots)
library(RColorBrewer)


args <- commandArgs(trailingOnly=TRUE)

data_file <- args[1]
out_file <- args[2]

data <- read.table(data_file, header=TRUE, row.names=1)

# normalize the columns
data_norm <- t(t(data)/colSums(data))
data_norm <- t(scale(t(data_norm), center=TRUE, scale=TRUE))
data_norm <- na.omit(data_norm)

write.table(data_norm, file="spotcheck.txt", sep='\t', quote=FALSE)

# then plot out for initial look
my_palette <- rev(colorRampPalette(brewer.pal(11, "RdBu"))(49))

data_hc <- hclust(dist(data_norm))
print(data_hc$order)
ordering <- data_hc$order

dend <- as.dendrogram(data_hc)
dend[[2]] <- rev(dend[[2]])
ordering <- order.dendrogram(dend)

print(ordering)

#mylmat = rbind(c(0,0,3,0),c(4,1,2,0),c(0,0,5,0))
#mylwid = c(2,0.5,6,2)
#mylhei = c(0.5,12,1.5)

mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(2,6,2)
mylhei = c(0.5,12,1.5)

#png("pwm_x_timepoint.ordered.png", height=18, width=6, units="in", res=200)
pdf(out_file, height=18, width=6)
heatmap.2(
	as.matrix(data_norm[ordering,]),
	Rowv=FALSE,
	Colv=FALSE,
    dendrogram="row",
    	#dendrogram="none",	
	trace="none",
        density.info="none",
    cexRow=2,
    cexCol=3,
        keysize=0.1,
        key.title=NA,
        key.xlab=NA,
        key.par=list(pin=c(4,0.1),
            mar=c(6.1,0,5.1,0),
            mgp=c(3,2,0),
            cex.axis=2.0,
            font.axis=2),
	margins=c(3,0),
	lmat=mylmat,
        lwid=mylwid,
        lhei=mylhei,
	col=my_palette)
dev.off()


q()

# global most seen
data <- read.table("global.most-seen.txt", header=FALSE, row.names=1)
print(head(data))

data <- cbind(data, data)

print(head(data))

my_palette <- rev(colorRampPalette(brewer.pal(11, "Purples"))(49))

png("most-seen.ordered.png", height=18, width=6, units="in", res=200)
heatmap.2(
	as.matrix(data[data_hc$order,]),
	Rowv=FALSE,
	Colv=FALSE,
	dendrogram="none",	
	trace="none",
	key=FALSE, # change later!!
	col=my_palette)
dev.off()

q()

# trajectory matrix

my_palette <- rev(colorRampPalette(brewer.pal(11, "RdBu"))(49))
data <- read.table("trajectories.pwm_x_timepoint.mat.txt", header=TRUE, row.names=1)

# normalize the columns
data_norm <- t(t(data)/colSums(data))

data_norm <- t(scale(t(data_norm), center=TRUE, scale=TRUE))

png("pwm_x_timepoint.trajectories.ordered.png", height=18, width=6, units="in", res=200)
heatmap.2(
	as.matrix(data_norm[data_hc$order,]),
	Rowv=FALSE,
	Colv=FALSE,
	dendrogram="none",	
	trace="none",
	key=FALSE, # change later!!
	col=my_palette)
dev.off()




