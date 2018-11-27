#!/usr/bin/env Rscript

# description: merge GREAT results

library(gplots)
library(RColorBrewer)
library(reshape2)

args <- commandArgs(trailingOnly=TRUE)
out_dir <- args[1]
files <- args[2:length(args)]

# output plot file
plot_file <- paste(out_dir, "/", "great.results.summary.pdf", sep="")

# go through files
for (i in 1:length(files)) {

    file <- files[i]
    print(file)

    # get a file id for file
    file_id <- unlist(strsplit(basename(file), ".GO_Biol", fixed=TRUE))[1]
    
    # read in file
    file_data <- read.table(file, header=TRUE, sep="\t")
    file_data <- data.frame(
        GO_ID=paste(file_data$ID, file_data$name),
        val=-log10(file_data$hyper_q_vals))
    colnames(file_data)[2] <- file_id

    # don't keep if no GO terms
    if (nrow(file_data) < 2) { next }

    # optionally cutoff for top 10 terms?
    cutoff <- min(15, nrow(file_data))
    file_data <- file_data[1:cutoff,]
    
    # merge in 
    if (!exists("all_data")) {
        all_data <- file_data
    } else {
        all_data <- merge(all_data, file_data, by="GO_ID", all=TRUE)
    }

}

# move GO terms to rownames
rownames(all_data) <- all_data$GO_ID
all_data$GO_ID <- NULL
all_data[is.na(all_data)] <- 0

# plot out
hc <- hclust(dist(all_data), method="ward.D2")
all_data <- all_data[hc$order,]
print(dim(all_data))

mylmat = rbind(c(0,3,0),c(2,1,0),c(0,4,0))
mylwid = c(1,6,9)
mylhei = c(0.5,6,0.75)

my_palette <- colorRampPalette(brewer.pal(9, "Reds"))(49)

color_granularity <- 50
data_melted <- melt(all_data)
my_breaks <- seq(
    quantile(data_melted$value, 0.01),
    quantile(data_melted$value, 0.99),
    length.out=color_granularity)
print(my_breaks)


pdf(plot_file, height=42, width=15)
heatmap.2(
    as.matrix(all_data),
    Rowv=FALSE,
    Colv=TRUE,
    dendrogram="none",
    trace="none",
    density.info="none",
    
    #labRow="",
    #labCol=labCol,
    #srtCol=270,
    cexCol=1,
    cexRow=1,
    
    keysize=0.1,
    key.title=NA,
    key.xlab=NA,
    key.par=list(pin=c(4,0.1),
        mar=c(9.1,1,15.1,1),
        mgp=c(3,2,0),
        cex.axis=2.0,
        font.axis=2),
    key.xtickfun=function() {
        breaks <- pretty(parent.frame()$breaks)
                                        #breaks <- breaks[c(1,length(breaks))]
        list(at = parent.frame()$scale01(breaks),
             labels = breaks)},

    margins=c(3,0),
    lmat=mylmat,
    lwid=mylwid,
    lhei=mylhei,

    col=my_palette,
    #breaks=my_breaks,
    useRaster=FALSE)
dev.off()









