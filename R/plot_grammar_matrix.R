
# description: code to plot matrix of grammars

library(ggplot2)
library(reshape2)

library(cluster)

args <- commandArgs(trailingOnly=TRUE)
files <- args

grammar_mat <- data.frame()

# load in components
for (i in 1:length(files)) {

    component_file <- files[i]

    # get task num
    task <- unlist(strsplit(basename(component_file), ".", fixed=TRUE))[1]
    task_num <- unlist(strsplit(task, "-"))[2]

    # get component num
    component <- unlist(strsplit(basename(component_file), ".", fixed=TRUE))[3]
    component_num<- unlist(strsplit(component, "_"))[2]
    
    # load in components and add to matrix
    component <- read.table(component_file, header=FALSE, row.names=NULL, stringsAsFactors=FALSE)
    colnames(component) <- c("pwm_name", "hgnc")
    component$task <- as.integer(task_num)
    component$intersection <- paste(task_num, component_num, sep=".")

    # and now add matrices
    grammar_mat <- rbind(grammar_mat, component)

}

#print(head(grammar_mat))

#quit()

# reorder the columns
task_nums <- c(15, 16, 17, 18, 19, 20, 21, 22, 23, 13, 14)
for (i in 1:length(task_nums)) {
    grammar_mat$task[grammar_mat$task == task_nums[i]] <- i
}


# TODO - what is optimal component ordering?

# reorder the rows, first pull global index and then sort
row_ordering_file <- "global.ordering.tmp"
row_ordering <- read.table(row_ordering_file, header=FALSE, row.names=NULL, stringsAsFactors=FALSE)
colnames(row_ordering) <- c("global_idx", "name")
row_ordering$index <- seq(1:nrow(row_ordering))
#print(head(row_ordering))

grammar_mat$pwm_global_idx <- 0
for (i in 1:nrow(row_ordering)) {
    grammar_mat$pwm_global_idx[grammar_mat$hgnc == row_ordering$name[i]] <- row_ordering$index[i]
}

pwm_global_indices <- sort(unique(grammar_mat$pwm_global_idx))
print(pwm_global_indices)

for (i in 1:length(pwm_global_indices)) {
    grammar_mat$pwm_idx[grammar_mat$pwm_global_idx == pwm_global_indices[i]] <- i
}


# set up motif by motif mat
num_pwms <- length(pwm_global_indices)
dist_mat <- data.frame(matrix(0, ncol=num_pwms, nrow=num_pwms))
intersect_sets <- unique(grammar_mat$intersection)

for (i in 1:length(intersect_sets)) {

    set_name <- intersect_sets[i]
    subset <- grammar_mat[grammar_mat$intersection == set_name,]
    for (idx_i in 1:nrow(subset)) {
        for (idx_j in 1:nrow(subset)) {
            if (idx_i == idx_j) {
                next
            } else {
                pwm_idx_i <- subset$pwm_idx[idx_i]
                pwm_idx_j <- subset$pwm_idx[idx_j]
                
                dist_mat[pwm_idx_i, pwm_idx_j] = dist_mat[pwm_idx_i, pwm_idx_j] + 1
            }
        }
    }

}

dist_mat <- dist_mat / max(dist_mat)
dist_mat <- 1 - dist_mat
dist_mat[dist_mat == 1] <- 1


#hc <- hclust(as.dist(dist_mat))
hc <- diana(as.dist(dist_mat))


better_ordering <- hc$order


grammar_mat$pwm_new_idx <- 0
for (i in 1:length(better_ordering)) {
    grammar_mat$pwm_new_idx[grammar_mat$pwm_idx == better_ordering[i]] <- i
}
print(better_ordering)


grammar_mat_unmelted <- dcast(grammar_mat, hgnc + pwm_new_idx ~ task)
grammar_mat_unmelted_sorted <- grammar_mat_unmelted[order(grammar_mat_unmelted$pwm_new_idx),]
pwm_names_sorted <- grammar_mat_unmelted_sorted$hgnc

print(length(pwm_names_sorted))
print(length(pwm_global_indices))
print(length(num_pwms))

#grammar_mat_unmelted[is.na(grammar_mat_unmelted)] <- 0
#grammar_mat_unmelted[grammar_mat_unmelted != 0] <- 1
#grammar_mat_unmelted$pwm_idx <- NULL



if (FALSE) {
    # order by intersection overlap
    grammar_mat_unmelted <- dcast(grammar_mat, pwm_idx ~ task)
    grammar_mat_unmelted[is.na(grammar_mat_unmelted)] <- 0
    grammar_mat_unmelted[grammar_mat_unmelted != 0] <- 1
    grammar_mat_unmelted$pwm_idx <- NULL

    hc <- hclust(dist(grammar_mat_unmelted))
    better_ordering <- hc$order

    grammar_mat$pwm_new_idx <- 0
    for (i in 1:length(better_ordering)) {
        grammar_mat$pwm_new_idx[grammar_mat$pwm_idx == better_ordering[i]] <- i
    }
}


# and plot
#p <- ggplot() + geom_point(data=grammar_mat, aes_string(x="task", y="pwm_new_idx")) +
#    geom_line(
#        data=grammar_mat,
#        aes_string(group="intersection", x="task", y="pwm_new_idx"),
#        position=position_dodge(width=0.2))
        
#ggsave("testing.mat.pdf")


p <- ggplot() +
    geom_point(
        data=grammar_mat,
        aes_string(group="intersection", x="task", y="pwm_new_idx"),
        size=2,
        position=position_dodge(width=0.2)) +
    scale_y_continuous(breaks=1:num_pwms, labels=pwm_names_sorted) +
    scale_x_continuous(breaks=1:length(task_nums), labels=1:length(task_nums)) +
    geom_line(
        data=grammar_mat,
        aes_string(group="intersection", x="task", y="pwm_new_idx"),
        position=position_dodge(width=0.2)) +
    theme(panel.grid.minor=element_blank())

        
ggsave("components_across_tasks.mat.pdf")





