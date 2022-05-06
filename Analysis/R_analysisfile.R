####General data selection####
library(tidyverse)

RDS_path = file.path("C:", "Users", "maudb", "Documents", "Psychologie", "2e_master_psychologie", 
                     "Master_thesis", "OpenFace_processing")

all_files <- list.files(path = folder, pattern = ".rds")
i <- 0
for (file in all_files){
  if (i == 0){ all_data <- readRDS(file.path(RDS_path, file))}
  else{all_data <- rbind(all_data, readRDS(file.path(RDS_path, file)))}
  i <- i+1
}

trial_data <- all_data[all_data$Frame_count == 0,]
trial_data <- trial_data[, c('block_count', 'Affect', 'accuracy', 'pp_number')]
library(vctrs)
vec_unique(trial_data$pp_number)


trial_data$accuracy <- c(trial_data$accuracy > 0)*1
vec_unique(trial_data$accuracy)

trial_data$pp_number[trial_data$pp_number == 317] <- 17
vec_unique(trial_data$pp_number)

pp_numbers <- c(1:33)

blocks <- c(1:3)
mean_accuracies <- data.frame(matrix(NA, nrow = length(pp_numbers), ncol = length(blocks)))
colnames(mean_accuracies) <- c('block 1', 'block 2', "block 3")
for (pp in pp_numbers){
  pp_data <- trial_data[trial_data$pp_number == pp,]
  for (block in blocks){
    block_data <- pp_data[pp_data$block_count == block-1,]
    mean_accuracies[pp, block] <- mean(block_data$accuracy)}
  mean_accuracies$pp_number[pp] <- pp}
vec_unique(mean_accuracies$pp_number)

#########################################################################
#####Wide to long format####
mean_accuracies.wide <- mean_accuracies
library(reshape2)
mean_accuracies.long <- melt(mean_accuracies, id.vars = "pp_number", variable.name = "block_count",
                             value.name = "accuracy")

####plotting####
library(lattice)
print(bwplot(accuracy ~ block_count, data = mean_accuracies.long))
print(xyplot(accuracy ~ block_count, groups = pp_number, data = mean_accuracies.long, type = c("l","g")))

####Exploring the means####
mean_accuracies.melt <- melt(mean_accuracies.long, measure.vars = "accuracy")
dcast(. ~ block_count, data = mean_accuracies.melt, mean)
# block 1   block 2   block 3
# 0.980404 0.9175758 0.9488889

####Anova to test ####

fit.uni <- aov(accuracy ~ block_count, data = mean_accuracies.long)
summary(fit.uni)
# F(2, 96) = 35.05, p < 0.0001


#############################################################################

library(tidyverse)
library(rstatix)
library(ggpubr)
mean_accuracies.long %>% sample_n_by(block_count, size = 1)

#Summary statistics
mean_accuracies.long %>%
  group_by(block_count) %>%
  get_summary_stats(accuracy, type = "mean_sd")

#Anova test to compare means
res.aov <- mean_accuracies.long %>% anova_test(accuracy ~ block_count)
res.aov

#pairwise t-test for multiple groups
pwc <- mean_accuracies.long %>%
  pairwise_t_test(accuracy ~ block_count, p.adjust.method = "bonferroni")
pwc

# Show adjusted p-values
bxp <- ggboxplot(mean_accuracies.long, x = "block_count", y = "accuracy", 
                 color = "black", palette = "jco")
bxp

pwc <- pwc %>% add_xy_position(x = "block_count")
bxp +
  stat_pvalue_manual(pwc, label = "p.adj.signif", step.increase = 0.08) +
  labs(
    subtitle = get_test_label(res.aov, detailed = TRUE),
    caption = get_pwc_label(pwc)
  )

