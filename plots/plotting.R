library(plyr)
library(tidyverse)
library(ggplot2)


######################################
#         HELPER FUNCTIONS          #
#####################################

# Create and save a plot for a single metric and model type
create.single.plot <- function(ntype, metric, data, name){
  df = data[data[, 1] == ntype, ]
  
   # stratification options to serperate models and setups that were used for training
    save.plot(apply.theme(ggplot(data = df, 
                                 aes_string(x =  "epoch", y = metric, color = "normalized", group = "normalized")) +
                            geom_line() + 
                            facet_grid(pretrained ~ full_data)), 
              prefix = name, 
              type = ntype, 
              metric = metric, 
              suffix = "normalized")
    
    save.plot(apply.theme(ggplot(data = df, 
                                 aes_string(x =  "epoch", y = metric, color = "full_data", group = "full_data")) +
                            geom_line() + 
                            facet_grid(pretrained ~ normalized)), 
              prefix = name, 
              type = ntype, 
              metric = metric, 
              suffix = "full_data")
    
    save.plot(apply.theme(ggplot(data = df, 
                                 aes_string(x =  "epoch", y = metric, color = "pretrained", group = "pretrained")) +
                            geom_line() + 
                            facet_grid(full_data ~ normalized)), 
              prefix = name, 
              type = ntype, 
              metric = metric, 
              suffix = "pretrained")
  
}

# Create and save a plot for a single metric grouped by model type
create.plot <- function(metric, data){
  name <- colnames(data)[1]
  sapply(unique(dat$pretrained), (function(x)  save.plot(apply.theme(ggplot(data=(data %>% filter(pretrained == x)), 
                                                                            aes_string(x="epoch", y = metric, color = name, group = name)) +
                                                                       geom_line() + 
                                                                       facet_grid(full_data ~ normalized)), 
                                                         prefix = name, 
                                                         type = "full", 
                                                         metric = metric, 
                                                         suffix = x)))
  
  net_type_unique <- levels(pull(data, 1))
  sapply(net_type_unique, create.single.plot, metric = metric, data = data, name = name)
}

# Create and save plot for a single metric for all nets of a family
create.plot.fam <- function(metric, data, name){
  
  if(name != "VGG"){
    data <- data %>% 
      filter(pretrained == x)
  }
  
  sapply(levels(net_families_base$pretrained), (function(x)  save.plot(apply.theme(ggplot(data = data, 
                                                                                          aes_string(x="epoch", y = metric, color = "net_type", group = "net_type")) +
                                                                                     geom_line() + 
                                                                                     facet_grid(full_data ~ normalized)), 
                                                                       prefix = name, 
                                                                       type = "full-family", 
                                                                       metric = metric, 
                                                                       suffix = x)))
}


# Save plot with same size
save.plot <- function(plot, prefix,  type,  metric, suffix){
  ggsave(
    paste(paste(prefix, type, metric, suffix, sep = "-"), "png", sep = "."),
    plot = plot,
    device = "png",
    path = ".",
    scale = 1,
    width = 40,
    height = 15,
    units = "cm",
    dpi = 300,
    limitsize = FALSE
  )
}

# Apply theme to plots for unified design
apply.theme <- function(plot){
  color.scheme <- c("#0070C0", "#F29F05", "#A4D955", "#6D33A6", "#BF0404", "#00B0F0")
  plot + 
    theme_bw() + 
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
    scale_color_manual(values = color.scheme) +
    scale_fill_manual(values = color.scheme) + 
    theme(axis.title = element_text(size = 18),
          axis.text = element_text(size = 12),
          legend.title = element_text(size = 18),
          legend.text = element_text(size = 12))
}

# Round (metric) values to percentages with 2 decimals
rounded_perc <- function(value){
  round(value * 100, 2)
}



######################################
#          PLOT GENERATION          #
#####################################

# Load data exported from neptune (file must be in same working directory)
dat <- read_csv("neptune_logs.csv", na = c("NA"))

# Preprocessing to calculated means for the test metrics for the different types of models (e.g. VGG11, ResNet18, DenseNet121)
net_types <- dat %>%
  mutate(net_type = as.factor(net_type),
         epoch = as.factor(x),
         pretrained = as.factor(pretrained),
         normalized = as.factor(normalized),
         full_data = as.factor(full_data)) %>%
  filter(!(net_family == "VGG" & optimizer == "Adam")) %>% # exclude plots with Adam Optimizer for VGG as these runs didn't yield useful results
  group_by(net_type, pretrained, normalized, full_data, epoch) %>%
  summarise(avg_valid_acc = mean(valid_acc),
            avg_valid_f1 = mean(valid_f1),
            avg_valid_precision = mean(valid_precision),
            avg_valid_recall = mean(valid_recall),
            avg_valid_roc_auc = mean(valid_roc_auc),
            avg_train_loss = mean(train_loss, na.rm = TRUE)) 


# Preprocessing to calculated means for the test metrics for the different types of models families (e.g. VGG, ResNet, DenseNet)
# To account for different number of runs between the model types within a family a weighted average is used, where each types mean contributes the same
net_families_base <- dat %>%
  mutate(net_family = as.factor(net_family),
         epoch = as.factor(x),
         pretrained = as.factor(pretrained),
         normalized = as.factor(normalized),
         full_data = as.factor(full_data)) %>%
  filter(!(net_family == "VGG" & optimizer == "Adam")) %>%
  group_by(net_family, net_type, pretrained, normalized, full_data, epoch) %>%
  summarise(avg_valid_acc = mean(valid_acc),
            avg_valid_f1 = mean(valid_f1),
            avg_valid_precision = mean(valid_precision),
            avg_valid_recall = mean(valid_recall),
            avg_valid_roc_auc = mean(valid_roc_auc),
            avg_train_loss = mean(train_loss, na.rm = TRUE))


net_families <- net_families_base %>%
  group_by(net_family, pretrained, normalized, full_data, epoch) %>%
  summarise(avg_valid_acc = mean(avg_valid_acc),
            avg_valid_f1 = mean(avg_valid_f1),
            avg_valid_precision = mean(avg_valid_precision),
            avg_valid_recall = mean(avg_valid_recall),
            avg_valid_roc_auc = mean(avg_valid_roc_auc),
            avg_train_loss = mean(avg_train_loss, na.rm = TRUE))

# Definition of the metrics that plots will be created for
metrics = c("avg_valid_acc", 
            "avg_valid_f1", 
            "avg_valid_precision", 
            "avg_valid_recall", 
            "avg_valid_roc_auc",
            "avg_train_loss")

  # Create plots for different model types
  sapply(metrics, create.plot, data = net_types)
  
  # Create plots for different model families
  sapply(metrics, create.plot, data = net_families)

  
  
  sapply(metrics, create.plot.fam, data = net_families_base %>%
           filter(net_family == "DenseNet"), name = "DenseNet")
  
  sapply(metrics, create.plot.fam, data = net_families_base %>%
           filter(net_family == "ResNet"), name = "ResNet")
  
  sapply(metrics, create.plot.fam, data = net_families_base %>%
           filter(net_family == "VGG"), name = "VGG")
  

  ######################################
  #          TABLE GENERATION         #
  #####################################

# Preprocessing for table with average values of model families
# The same weighted averaging strategy as for the plots is applied
tbl <- dat %>%
  mutate(net_type = as.factor(net_type),
         net_family = as.factor(net_family),
         epoch = as.factor(x),
         pretrained = as.factor(pretrained),
         normalized = as.factor(normalized),
         full_data = as.factor(full_data)) %>%
  filter(epoch == "30") %>%
  filter(!(net_family == "VGG" & optimizer == "Adam")) %>%
  group_by(net_family, net_type, pretrained, normalized, full_data) %>%
  summarise(avg_valid_acc = mean(valid_acc),
            avg_valid_f1 = mean(valid_f1),
            avg_valid_precision = mean(valid_precision),
            avg_valid_recall = mean(valid_recall),
            avg_valid_roc_auc = mean(valid_roc_auc)) %>%
  group_by(net_family, pretrained, normalized, full_data) %>%
  summarise(avg_valid_acc = rounded_perc(mean(avg_valid_acc)),
            avg_valid_f1 = rounded_perc(mean(avg_valid_f1)),
            avg_valid_precision = rounded_perc(mean(avg_valid_precision)),
            avg_valid_recall = rounded_perc(mean(avg_valid_recall)),
            avg_valid_roc_auc = rounded_perc(mean(avg_valid_roc_auc))) %>%
  inner_join(dat %>%
               mutate(net_type = as.factor(net_type),
                      net_family = as.factor(net_family),
                      epoch = as.factor(x),
                      pretrained = as.factor(pretrained),
                      normalized = as.factor(normalized),
                      full_data = as.factor(full_data)) %>%
               filter(epoch == "30") %>%
               filter(!(net_family == "VGG" & optimizer == "Adam")) %>%
               group_by(net_family, pretrained, normalized, full_data) %>%
               summarise(min_valid_acc = rounded_perc(min(valid_acc)),
                         max_valid_acc = rounded_perc(max(valid_acc)),
                         min_valid_f1 = rounded_perc(min(valid_f1)),
                         max_valid_f1 = rounded_perc(max(valid_f1)),
                         min_valid_precision = rounded_perc(min(valid_precision)),
                         max_valid_precision = rounded_perc(max(valid_precision)),
                         min_valid_recall = rounded_perc(min(valid_recall)),
                         max_valid_recall = rounded_perc(max(valid_recall)),
                         min_valid_roc_auc = rounded_perc(min(valid_roc_auc)),
                         max_valid_roc_auc = rounded_perc(max(valid_roc_auc))))
                       
# Write results for file for conversion to latex                     
write_csv(tbl, path = "table.csv")

# Preprocessing for table with average values of model types
tbl_2 <- dat %>%
  mutate(net_type = as.factor(net_type),
         epoch = as.factor(x),
         pretrained = as.factor(pretrained),
         normalized = as.factor(normalized),
         full_data = as.factor(full_data)) %>%
  filter(epoch == "30") %>%
  filter(!(net_family == "VGG" & optimizer == "Adam")) %>%
  group_by(net_type, pretrained, normalized, full_data) %>%
  summarise(avg_valid_acc = mean(valid_acc),
            avg_valid_f1 = mean(valid_f1),
            avg_valid_precision = mean(valid_precision),
            avg_valid_recall = mean(valid_recall),
            avg_valid_roc_auc = mean(valid_roc_auc))

write_csv(tbl_2, path = "table2.csv")


# Preprocessing to find models with best result for each metric
base <- dat %>%
  mutate(net_type = as.factor(net_type),
         net_family = as.factor(net_family),
         epoch = as.factor(x),
         pretrained = as.factor(pretrained),
         normalized = as.factor(normalized),
         full_data = as.factor(full_data)) %>%
  filter(!(net_family == "VGG" & optimizer == "Adam")) %>%
  filter(full_data == "full data") %>%
  filter(epoch == "30") 
  

# Find best model for each metric
best.metrics <- base %>% 
  slice_max(valid_acc) %>%
  mutate(result = "best_valid_acc") %>%
  union(base %>% 
          slice_max(valid_f1) %>%
          mutate(result = "best_valid_f1")) %>%
  union(base %>% 
          slice_max(valid_precision) %>%
          mutate(result = "best_valid_precision")) %>%
  union(base %>% 
          slice_max(valid_recall) %>%
          mutate(result = "best_valid_recall")) %>%
  union(base %>% 
          slice_max(valid_roc_auc) %>%
          mutate(result = "best_valid_roc_auc")) %>%
union(base %>% 
        slice_min(valid_acc) %>%
        mutate(result = "worst_valid_acc")) %>%
union(base %>% 
        slice_min(valid_f1) %>%
        mutate(result = "worst_valid_f1")) %>%
  union(base %>% 
          slice_min(valid_precision) %>%
          mutate(result = "worst_valid_precision")) %>%
  union(base %>% 
          slice_min(valid_recall) %>%
          mutate(result = "worst_valid_recall")) %>%
  union(base %>% 
          slice_min(valid_roc_auc) %>%
          mutate(result = "worst_valid_roc_auc"))

# Write results for file for conversion to latex      
write_csv(best.metrics, path = "table_best_worst.csv")


######################################
#         AUC vs RECALL PLOTS       #
#####################################

paper <- net_families %>%
  filter(pretrained == "not pretrained") %>%
  filter(normalized == "not normalized")
  
save.plot(apply.theme(ggplot(data = paper, 
                             aes_string(x =  "epoch", y = "avg_valid_recall", color = "net_family", group = "net_family")) +
                        geom_line() + 
                        facet_grid(full_data ~ .)),
          prefix = "paper", 
          type = "1", 
          metric = "avg_valid_recall", 
          suffix = "paper")

save.plot(apply.theme(ggplot(data = paper, 
                             aes_string(x =  "epoch", y = "avg_valid_roc_auc", color = "net_family", group = "net_family")) +
                        geom_line() + 
                        facet_grid(full_data ~ .)),
          prefix = "paper", 
          type = "1", 
          metric = "avg_valid_roc_auc", 
          suffix = "paper")


save.plot(apply.theme(ggplot(data = paper %>% filter(full_data == "full data"), 
                             aes_string(x =  "epoch", y = "avg_valid_roc_auc", color = "net_family", group = "net_family")) +
                        geom_line()),
          prefix = "paper", 
          type = "2", 
          metric = "avg_valid_roc_auc", 
          suffix = "paper")


######################################
#    CLASS DISTRIBUTION PLOT GEN     #
#####################################

labels <- read_csv("data/train_labels.csv", na = c("NA"))

class.dist <- labels %>%
  select(label) %>%
  mutate(label = revalue(as.factor(label), c("0" = "Negative (0)", "1" = "Positive (1)"))) %>%
  count(label) %>%
  mutate(n_perc = round(n/nrow(labels) * 100, 2))
  

plot <- apply.theme(ggplot(data = class.dist, aes(x = label, y = n, fill = label)) +
  geom_bar(width = 0.6, stat = "identity")) + 
  geom_text(aes(label= paste("n: ", n, " (", n_perc, "0%)", sep = "")), position=position_dodge(width=1), vjust=2, size = 3, color="white") +
  theme(axis.title.x = element_blank(),
                      legend.position = "None")

ggsave(
  "class-distribution.png",
  plot = plot,
  device = "png",
  path = ".",
  scale = 1,
  width = 14,
  height = 5,
  units = "cm",
  dpi = 300,
  limitsize = FALSE
)
