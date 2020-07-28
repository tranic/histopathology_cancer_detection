library(tidyverse)
library(ggplot2)


dat <- read_csv("/Users/simon/Downloads/neptune_logs.csv", na = c("NA"))

net_type_dat <- dat %>%
  mutate(net_type = as.factor(net_type),
         epoch = as.factor(x),
         pretrained = as.factor(pretrained),
         normalized = as.factor(normalized),
         full_data = as.factor(full_data)) %>%
  group_by(net_type, pretrained, normalized, full_data, epoch) %>%
  summarise(avg_train_acc = mean(train_acc),
            avg_train_f1 = mean(train_f1),
            avg_train_precision = mean(valid_precision),
            avg_train_recall = mean(train_recall),
            avg_train_roc_auc = mean(train_roc_auc),
            avg_train_loss = mean(train_loss, na.rm = TRUE),
            avg_valid_acc = mean(valid_acc),
            avg_valid_f1 = mean(valid_f1),
            avg_valid_precision = mean(valid_precision),
            avg_valid_recall = mean(valid_recall),
            avg_valid_roc_auc = mean(valid_roc_auc),
            avg_valid_loss = mean(valid_loss, na.rm = TRUE)) 



net_fam_dat <- dat %>%
  mutate(net_type = as.factor(net_family),
         epoch = as.factor(x),
         pretrained = as.factor(pretrained),
         normalized = as.factor(normalized),
         full_data = as.factor(full_data)) %>%
  group_by(net_family, pretrained, normalized, full_data, epoch) %>%
  summarise(avg_train_acc = mean(train_acc),
            avg_train_f1 = mean(train_f1),
            avg_train_precision = mean(valid_precision),
            avg_train_recall = mean(train_recall),
            avg_train_roc_auc = mean(train_roc_auc),
            avg_train_loss = mean(train_loss, na.rm = TRUE),
            avg_valid_acc = mean(valid_acc),
            avg_valid_f1 = mean(valid_f1),
            avg_valid_precision = mean(valid_precision),
            avg_valid_recall = mean(valid_recall),
            avg_valid_roc_auc = mean(valid_roc_auc),
            avg_valid_loss = mean(valid_loss, na.rm = TRUE)) 

metrics = c("avg_valid_acc", 
            "avg_valid_f1", 
            "avg_valid_precision", 
            "avg_valid_recall", 
            "avg_valid_roc_auc",
            "avg_valid_loss",
            "avg_valid_acc", 
            "avg_valid_f1", 
            "avg_valid_precision", 
            "avg_valid_recall", 
            "avg_valid_roc_auc",
            "avg_valid_loss")
net_type_unique <- unique(as.character(net_type_dat$net_type))

  # net_type_dat <- net_fam_dat
  sapply(metrics, create.plot)


create.plot <- function(metric){
  plot <- ggplot(data=(net_type_dat %>% filter(pretrained == "pretrained")), aes_string(x="epoch", y=metric, color = "net_type", group = "net_type")) +
     geom_line() + 
     facet_grid(full_data ~ normalized) +
     theme_bw() + 
     scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
     scale_fill_manual(values = c("#6D33A6", "#A4D955", "#F29F05", "#BF0404", "#00B0F0", "#0070C0"))
  
  ggsave(
    paste(paste("netfam", "full", "pretrained", metric, sep = "-"), "png", sep = "."),
    plot = plot,
    device = "png",
    path = "Desktop/Deep Learning",
    scale = 1,
    width = 40,
    height = 15,
    units = "cm",
    dpi = 300,
    limitsize = FALSE)
  
  plot <- ggplot(data=(net_type_dat %>% filter(pretrained != "pretrained")), aes_string(x="epoch", y=metric, color = "net_type", group = "net_type")) +
    geom_line() + 
    facet_grid(full_data ~ normalized) +
    theme_bw() + 
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
    scale_fill_manual(values = c("#6D33A6", "#A4D955", "#F29F05", "#BF0404", "#00B0F0", "#0070C0"))
  
  ggsave(
    paste(paste("netfam", "full", "not-pretrained", metric, sep = "-"), "png", sep = "."),
    plot = plot,
    device = "png",
    path = "Desktop/Deep Learning",
    scale = 1,
    width = 40,
    height = 15,
    units = "cm",
    dpi = 300,
    limitsize = FALSE)
  
  sapply(net_type_unique, create.single.plot, metric = metric)
}


  
create.single.plot <- function(ntype, metric){
  df = net_type_dat %>% filter(net_type == ntype)
  plot <- ggplot(data=df, aes_string(x="epoch", y=metric, color = "normalized", group = "normalized")) +
    geom_line() + 
    facet_grid(full_data ~ pretrained) +
    theme_bw() + 
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
    scale_fill_manual(values = c("#6D33A6", "#A4D955", "#F29F05", "#BF0404", "#00B0F0", "#0070C0"))
  
  ggsave(
    paste(paste("nettype", ntype, metric, "normalized", sep = "-"), "png", sep = "."),
    plot = plot,
    device = "png",
    path = "Desktop/Deep Learning",
    scale = 1,
    width = 40,
    height = 15,
    units = "cm",
    dpi = 300,
    limitsize = FALSE
  )
  
  plot <- ggplot(data=df, aes_string(x="epoch", y=metric, color = "full_data", group = "full_data")) +
    geom_line() + 
    facet_grid(normalized ~ pretrained) +
    theme_bw() + 
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
    scale_fill_manual(values = c("#6D33A6", "#A4D955", "#F29F05", "#BF0404", "#00B0F0", "#0070C0"))
  
  ggsave(
    paste(paste("nettype", ntype, metric, "full_data", sep = "-"), "png", sep = "."),
    plot = plot,
    device = "png",
    path = "Desktop/Deep Learning",
    scale = 1,
    width = 40,
    height = 15,
    units = "cm",
    dpi = 300,
    limitsize = FALSE
  )
  
  plot <- ggplot(data=df, aes_string(x="epoch", y=metric, color = "pretrained", group = "pretrained")) +
    geom_line() + 
    facet_grid(normalized ~ full_data) +
    theme_bw() + 
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
    scale_fill_manual(values = c("#6D33A6", "#A4D955", "#F29F05", "#BF0404", "#00B0F0", "#0070C0"))
  
  ggsave(
    paste(paste("nettype", ntype, metric, "pretrained", sep = "-"), "png", sep = "."),
    plot = plot,
    device = "png",
    path = "Desktop/Deep Learning",
    scale = 1,
    width = 40,
    height = 15,
    units = "cm",
    dpi = 300,
    limitsize = FALSE
  )
}



# Method to save plot to file




# preo

# Innerhalb eines Netzes + Alle Netze zusammen

# Einzelne Metrik --> Ueber Metrik rotaten
# einfach ueber spalten rotaten

# Erst plot fuer alle netze

# Einzelne Netze --> rotieren

# Group by NetType
# Group by pretrained
# Group by normalised # immer 1 plot pro anderer category
# Group by full-data
# Group by epoch

# Group by NetFamily
# Group by pretrained
# Group by normalised
# Group by full-data
# Grouo by epoch