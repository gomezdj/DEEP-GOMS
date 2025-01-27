library(ComplexHeatmap)
library(circlize)

# Load phylogenetic analysis results
load("results/phylogenetic_results.RData")

# Create heatmap for differential analysis
heatmap_matrix <- results$differential_abundance
Heatmap(heatmap_matrix, 
        name = "Differential Abundance", 
        cluster_rows = TRUE, 
        cluster_columns = TRUE, 
        show_row_names = TRUE, 
        show_column_names = TRUE) 

# Load statistical analysis and model results
load("results/stat_analysis_results.RData")

pdf("results/differential_heatmap.pdf")
draw(Heatmap(heatmap_matrix))

# Load results
lodo_results <- read.csv("results/lodo_auc_results.csv")

# Plot AUC results
library(ggplot2)
ggplot(lodo_results, aes(x = Cohort, y = AUC)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(AUC, 2)), vjust = -0.3, size = 3.5) +
  theme_minimal() +
  labs(title = "LODO AUC Results", x = "Cohort", y = "AUC") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save the plot
ggsave("results/lodo_auc_plot.pdf")
dev.off()