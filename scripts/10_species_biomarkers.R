library(randomForest)
library(tibble)
library(dplyr)

# Load data and model
load("results/all_data_with_features.RData")

# Train RF model on full data (can use the model trained during LODO if saved)
rf_model <- randomForest(features, as.factor(labels), importance = TRUE, ntree = 100)

# Extract feature importances
importance <- importance(rf_model)
importance <- data.frame(Feature = rownames(importance), MeanDecreaseGini = importance[, "MeanDecreaseGini"])
importance <- importance[order(-importance$MeanDecreaseGini), ]

# Filter top biomarkers
top_biomarkers <- importance %>% top_n(30, MeanDecreaseGini)

# Save biomarker results
write.csv(top_biomarkers, "results/top_biomarkers.csv", row.names = FALSE)

# Plot feature importances
library(ggplot2)
ggplot(top_biomarkers, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 30 Biomarkers", x = "Feature", y = "Mean Decrease in Gini")

ggsave("results/top_biomarkers_importance.pdf")