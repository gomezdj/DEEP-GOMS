source("R/functions.R")

# Load integrated data
integrated_data <- read.csv("results/integrated_data.csv")

# Train model
model_results <- train_model(integrated_data)

# Save model and accuracy
saveRDS(model_results$model, "results/model.rds")
write.csv(data.frame(accuracy = model_results$accuracy), "results/model_accuracy.csv", row.names = FALSE)