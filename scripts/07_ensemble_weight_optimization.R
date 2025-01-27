library(caret)
library(e1071)

ensemble_fitness_function <- function(weights, base_learners, train_data, test_data) {
  partitions <- createDataPartition(train_data$response, p = 0.25, times = 4, list = TRUE)
  
  S <- 0
  for (partition in partitions) {
    train_indices <- unlist(partition)
    train_set <- train_data[-train_indices,]
    test_set <- train_data[train_indices,]
    
    predictions <- sapply(base_learners, function(learner) {
      model <- train(as.formula(paste("response ~", paste(names(train_data)[-1], collapse = "+"))),
                     data = train_set, method = learner$method)
      predict(model, test_set)
    })
    
    aggregated_predictions <- colSums(predictions * weights)
    S <- S + sum(aggregated_predictions == test_set$response)
  }
  
  return(S)
}

weights <- runif(length(base_learners))  # Initialize weights
weights <- weights / sum(weights)  # Normalize
fitness <- ensemble_fitness_function(weights, base_learners, train_data, test_data)