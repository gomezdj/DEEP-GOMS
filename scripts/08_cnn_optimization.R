library(keras)
library(GA)

# Function to initialize a CNN model given a set of parameters
initialize_cnn_model <- function(params) {
  model <- keras_model_sequential() %>% 
    layer_conv_2d(filters = params$filters, kernel_size = params$kernel_size, activation = 'relu',
                  input_shape = c(params$input_shape)) %>% 
    layer_max_pooling_2d(pool_size = c(params$pool_size, params$pool_size)) %>% 
    layer_flatten() %>% 
    layer_dense(units = params$dense_units, activation = 'relu') %>% 
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )
  
  return(model)
}

cnn_fitness_function <- function(params) {
  model <- initialize_cnn_model(params)
  history <- model %>% fit(
    x_train, y_train,
    epochs = 5, batch_size = 32,
    validation_split = 0.2,
    verbose = 0
  )
  max(history$metrics$val_accuracy)
}

ga_optimization <- ga(
  type = "real-valued",
  fitness = cnn_fitness_function,
  lower = c(filters = 32, kernel_size = 3, pool_size = 2, dense_units = 64, input_shape = c(28, 28, 1)),
  upper = c(filters = 64, kernel_size = 5, pool_size = 3, dense_units = 128, input_shape = c(28, 28, 1)),
  popSize = 20, maxiter = 50, run = 10
)

best_solution <- ga_optimization@solution