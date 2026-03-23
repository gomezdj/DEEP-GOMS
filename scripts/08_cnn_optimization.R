library(keras)
library(kerastuneR)
library(tfruns)

# Define fixed parameters (Data dependent)
INPUT_SHAPE <- c(28, 28, 1)

# Fitness Function
# The 'x' argument is the vector of values provided by the GA
cnn_fitness_function <- function(x) {
  
  # Decode the GA vector (floats) into model parameters (integers)
  # x[1] = filters, x[2] = kernel_size, x[3] = pool_size, x[4] = dense_units

  filters       <- floor(x[1])
  kernel_size   <- floor(x[2])
  pool_size     <- floor(x[3])
  dense_units   <- floor(x[4])

  # Constraint checks (Kernel size cannot be larger than input)
  if(kernel_size < 1) kernel_size <- 1
  if(pool_size < 1) pool_size <- 1  
  
  # Build Model
  model <- keras_model_sequential() %>% 
    layer_conv_2d(filters = filters, 
                  kernel_size = c(kernel_size, kernel_size), 
                  activation = 'relu',
                  padding = 'same', # 'same' prevents shape errors with large kernels
                  input_shape = INPUT_SHAPE) %>% 
    layer_max_pooling_2d(pool_size = c(pool_size, pool_size)) %>% 
    layer_flatten() %>% 
    layer_dense(units = dense_units, activation = 'relu') %>% 
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

# Train Model
# verbose = 0 is crucial to keep the console clean
history <- model %>% fit(
  x_train, y_train,
  epochs = 3,           # Lower epochs for speed during search
  batch_size = 64,      # Larger batch size for speed
  validation_split = 0.2,
  verbose = 0
)

# Get validation accuracy
score <- max(history$metrics$val_accuracy)

# Clean up memory (Important in R Keras loops)
k_clear_session()
gc()

  return(score)
}

# 3. Run GA Optimization
ga_optimization <- ga(
  type = "real-valued",
  fitness = cnn_fitness_function,
  # Define bounds for: filters, kernel_size, pool_size, dense_units
  lower = c(32, 3, 2, 64),
  upper = c(128, 7, 3, 256),
  popSize = 10,       # Reduced for demonstration speed
  maxiter = 5,        # Reduced for demonstration speed
  run = 3,            # Stop if no improvement after 3 gens
  parallel = TRUE,    # ENABLE PARALLEL PROCESSING
  monitor = TRUE
)

# 4. View Best Solution
summary(ga_optimization)
print(ga_optimization@solution)
