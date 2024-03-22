# Import dataset
pixel_numbers <- seq(1, 784)
x_column_names <- paste("pixel", pixel_numbers, sep = "")
y_column_names <- c("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
X.train <- as.matrix(read.csv('X_train.csv', header = FALSE, col.names = x_column_names))
X.test <- as.matrix(read.csv('X_test.csv', header = FALSE, col.names = x_column_names))
Y.train <- as.matrix(read.csv('Y_train.csv', header = FALSE, col.names = y_column_names))
Y.test <- as.matrix(read.csv('Y_test.csv', header = FALSE, col.names = y_column_names))

train_set <- cbind(X.train, Y.train)
test_set <- cbind(X.test, Y.test)

# Create logistic function without using glm function
# Define sigmoid activation function
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# Using cross-entropy as the loss function
cross_entropy <- function(X, y, theta) {
  m <- length(y)
  predictions <- sigmoid(X %*% theta)
  cost <- -(1 / m) * sum(y * log(predictions) + (1 - y) * log(1 - predictions))
  return(cost)
}

# We are going to use gradient descent to find the optimal theta value
gradient_descent <- function(X, y, theta, alpha, num_iters) {
  m <- length(y)
  cost_history <- numeric(num_iters)
  
  for (i in 1:num_iters) {
    prediction <- sigmoid(X %*% theta)
    errors <- prediction[, 1] - y
    updates <- alpha * (1 / m) * (t(X) %*% errors)
    theta <- theta - updates
    cost_history[i] <- cross_entropy(X, y, theta)
  }
  
  list(theta = theta, cost_history = cost_history)
}

# Tune theta using training set
alpha <- 0.01
num_iters <- 1000
initial_beta <- rep(0, ncol(X.train))

# Tune logistic regression beta using
results <- lapply(1:ncol(Y.train), function(i) {
  gradient_descent(X.train, Y.train[, i], theta = initial_beta,
                   alpha = alpha, num_iters = num_iters)
})




