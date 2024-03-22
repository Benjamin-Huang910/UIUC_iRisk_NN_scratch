# Import dataset
pixel_numbers <- seq(1, 784)
x_column_names <- paste("pixel", pixel_numbers, sep = "")
y_column_names <- c("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
X.train <- read.csv('X_train.csv', header = FALSE, col.names = x_column_names)
X.test <- read.csv('X_test.csv', header = FALSE, col.names = x_column_names)
Y.train <- read.csv('Y_train.csv', header = FALSE, col.names = y_column_names)
Y.test <- read.csv('Y_test.csv', header = FALSE, col.names = y_column_names)

train_set <- cbind(X.train, Y.train)
test_set <- cbind(X.test, Y.test)

# Build logistic regression for each number
lm.fit0 <- glm(Zero ~. -One - Two - Three - Four - Five - Six - Seven - Eight - Nine,
               data = train_set, family = "binomial")
lm.fit1 <- glm(One ~. - Zero - Two - Three - Four - Five - Six - Seven - Eight - Nine,
               data = train_set, family = "binomial")
lm.fit2 <- glm(Two ~. - Zero - One - Three - Four - Five - Six - Seven - Eight - Nine,
               data = train_set, family = "binomial")
lm.fit3 <- glm(Three ~. - Zero - One - Two - Four - Five - Six - Seven - Eight - Nine,
               data = train_set, family = "binomial")
lm.fit4 <- glm(Four ~. - Zero - One - Two - Three - Five - Six - Seven - Eight - Nine,
               data = train_set, family = "binomial")
lm.fit5 <- glm(Five ~. - Zero - One - Two - Three - Four - Six - Seven - Eight - Nine,
               data = train_set, family = "binomial")
lm.fit6 <- glm(Six ~. - Zero - One - Two - Three - Four - Five - Seven - Eight - Nine,
               data = train_set, family = "binomial")
lm.fit7 <- glm(Seven ~. - Zero - One - Two - Three - Four - Five - Six - Eight - Nine,
               data = train_set, family = "binomial")
lm.fit8 <- glm(Eight ~. - Zero - One - Two - Three - Four - Five - Six - Seven - Nine,
               data = train_set, family = "binomial")
lm.fit9 <- glm(Nine ~. - Zero - One - Two - Three - Four - Five - Six - Seven - Eight,
               data = train_set, family = "binomial")


# Define softmax function
softmax <- function(x) {
  exp_x <- exp(x)
  return(exp_x / rowSums(exp_x))
}

# Function to predict using the model and new data
predict_response <- function(model, newdata = NULL) {
  as.vector(predict(model, type = "response", newdata = newdata))
}

# --- Training set predictions ---
train.preds <- lapply(list(lm.fit0, lm.fit1, lm.fit2, lm.fit3, lm.fit4, lm.fit5, lm.fit6, lm.fit7, lm.fit8, lm.fit9), predict_response)
train.result.matrix <- do.call(cbind, train.preds)

# Apply softmax and find the predicted classes
softmax_predictions_train <- softmax(train.result.matrix)
predicted_classes_train <- max.col(softmax_predictions_train) - 1
actual_train <- max.col(Y.train) - 1
train_accuracy <- sum(predicted_classes_train == actual_train) / length(actual_train)

# --- Testing set predictions ---
test.preds <- lapply(list(lm.fit0, lm.fit1, lm.fit2, lm.fit3, lm.fit4, lm.fit5, lm.fit6, lm.fit7, lm.fit8, lm.fit9), function(model) predict_response(model, newdata = test_set))
test.result.matrix <- do.call(cbind, test.preds)

# Apply softmax and find the predicted classes
softmax_predictions_test <- softmax(test.result.matrix)
predicted_classes_test <- max.col(softmax_predictions_test) - 1
actual_test <- max.col(Y.test) - 1
test_accuracy <- sum(predicted_classes_test == actual_test) / length(actual_test)

# Print the accuracies
print(paste("Training Set Accuracy:", round(train_accuracy, 3)))
print(paste("Testing Set Accuracy:", round(test_accuracy, 3)))
