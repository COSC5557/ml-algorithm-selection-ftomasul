# Finn Tomasula Martin
# COSC-4557
# ML Algorithm Selection
# This file contains code testing the performance of various machine learning algorithms run on the same dataset.

# Clear environment
rm(list = ls())
while (!is.null(dev.list())) dev.off()

# Load libraries
library(caret)
library(randomForest)
library(e1071)

# Load in data
wine <- read.csv("winequality-red.csv", sep=";")

# Add new new column to classify wine as good or bad based on a cutoff
cutoff <- mean(wine$quality)
wine$quality.bin <- ifelse(wine$quality >= cutoff, "good", "bad")
wine$quality.bin <- factor(wine$quality.bin)

# Split data for 10-fold cross validation 
set.seed(123)
num_folds <- 10
folds <- createFolds(wine$quality.bin, k = num_folds, list = TRUE, returnTrain = TRUE)

# Run logistic regression on the 10 folds
logis_reg_accuracy <- numeric(num_folds)

for(i in 1:num_folds) {
  indices <- folds[[i]]
  train <- wine[indices, ]
  test <- wine[-indices, ]
  model <- glm(quality.bin ~ .-quality, data = train, family = binomial)
  probs <- predict(model, newdata = test, type = "response")
  len <- length(test$quality.bin)
  preds <- rep("bad", len)
  preds[probs > 0.5] <- "good"
  correct <- sum(preds == test$quality.bin)
  logis_reg_accuracy[i] <- correct / len
}

# Run random forest on the 10 folds
rand_for_accuracy <- numeric(num_folds)

for(i in 1:num_folds) {
  indices <- folds[[i]]
  train <- wine[indices, ]
  test <- wine[-indices, ]
  model <- randomForest(quality.bin ~.-quality, data = train)
  preds <- predict(model, newdata = test)
  len <- length(test$quality.bin)
  correct <- sum(preds == test$quality.bin)
  rand_for_accuracy[i] <- correct / len
}

# Run SVM on the 10 folds
svm_accuracy <- numeric(num_folds)

for(i in 1:num_folds) {
  indices <- folds[[i]]
  train <- wine[indices, ]
  test <- wine[-indices, ]
  model <- svm(quality.bin ~.-quality, data = train)
  preds <- predict(model, newdata = test)
  len <- length(test$quality.bin)
  correct <- sum(preds == test$quality.bin)
  svm_accuracy[i] <- correct / len
}

# Check the results
logis_reg_mean <- mean(logis_reg_accuracy)
rand_for_mean <- mean(rand_for_accuracy)
svm_mean <- mean(svm_accuracy)

data <- data.frame(
  Group = rep(c("Logistic Regression", "Random Forest", "SVM"), each = 10),
  Value = c(logis_reg_accuracy, rand_for_accuracy, svm_accuracy)
)

boxplot(Value ~ Group, data = data, col = c("blue", "green", "red"), main = "Model Comparison", xlab = "Model", ylab = "Values")




