# Setup (R verison: 3.6.0)
library(data.table)
library(lattice)
library(ParamHelpers)
library(tidyverse)
library(xgboost)
library(mice)
library(mlr)
library(caret)
library(viridis)
library(GGally)
library(e1071)

# Import tables. 2 files
lifetime_data <- fread('Courier_lifetime_data.csv')
weekly_data <- fread('Courier_weekly_data.csv')

# Task 1

# Lifetime

# Quick exploration
summary(lifetime_data)
glimpse(lifetime_data)
# courier (numeric), feature_1 (categorical) and feature_2 (numeric, NAs present).

# Count NAs (14% in feature_2)
lifetime_data <- lifetime_data %>%
  mutate(is_na = ifelse(is.na(lifetime_data$feature_2), T, F))

# Check how many NAs we have
lifetime_data %>%
  count(is_na) # ~12% of the elements in feature 2 are NAs

# Count NAs grouped by feature_1 and plot
lifetime_data %>%
  group_by(feature_1) %>%
  summarise(sum(is_na)/n())

ggplot(lifetime_data, aes(x = feature_1, fill = is_na))+
  geom_bar(colour = "black")+
  ggtitle("Lifetime's feature_1 Barplot")+
  theme_minimal()   # More NAs in 'c' in proportion due to low sample size of 'c'.

# Delete is_na column
lifetime_data <- lifetime_data %>%
  select(-is_na)

# Plot feature_2 values and histogram
plot(lifetime_data$feature_2, main = "Overview of Lifetime's feature 2")
ggplot(lifetime_data, aes(x = feature_2))+
  geom_histogram(binwidth = 1)+
  theme_minimal()+
  ggtitle("Histogram of feature_2")
# We see two distributions (though they are difficult to see due to outliers).

# Check outliers in feature_2 and filter them out
lifetime_data %>%
  drop_na() %>%
  filter(feature_2 > 200) # Upper, 3 instances above 900.

lifetime_data %>%
  drop_na() %>%
  filter(feature_2 < 0)  # Lower (negative)

lifetime_filtered <- lifetime_data %>%
  drop_na() %>%
  filter(feature_2 %in% -1:75)
# It is difficult to deal with outliers in this case, since we do not know what feature_2 means. If, for instance, feature_2 represents a quantity related to time such as ETA, the negative instances do not male sense. Anyhow, they do not seem to be just values that differ from the mean, so we filter them out regardless.

# feature_2 histograms grouped by feature_1
ggplot(data = lifetime_filtered, aes(x = feature_2))+
  geom_histogram(binwidth = 1)+
  facet_wrap(~feature_1, scales = 'free')+
  theme_minimal()+
  ggtitle("Histrograms of feature_2 by feature_1")
# The distributions do not show imputing NAs with zero or median/mean values is a good idea.

# Use MICE instead
set.seed(666)
mice_mod <- mice(lifetime_filtered, method='rf')
mice_output <- complete(mice_mod)

# Replace feature_2 variable by the mice_output. Check there are no NAs in feature_2 anymore
lifetime_filtered$feature_2 <- mice_output$feature_2
sum(is.na(lifetime_filtered$feature_2))

# Plot feature_2 distributions with and without MICE to check they are the same
par(mfrow = c(1,2))
hist(lifetime_filtered$feature_2, freq  = F, main = 'feature_2: Original Data',
     col = 'darkgreen', ylim=c(0,0.04))
hist(mice_output$feature_2, freq = F, main='feature_2: MICE Output',
     col='lightgreen', ylim=c(0,0.04))



# Weekly

# Quick exploration
summary(weekly_data)
glimpse(weekly_data)
sum(is.na(weekly_data))
# courier (numeric), week (numeric), feature_1 (numeric), ..., feature_17 (numeric). No NA instances present.

# Plot #working weeks vs % couriers
weekly_data %>%
  group_by(courier) %>%
  summarise(n = n()) %>%
  ggplot() +
  geom_bar(aes(x = as.factor(n), y  = ..count../sum(..count..)*100, fill = (..count..))) +
  theme_minimal() +
  scale_fill_viridis(option = "D") +
  theme(legend.position = "NONE") +
  xlab("Number of Weeks worked") +
  ylab("(%)")+
  ggtitle("Barplot of total worked weeks")
# Most of the couriers work 1-4 weeks

# Clean data according to guidance
clean_weekly <- weekly_data %>%
  group_by(courier) %>%
  mutate(max_week = max(week)) %>%
  mutate(target = ifelse(max_week >= 9, 1, 0) %>% as.factor()) %>%
  filter(week < 9) %>%
  select(-max_week)

# Merge the two databases into one
complete_data <- clean_weekly %>%
  inner_join(
    lifetime_filtered %>%
      rename(feature_18 = feature_1,
             feature_19 = feature_2)
  ) %>%
  mutate(feature_18 = ifelse(feature_18 == 'a', 1, ifelse(feature_18 == 'b', 2, ifelse(feature_18   == 'c', 3, 4)))) %>%
  ungroup()

# Explore features and correlations with ggpairs()
features = c("feature_4", "feature_5")
ggpairs(weekly_data,
        columns = features, aes(alpha = 0.2))
correlations <- round(cor(complete_data[,-20]), 1) # Remove column 20 (target) and compute.
# feature_4 and feature 5 are completely uncorrelated, cor(4,5) = -1. feature_3 and feature_11 are highly correlated, cor(3,11) ~ 0.9. And so on. The variables coming from lifetime_filtered are not particularly highly correlated.

# Plot density distribution of values distributed by target
complete_data %>%
  melt(id.vars = c('courier', 'week', 'target')) %>%
  ggplot(aes(x = value, fill = target)) +
  geom_density(alpha = 0.5) +
  facet_wrap(variable~., scales = 'free')
# feature_2, feature_3 and feature_11 are the features that separate the variable target best.

# It would be wise to perform feature engineering to aim for better artifficial features. Nevertheless, since our aim is not getting highest performance we can work with these. It should also be pointed out that perform such a task would require the knowledge of the meaning of the features, for without that it is very difficult to know how to combine features for generating better ones that separate classes better, which is perhaps the weakest point of the model. That could be done automatically with a neural network, for instance. However, that would not be too wise given the low sample size (3k is not too much) and the simple nature of the given data (neural nets are used for very difficult automatic tasks such as image or speech recognition).

# Another technique that is often done prior to classification is to standarize data. For instance, subtracting from every intsance its vector mean and dividing by its vector standard deviation. This is done to ease the task of classifiers that are sensitive to scaling, such as neural networks. Algorithms based on trees, however do not need such thing to perform well.


# Task 2

# Tha algorithm used for predicting couriers churn is the XGBoost. It is widely used nowadays in Kaggle competitions for its efficiency and good performance in supervised machine learning problems. It is especially recommended when variables are numerical and not categorical (such as the Titanic dataset) and also for small datasets.

# Apart from this, it enables L2 regularization to reduce overfitting and cross-validation setup. It handles missing values by itself (although those were imputed in this case). It can work for regression and ranking problmes, as well as classification. It runs on several platforms such as R, Python, Java, Julia, and Scala. And finally, it is possible to save results in the computer to avoid repeating computations.

# Alternative algorithms worth mentioning:

# Logistic regression: it works well for small datasets, but may overfit.
# K-NN: computationally expensive for large datasets, but handles overfitting better.
# More complex classifiers may be overkill, but worth a try: SVM, NNs, ...


# Define dataset for training and testing
tr_te <- complete_data %>%
  mutate(target = as.numeric(target)) %>%
  select(-courier, -week, -feature_4)
# Removing feature_4 (or feature_5) will not have any negative effect on the performance, since they are perfectly negatively correlated. One of them can thus be removed. Furthermore, emoving redundant information may increase performance.

# Select 80% of data for training, 20% for testing
tr <- 1:floor(nrow(tr_te)*0.8)

train <- tr_te[tr,]
test <-  tr_te[-tr,]

# Select samples for training and testing randomly and generate datasets for XGBoost.
train_model <- function(train, test){
  y <- train$target
  tri_val <- sample(seq_along(y), length(y)*0.1)
  tr <- seq_along(y)[!(seq_along(y) %in% tri_val)]
  y_test <- test$target

  tr_te <- bind_rows(train,test)
  train_xgb <- xgb.DMatrix(tr_te %>% select(-target) %>% .[tr,] %>%  as.matrix(), label = y[tr])
  val_xgb <- xgb.DMatrix(tr_te %>% select(-target) %>% .[tri_val,] %>%  as.matrix(), label = y[tri_val])
  test_xgb <- xgb.DMatrix((tr_te %>% .[-(c(tr,tri_val)),] %>% select(-target) %>%  as.matrix()), label = y_test)

  # Optimize hyperparameters. Run the file tunexgb.R first to get the variable tune_xgb
  tuning_scores <- train %>% sample_n(1e3) %>% tune_xgb(target_label = 'target', eval_metric = 'auc', fast = TRUE) # "error", alternatively

  # Select best parameters
  m <- which.max(tuning_scores$scores)
  currentSubsampleRate <- tuning_scores[["subsample"]][[m]]
  currentColsampleRate <- tuning_scores[["colsample_bytree"]][[m]]
  lr <- tuning_scores[["lr"]][[m]]
  mtd <- tuning_scores[["mtd"]][[m]]
  mcw <- tuning_scores[["mcw"]][[m]]
  # currentLambda <- tuning_scores[["lambda"]][[m]]

  # Options list for the algorithm
  ntrees <- 1e3
  p <- list(objective = "binary:logistic",
            booster = "gbtree",
            eval_metric = "auc", # "error", alternatively
            nthread = 4,
            eta = lr/ntrees,
            max_depth = mtd,
            min_child_weight = 30,
            gamma = 0,
            subsample = currentSubsampleRate,
            colsample_bytree = currentColsampleRate,
            colsample_bylevel = 0.632,
            alpha = 0,
            lambda = 0.001,
            nrounds = ntrees)


  # xgb_setup <- xgb.train(p, test_xgb, 1000, list(val = val_xgb), print_every_n = 10, early_stopping_rounds = 300)  # run this instead to be able to use predict() and comput confusion matrix

  xgb_setup_cv <- xgb.cv(p, train_xgb, 1000, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stop_round = 300, maximize = F)  # run this to get a more realistic sense of the performance of the system with a cross-validation setup
}


# Task 3


# The metric chosen is the ROC AUC score, commonly used in a binary classification problem. The ROC AUC is the area under the receiver-operating characteristic. The ROC curve represents the false positive rate (FPR = FP  /(FP + TN)) and the true positive rate (TPR = TP / TP + FN).

# It tells information about about the separability of the classes, without the need of choosing a threshold.  It is thus a very powerful metric for binary classifiers, and the one I would go for to present a performance metric given the problem at hand. A value closer to 1 means means the classifier distinguishes better between classes.

# It is worth exploring other metrics. An alternative would be the error rate, that accounts for the proportion of wrongly classified instances. These metrics can be computed from the confusion matrix (CM).

#Another viable alternative could be the log-likelihood loss, which penalizes false classifications. This last one is however less intuitive to interpret, since it ranges from 0 to infinity, whereas the others range from 0 to 1.

# Train and validate model
xgb_model <- train_model(train,test)
# it takes ~2min (on this laptop) while to compute due to the hyperparameter tuning.

xgb_pred <- predict(xgb_model, test_xgb)  # run lines from y <- ... to test_xgb <- ..., first
xgb_pred <- ifelse(xgb_pred > 0.5, 1, 0)  # get predicted labels. Threshold set to 0.5
confusion_matrix <- as.data.frame(table(xgb_pred, y_test))  # get confusion matrix

# Plot confusion matrix
ggplot(data = confusion_matrix,
       mapping = aes(x = y_test,
                     y = xgb_pred)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "blue",
                      high = "red",
                      trans = "log")

# Results:

#       Normal Setup          CV Setup
# auc: (0.7534),        (0.8029+0.0184)
# error: (0.2541)       (0.2289+-0.0057)



# Plot variables' importance in the classification algorithm. Each bar represents a value which tells how much a variable has participated in the decision making during the algorithm. This diagram is important to get a glance of what features are more relevant to the algorithm, and diagnose what is more conveninet for a company, for instance, from a business point of view.
xgb.importance(model = xgb_model_cv) %>%
  ggplot(aes(x = reorder(Feature, Gain), y = Gain, fill = Gain)) +
  geom_col() +
  coord_flip() +
  scale_fill_viridis(begin = 0.2, direction = - 1)
# Some of the most relevant ones coincide with variables a priori important for solving the problem such as feature_2 and feature_3. feature_11, however, did not participate as much. This is not necessarily bad or wrong, since the algorithm involves non-linear dynamics that make some variables a priori important, become less.
