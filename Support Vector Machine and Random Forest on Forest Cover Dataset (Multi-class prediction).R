# To print all the values and override max.print warning
options(max.print = 1000)

# Set working directory
setwd("/Users/abhishekpadalkar/Documents/IRELAND Admissions/NCI/Course Modules/Modules/Sem 1/DMML/Final Project/")

# First we create a train and test function for later model evaluations

create_train_test <- function(data, size = 0.75, train = TRUE) {
  n_row = nrow(data)
  total_row = round(size * n_row)
  set.seed(311)    # Set seed for reproducible results
  tindex <- sample(n_row, total_row)   # Create a random index
  if (train == TRUE) {
    return (data[tindex, ])
  } else {
    return (data[-tindex, ])
  }
}


###### -------------------------------------------- COVER TYPE DATA -------------------------------------------- ######

# Get the data.
forest_cover_data <- read.csv("Forest Cover type/Forest_Cover_Type.csv", header = T)

# Check the Structure
str(forest_cover_data)  # Total of 581012 observations! There are many binary variables of Wilderness Type and Soil Type. These can be computationally intensive, thus we will convert them into categorical.
library(psych)
describe(forest_cover_data)

# Check missing values
sapply(forest_cover_data, function(x) sum(is.na(x))) # => No missing Values

# Checking Duplicates in datasets!
library(dplyr)
forest_cover_data <- distinct(forest_cover_data)
str(forest_cover_data) # No duplicates found!

# EDA
# As this dataset has a binary categorical dependent variable, we first look at how is the distribution of the y values.
count_df <- forest_cover_data %>% 
  group_by(CoverType) %>%
  summarise(no_rows = length(CoverType))
count_df$CoverType <- as.factor(count_df$CoverType)
count_df["Tree_name"] <- factor(c("Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"))
ggplot(count_df, aes(x=Tree_name, y=no_rows/1000, colour = CoverType)) + geom_col(width=0.25) + coord_flip() +
  labs(title="Distribution of Forest Tree types in the data", x="Cover (Tree) Type", y="Count in 1000s'")

# We observe the ratio of the categories in our dependent variable in the dataset first.
prop.table(table(forest_cover_data$CoverType))  

# Before starting modeling process, we sample (stratified) the data to 10000 rows so the modeling takes convinient amount of time.
set.seed(2)
forest_cover_data_sampled <- forest_cover_data[sample(nrow(forest_cover_data), 10000), ]
forest_cover_data_sampled$CoverType <- as.factor(forest_cover_data_sampled$CoverType)   # We factor of multiclass variable for SVM and Random Forest Algorithms
prop.table(table(forest_cover_data_sampled$CoverType))  # The ratios almost match for each variable from the main dataset.

# Create train and test split
train_forest_cover_data <- create_train_test(forest_cover_data_sampled, 0.75, train = TRUE)
test_forest_cover_data <- create_train_test(forest_cover_data_sampled, 0.75, train = FALSE)

# Check if the ratio of covertypes are same in the train and test set
prop.table(table(train_forest_cover_data$CoverType))
prop.table(table(test_forest_cover_data$CoverType))
# They  are almost same for all except for covertype 3 and 4. 

#### SVM ####

library(e1071)

# We create a copy of train and test set for svm to try modeling with non processed data and preprocessed data
# The original train set and test set are used for modeling svm models with linear and radial kernel, and new svm train set and test set are preprocessed to rescaled continuous variables and changing binary categorical variables to -1 and 1.
train_forest_cover_data_svm <- train_forest_cover_data
test_forest_cover_data_svm <- test_forest_cover_data

# As SVMs are binary classifiers inherently and can only have binary categorical variables to work better, our dataset already have binary categorical data. So we don't need to split any categorical into individual category.
# Data Pre-processing: Change binary categorical predictors from 0, 1 to -1, 1 AND rescale all continuous variables to range -1 to 1

# Chaning Binary 0 1 to Binary -1 1 Train and Test
for (i in 11:54){
  for (j in 1:length(train_forest_cover_data_svm)) {
    if (train_forest_cover_data_svm[j, i] == 0){
      train_forest_cover_data_svm[j, i] <- -1
    }
  }
  for (k in 1:length(test_forest_cover_data_svm)) {
    if (test_forest_cover_data_svm[k, i] == 0){
      test_forest_cover_data_svm[k, i] <- -1
    }
  }
}

# Range all continuos variables in [-1, 1]
for (i in 1:10){
  maxtr <- max(train_forest_cover_data_svm[, i])
  mintr <- min(train_forest_cover_data_svm[, i])
  maxte <- max(test_forest_cover_data_svm[, i])
  minte <- min(test_forest_cover_data_svm[, i])
  for (j in 1:length(train_forest_cover_data_svm)) {
    train_forest_cover_data_svm[j, i] <- (2*train_forest_cover_data_svm[j, i] - maxtr - mintr)/(maxtr - mintr)
  }
  for (k in 1:length(test_forest_cover_data_svm)) {
    test_forest_cover_data_svm[k, i] <- (2*test_forest_cover_data_svm[k, i] - maxte - minte)/(maxte - minte) 
  }
}

# Build 4 SVM models. 2 models on original unpre-processed data with Radial and Linear kernels AND 2 models on newly pre-processed data again with Radial and Linear kernels
print("For SVM Radial with non-scaled and unchanged binary variables:")
start_time <- Sys.time()
set.seed(100)
svm_radial_og <- svm(CoverType~., data=train_forest_cover_data, type="C-classification", kernel="radial", gamma=0.1, cost=10)
Sys.time() - start_time
print("Radial SVM for non-scaled and unchanged binary variables Done!")

print("For SVM Linear with non-scaled and unchanged binary variables:")
start_time <- Sys.time()
set.seed(100)
svm_linear_og <- svm(CoverType~., data=train_forest_cover_data, type="C-classification", kernel="linear", cost=10)
Sys.time() - start_time
print("Linear SVM for non-scaled and unchanged binary variables Done!")

print("For SVM Radial with scaled and changed binary variables:")
start_time <- Sys.time()
set.seed(100)
svm_radial_new <- svm(CoverType~., data=train_forest_cover_data_svm, type="C-classification", kernel="radial", gamma=0.1, cost=10)
Sys.time() - start_time
print("Radial SVM for scaled and changed binary variables Done!")

print("For SVM Linear with scaled and changed binary variables:")
start_time <- Sys.time()
set.seed(100)
svm_linear_new <- svm(CoverType~., data=train_forest_cover_data_svm, type="C-classification", kernel="linear", cost=10)
Sys.time() - start_time
print("Linear SVM for scaled and changed binary variables Done!")

# Check Summary of all 4 models
summary(svm_radial_og)
summary(svm_linear_og)
summary(svm_radial_new)
summary(svm_linear_new)

# Make Predictions using each of the above 4 models. And check which performed best based on Accuracy, and Kappa.

# Make prediction for radial Non-Scaled Unchanged Categorical #
predict_svm_radial_og <- predict(svm_radial_og, test_forest_cover_data[,-55],type = "class")
confusionMatrix(as.factor(predict_svm_radial_og), test_forest_cover_data$CoverType)  # Accuracy = 50.24%, Kappa = 0

# Make Prediction for linear Non-Scaled Unchanged Categorical #
predict_svm_linear_og <- predict(svm_linear_og, test_forest_cover_data[,-55],type = "class")
confusionMatrix(as.factor(predict_svm_linear_og), test_forest_cover_data$CoverType)  # Accuracy = 66%, Kappa = 0.4322

# Make prediction for radial Scaled changed Categorical #
predict_svm_radial_new <- predict(svm_radial_new, test_forest_cover_data_svm[,-55],type = "class")
confusionMatrix(as.factor(predict_svm_radial_new), test_forest_cover_data_svm$CoverType)  # Accuracy = 76.52%, Kappa = 0.6129

# Make prediction for linear Scaled changed Categorical #
predict_svm_linear_new <- predict(svm_linear_new, test_forest_cover_data_svm[,-55],type = "class")
confusionMatrix(as.factor(predict_svm_linear_new), test_forest_cover_data_svm$CoverType)  # Accuracy = 72.68%, Kappa = 0.5458

# For each svm model plot ROC Curve and Get Mean AUC
classes <- levels(test_forest_cover_data_svm$CoverType)
all_predictions <- data.frame(predict_svm_radial_og, predict_svm_linear_og, predict_svm_radial_new, predict_svm_linear_new)

for (p in 1:4){
  # For each class
  auc_of_model_p <- vector()
  for (i in 1:7){   # Since 7 classes 
    predict_roc_curve <- all_predictions[p]
    # Change the predictions to binary Based on One VS ALL Multiclass
    predict_roc_curve <- ifelse(predict_roc_curve==i, 1, 0)
    # Define which observations belong to class[i]
    true_values <- ifelse(test_forest_cover_data_svm[,55]==classes[i],1,0)
    # Assess the performance of classifier for class[i]
    pred <- prediction(predict_roc_curve,true_values)
    perf <- performance(pred, "tpr", "fpr")
    if (i==1)
    {
      plot(perf,main="ROC Curve",col=pretty_colours[i], lwd=3)
      legend("bottomright", legend = c(1,2,3,4,5,6,7), fill=pretty_colours, title = "Forest Cover Type")
    }
    else
    {
      plot(perf,main="ROC Curve",col=pretty_colours[i],add=TRUE, lwd=3)
    }
    # # Calculate the AUC and print it to screen
    auc.perf <- performance(pred, measure = "auc")
    auc_of_model_p[i] <- auc.perf@y.values[[1]]
  }
  print(paste0("Mean AUC for model ", p, " is: ", mean(auc_of_model_p)))
}
# "Mean AUC for model 1 is: 0.5"
# "Mean AUC for model 2 is: 0.66564195206022"
# "Mean AUC for model 3 is: 0.774673361460051"    --> Radial with scaled variables
# "Mean AUC for model 4 is: 0.720970466644237"


# Thus we now select SVM with Radial For Converted Data. Now we tweak the hyperparameters to get best subset model
# Create a function to get accuracies, and errors of models based on tuning parameters for getting best tuning parameters.
get_acc_by_tuning_parameters <- function(norow, gamma_values, cost_range){
  accuracy_svm_radial_df <- data.frame(matrix(ncol = 4, nrow = norow))
  colnames(accuracy_svm_radial_df) <- c("Gamma", "Cost", "Accuracy", "Error")
  
  print("Searching Best SVM Radial with scaled and changed binary variables using given tuning parameters:")
  start_time <- Sys.time()
  roww <- 1
  for (gmma in gamma_values) {
    print(paste0("For Gamma: ", gmma))
    print("")
    print("")
    for (cost in 1:cost_range) {
      print(paste0("\tFor Cost: ", cost))
      set.seed(100)
      svm_radial_loop_model <- svm(CoverType~., data=train_forest_cover_data_svm, type="C-classification", kernel="radial", gamma=gmma, cost=cost)
      
      predict_svm_radial_loop <- predict(svm_radial_loop_model, test_forest_cover_data_svm[,-55],type = "class")
      conf_matrix <- confusionMatrix(as.factor(predict_svm_radial_loop), test_forest_cover_data_svm$CoverType)
      
      sum_acc = 0
      total_cases = 0
      
      for (i in 1:7){
        for (j in 1:7){
          total_cases = total_cases + conf_matrix$table[i,j]
          if (i  == j){
            sum_acc = sum_acc + conf_matrix$table[i,j]
          }
        }
      }
      
      acc = sum_acc/total_cases
      print(paste0("\t\tAccuracy: ", acc))
      err = 1 - acc
      
      accuracy_svm_radial_df[roww, ] <- c(gmma, cost, acc, err)
      
      print(Sys.time() - start_time)
      print(paste0("\tCost ",cost," Done!"))
      print("")
      
      roww <- roww + 1
    }
    print(Sys.time() - start_time)
    print(paste0("Gamma ",gmma," Done!"))
    print("")
  }
  Sys.time() - start_time
  print("Search for Radial SVM for scaled and changed binary variables Done!")
  return(accuracy_svm_radial_df)
}

# For this project purpose, we consider gamma values 0.01, 0.05, 0.1, 0.25, 0.5 and cost range from 1:100
acc_for_tuned_models <- get_acc_by_tuning_parameters(500, c(0.01, 0.05, 0.1, 0.25, 0.5), 100)
write_csv(acc_for_tuned_models, "Best Model comparison based on tuning parameters SVM.csv")  # Save this data as we've build data from 500 svms 

# We convert Gamma variable into Factor for better visualization
acc_for_tuned_models$Gamma <- factor(acc_for_tuned_models$Gamma, levels = c(0.01, 0.05, 0.1, 0.25, 0.5))

# Plot This Data for Clear understanding.
ggplot(acc_for_tuned_models, aes(x = Cost, y = Accuracy)) + 
  geom_line(aes(color = Gamma)) +
  scale_color_manual(values = c("#90be6d", "#ff993b", "#000075", "#f9c74f", "#e6194B")) +
  labs(title = "Accuracy for Radial SVM on processed Dataset with with Tuned Gamma and Cost", x = "Cost", y = "Accuracy", color = "Gamma")# +
gghighlight(Accuracy == max(Accuracy), label_key = Accuracy)

# We select highest Accuracy with minimum cost with gamma = 0.05. 
best_cost <- acc_for_tuned_models[acc_for_tuned_models$Accuracy == max(acc_for_tuned_models$Accuracy), "Cost"][1]
best_gamma <- as.numeric(acc_for_tuned_models[acc_for_tuned_models$Accuracy == max(acc_for_tuned_models$Accuracy), "Gamma"][1])

# Since gamma was a factor variable
if (best_gamma == 1){
  best_gamma <- 0.01
}
if (best_gamma == 2){
  best_gamma <- 0.05
}
if (best_gamma == 3){
  best_gamma <- 0.1
}
if (best_gamma == 4){
  best_gamma <- 025
}
if (best_gamma == 5){
  best_gamma <- 0.5
}

# Create best model
set.seed(100)
best_svm_model <- svm(CoverType~., data=train_forest_cover_data_svm, type="C-classification", kernel="radial", gamma=best_gamma, cost=best_cost)

# Perform Prediction based on our best model
predict_svm_radial_best <- predict(best_svm_model, test_forest_cover_data_svm[,-55], type = "class")
confusionMatrix(as.factor(predict_svm_radial_best), test_forest_cover_data_svm$CoverType)

# Multi One vs ALL ROC Curve for our best model and Mean AUC.
# For each class
auc_of_best_model_svm <- vector()
for (i in 1:7){   # Since 7 classes 
  per_class_predictions <- predict_svm_radial_best
  # Change the predictions to binary
  per_class_predictions <- ifelse(per_class_predictions==i, 1, 0)
  # Define which observations belong to class[i]
  true_values <- ifelse(test_forest_cover_data_svm[,55]==classes[i],1,0)
  # Assess the performance of classifier for class[i]
  pred <- prediction(per_class_predictions, true_values)
  perf <- performance(pred, "tpr", "fpr")
  if (i==1)
  {
    plot(perf,main="ROC Curve(One vs All) for Support Vector Machine",col=pretty_colours[i], lwd=3)
    legend("bottomright", legend = c(1,2,3,4,5,6,7), fill=pretty_colours, title = "Forest Cover Type")
  }
  else
  {
    plot(perf,main="ROC Curve",col=pretty_colours[i],add=TRUE, lwd=3)
  }
  # # Calculate the AUC and print it to screen
  auc.perf <- performance(pred, measure = "auc")
  auc_of_best_model_svm[i] <- auc.perf@y.values[[1]]
  # print(auc.perf@y.values)
}
print(paste0("Mean AUC for model is: ", mean(auc_of_best_model_svm)))   # AUC = 0.79349 | Previously best model from the first 4 was AUC = 0.77467 (with gamma=0.1, cost=10)


#### END SVM ####

#### RANDOM FOREST ####

library(randomForest)

## Changing Binary Variables into Categorical for Forest Cover Data To Avoid Curse of Dimensionality while training ##

# First Combine Wilderness Category
forest_wilderness <- forest_cover_data[, 11:14]  # Get Data of Wilderness Area, 4 columns : 4 Types
wilderness_area <- vector(length = nrow(forest_wilderness))

# Merge 4 Wilderness binary columns into 1 Wilderness Area Categorical column with 4 levels
for (row in 1:nrow(forest_wilderness)){
  # names(forest_wilderness[row, ])[which(forest_wilderness[row, ] == 1, arr.ind=T)[, "col"]]    => Get the name of a column where column value is "1"
  # match(names(forest_wilderness[row, ])[which(forest_wilderness[row, ] == 1, arr.ind=T)[, "col"]], names(forest_wilderness))    => Get the index of the column of which the name found above is a match with the columns in the dataframe. This function gives index of column provided name of the column.
  # Save the index as the categorical value of Wilderness Area Type.
  wilderness_area[row] = match(names(forest_wilderness[row, ])[which(forest_wilderness[row, ] == 1, arr.ind=T)[, "col"]], names(forest_wilderness))
}

# Second Combine Soil Type Category
forest_soil_type <- forest_cover_data[, 15:54]  # Get Data of Soil Type, 40 columns : 40 Types
soil_type <- vector(length = nrow(forest_soil_type))

# Merge 40 Soil Type binary columns into 1 Soil Type Categorical column with 40 levels
for (row in 1:nrow(forest_wilderness)){
  soil_type[row] = match(names(forest_soil_type[row, ])[which(forest_soil_type[row, ] == 1, arr.ind=T)[, "col"]], names(forest_soil_type))
}

# Creating new dataframe by removing the binary Wilderness Areas and Soil Type Columns and adding Nominal Data for respective column
forest_cover_data_new <- subset(forest_cover_data, select = -c(11:54))
str(forest_cover_data_new)
forest_cover_data_new$Wilderness_Area <- wilderness_area
forest_cover_data_new$Soil_Type <- soil_type
write.csv(forest_cover_data_new, file = "Forest_Cover_Type_updated.csv")  # Save the changes for later use if needed to avoid running above code again
## Done making changes in the Forest Cover Type Data ##

# We observe the ratio of the categories in our dependent variable in the dataset first.
prop.table(table(forest_cover_data_new$CoverType)) 

# Create a copy of train and test set of data for Random Forest evaluation with respect to original data first.
set.seed(2)
rf_forest_cover_data_sampled <- forest_cover_data_new[sample(nrow(forest_cover_data_new), 10000), ]
rf_forest_cover_data_sampled$CoverType <- as.factor(rf_forest_cover_data_sampled$CoverType)
# Check for the ratio of the categories in the sample
prop.table(table(rf_forest_cover_data_sampled$CoverType)) # The ratios for all categories is almost same, Thus our modeling process will not be biased to any category.


rf_train_cover_type <- create_train_test(rf_forest_cover_data_sampled, 0.75, train = TRUE)
rf_test_cover_type <- create_train_test(rf_forest_cover_data_sampled, 0.75, train = FALSE)

# Build Model based on Original Dataset First and then with the new reduced dimensions to avoid curse of dimensionality.
# We check if reducing variables actually help in improving accuracy and model performance overall.
set.seed(100) # Run this to get same results.
rf_og <-randomForest(CoverType~.,data=train_forest_cover_data, ntree=500) 
print(rf_og)

rf_new <- randomForest(CoverType~., data = rf_train_cover_type, ntree = 500)
print(rf_new) # We can see that error rate reduced by 2.8%. Thus, we use this data for further model building.

# Predict using test set created above
# First, Model fitted on original higher dimension data
rf_pred_og = predict(rf_og, test_forest_cover_data[,-55], type = "class")
confusionMatrix(as.factor(rf_pred_og), test_forest_cover_data$CoverType)

# Second, Model fitted on new reduced dimension data
rf_pred_new = predict(rf_new, rf_test_cover_type[,-11], type = "class")
confusionMatrix(as.factor(rf_pred_new), rf_test_cover_type$CoverType)

# We can see that along with 2.52% increase in accuracy, Kappa measure is increased by 0.0437. Thus, reducing dimensions and combining categorical data worked only a little better in RF
# Thus, anyway we now use this data for further best model building.

# Now, to  select a reasonably best m(no. of predictor variables considered) and no. of trees we run rf for all variables in increasing order and including trees from 1 to 300 in the models.
oob_err_train_df <- data.frame(matrix(ncol = 300, nrow = 11))   # Create a DF to store Overall OOB rate of each model from all our model building.
oob_err_test_df <- data.frame(matrix(ncol = 300, nrow = 11))    # Do same for test set

# There's no need for test set in RF, as in the bagging process 37-38% of trees are left on which OOB error is calculated but we do testing on test set anyway.

start_time <- Sys.time()  # Start counter to keep check on progress and time estimation
for (i in 2:12){    # No. Predictors to be considered in each decision tree in forest
  oob_train <- vector()
  oob_test <- vector()
  for (n in 1:300){   # No. of trees to be considered in the forest
    set.seed(100)
    rf <-randomForest(CoverType~.,data=rf_train_cover_type, ntree=n, mtry = i)
    pred_test <- predict(rf, rf_test_cover_type[, -11], type = "class")
    
    oob_test[n] <- mean(pred_test != rf_test_cover_type$CoverType)
    
    sum_misclass = 0
    total_cases = 0
    
    # Calculate Overall OOB Error Rate
    for (j in 1:7){
      for (k in 1:7){
        total_cases = total_cases + rf$confusion[j,k]
        if (j  != k){
          sum_misclass = sum_misclass + rf$confusion[j,k]
        }
      }
    }
    
    oob_train[n] <- sum_misclass/total_cases
  }
  oob_err_test_df[i-1, ] <- oob_test
  oob_err_train_df[i-1, ] <- oob_train
  print(paste0("For no. of trees ",i," done!"))
}
paste("Time taken for 11*300 Random Forest Models:")  # It took a total of 1hr 30mins approximately for a mac machine with intel 2.3Ghz 8-core i9, 16GB RAM. 4GB Graphics RAM.
Sys.time() - start_time

new_test_df <- as.data.frame(t(oob_err_test_df))  # We transpose the matrix as we stored oob rates such that no. of trees are columns and no. of predictors are rows.
colnames(new_test_df) <-c("2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12")   # No. of predictors to be selected in each iteration of model building
new_train_df <- as.data.frame(t(oob_err_train_df))
colnames(new_train_df) <-c("2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12")
new_test_df["No. of Trees"] <- 1:300    # Add a no. of Trees column for further visualization
new_train_df["No. of Trees"] <- 1:300
# We save all the 11*300 model's error values for future reference if needed
write.csv(new_train_df, "OOB_Error_variable_selection_300trees_train_final.csv")
write.csv(new_test_df, "OOB_Error_variable_selection_300trees_test_final.csv")

# For visualizing the above models data, we pivot table so that it becomes easy for grouping variables in plotting
library("tidyverse")
df_plot_train <- new_train_df %>%
  select(`2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `No. of Trees`) %>%
  gather(key = "variable", value = "value", -`No. of Trees`)

df_plot_test <- new_test_df %>%
  select(`2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `No. of Trees`) %>%
  gather(key = "variable", value = "value", -`No. of Trees`)

# Color Palette for our 11 variable selection
col_pal_11 <- c("#800000", "#e6194B", "#bfef45", "#42d4f4", "#ffe119", "#fabed4", "#000075", "#f032e6", "#911eb4", "#aaffc3", "#808000")
# col_pal_11_extra <- c("#88ab75", "#dbd56e", "#a63a50", "#dad2bc", "#e6aace", "#0d1821", "#db2955", "#4ea3d1", "#f9f977", "#2ec4b6", "#e7f9a9")

ggplot(df_plot_train, aes(x = `No. of Trees`, y = value)) + 
  geom_line(aes(color = variable)) +# geom_point()# +
  scale_color_manual(values = col_pal_11) +
  labs(title = "OOB Error rate for Train set with different no. of variable selection", x = "No. of trees considered", y = "OOB Error Rate", color = "No. of Predictors")# +
gghighlight(value==min(value), label_key = variable)


ggplot(df_plot_test, aes(x = `No. of Trees`, y = value)) + 
  geom_line(aes(color = variable)) +
  scale_color_manual(values = col_pal_11) +
  labs(title = "OOB Error rate for Test set with different no. of variable selection", x = "No. of trees considered", y = "OOB Error Rate", color = "No. of Predictors")

# From the above Visualization, we consider the train plot for parameter selection. 
# We can see that including 5 no. of variables is enough for getting reasonably best result for our classification purpose. 
# Also, after including around 100-120 trees, we start getting almost around same error rate with around 2%-3% fluctuations.

# To have best model from what we calculated from all the models, we use the minimum OOB rate produced.
best_m <- as.numeric(df_plot_train[df_plot_train$value==min(df_plot_train$value), "variable"])
best_no_of_trees <- df_plot_train[df_plot_train$value==min(df_plot_train$value), "No. of Trees"]

df_plot_train[df_plot_train$value==min(df_plot_train$value), "value"] 

# Build a reasonably best RF model on this dataset from our work above
set.seed(100)
rf_best_model <- randomForest(CoverType~., data = rf_train_cover_type, ntree = best_no_of_trees, mtry = best_m)
print(rf_best_model)

# Predict the test data with above built model
predict_rf_best <- predict(rf_best_model, rf_test_cover_type[,-11], type="class")
confusionMatrix(as.factor(predict_rf_best), rf_test_cover_type$CoverType)

# Plot ROC curves for each category based on ONE vs ALL approach and calculating mean AUC for this multiclass
predict_rf_best <- predict(rf_best_model, rf_test_cover_type[,-11], type="prob")

auc_of_best_model_rf <- vector()
# For each class
for (i in 1:7){
  # Define which observations belong to class[i]
  true_values <- ifelse(rf_test_cover_type[,11]==classes[i],1,0)
  # Assess the performance of classifier for class[i]
  pred <- prediction(predict_rf_best[,i],true_values)
  perf <- performance(pred, "tpr", "fpr")
  
  if (i==1)
  {
    plot(perf,main="ROC Curves(One vs All) for Random Forest",col=pretty_colours[i], lwd=3)
    legend("bottomright", legend = c(1,2,3,4,5,6,7), fill=pretty_colours, title = "Forest Cover Type")
  }
  else
  {
    plot(perf,main="ROC Curve",col=pretty_colours[i],add=TRUE, lwd=3)
  }
  # # Calculate the AUC and print it to screen
  auc.perf <- performance(pred, measure = "auc")
  auc_of_best_model_rf[i] <- auc.perf@y.values[[1]]
}
print(paste0("Mean AUC for Final RF model is: ", mean(auc_of_best_model_rf)))   # AUC = 0.9581

# We now check for which variable is actually has more importance which is significant for starting the first decision of split.
imp <- varImpPlot(rf_best_model) # Save the varImp object for ggplot visualization

# this part just creates the data.frame for the plot part
library(dplyr)
imp_rf <- as.data.frame(imp)
imp_rf$varnames <- rownames(imp) # row names to column  

ggplot(imp_rf, aes(x=reorder(varnames, MeanDecreaseGini), y=MeanDecreaseGini, color="#2f3e46")) + 
  geom_point() +
  geom_segment(aes(x=varnames,xend=varnames,y=0,yend=MeanDecreaseGini)) +
  # scale_color_discrete(name="Variable Group") +
  labs(title = "Variable Importance Graph RF") + 
  ylab("Mean Decrease Gini") +
  xlab("Variable Name") +
  coord_flip()


#### RANDOM FOREST ####



###### -------------------------------------------- END COVER TYPE DATA -------------------------------------------- ######

