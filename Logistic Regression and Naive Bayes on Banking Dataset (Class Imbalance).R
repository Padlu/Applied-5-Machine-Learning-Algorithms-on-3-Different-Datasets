# To print all the values and override max.print warning
options(max.print = 1000)

# Set working directory
setwd("/Users/abhishekpadalkar/Documents/IRELAND Admissions/NCI/Course Modules/Modules/Sem 1/DMML/Final Project/")

###### -------------------------------------------- BANK DATA -------------------------------------------- ######

# Data Details:
# Input variables:
# # bank client data:
# 1 -- age (numeric)
# 2 -- job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
# 3 -- marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
# 4 -- education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
# 5 - default: has credit in default? (categorical: "no","yes","unknown")
# 6 -- housing: has housing loan? (categorical: "no","yes","unknown")
# 7 -- loan: has personal loan? (categorical: "no","yes","unknown")
# # related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: "cellular","telephone") 
# 9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 10 -- day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
# 11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# # other attributes:
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 13 -- pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 14 -- previous: number of contacts performed before this campaign and for this client (numeric)
# 15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
# # social and economic context attributes
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)     
# 18 -- cons.conf.idx: consumer confidence index - monthly indicator (numeric) # (CCI) is defined as the degree of optimism about the state of the economy that consumers (like you and me) are expressing through their activities of saving and spending    
# 19 - euribor3m: euribor(Euro Interbank Offer Rate) 3 month rate - daily indicator (numeric) # It is a reference rate that is constructed from the average interest rate at which eurozone banks offer unsecured short-term lending on the inter-bank market.
# 20 -- nr.employed: number of employees - quarterly indicator (numeric)
# 
# Output variable (desired target):
#   21 - y - has the client subscribed a term deposit? (binary: "yes","no")


# Get the data.
bank_data <- read.csv("Bank Subscription/bank_full.csv", header = T, sep = ";")

# Check the Structure
str(bank_data)  # All variables are fine!

# Check missing values
sapply(bank_data, function(x) sum(is.na(x))) # => No missing Values

# Checking Duplicates in datasets!
library(dplyr)
bank_data <- distinct(bank_data) # Removed 12 duplicates.

# Exploratory Data Analytics | Data Visualization
library(ggplot2)

# EDA
# As this dataset has a binary categorical dependent variable, we first look at how is the distribution of the y values.
count_df <- bank_data %>% 
  group_by(y) %>%
  summarise(no_rows = length(y))
ggplot(count_df, aes(x=y, y=no_rows, colour = y)) + geom_col(width=0.25) +
  labs(title="Count of Subcriptions", x="Subcribed as", y="Count")

# An imbalance in the predictor variable is observed. Thus, we keep this in mind for further modeling process

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

##### LOGISTIC REGRESSION #####
# Categorizing Age into 4 age brackets for better prediction
age <- bank_data$age
unique(sort(age)) # => We see that min and max age range from 17 - 98 :: We then put age category from 15 - 30, 31 - 50, 51 - 70, 71 - 100
age_brackets <- c('[15 - 30]', '[31 - 50]', '[51 - 70]', '[71 - 100]')

for (i in 1:length(age)){
  if (age[i] <31){
    age[i] <- age_brackets[1]
  }
  if (age[i] >30 & age[i] < 51){
    age[i] <- age_brackets[2]
  }
  if (age[i] > 50 & age[i] < 71){
    age[i] <- age_brackets[3]
  }
  if (age[i] > 70){
    age[i] <- age_brackets[4]
  }
}

# Replacing with the age brackets in the original dataset
bank_data$age <- as.factor(age)
# str(bank_data$age)    # Check for the changes made

# Changing the pdays 999(Client was not contacted previously of the campaign) and creating a variable to use that information. 999 is then replaced with -1 closer value to 0 as this variable is continuous in nature.
unique(bank_data$pdays)
client_contacted <- bank_data$pdays
for (i in 1:length(client_contacted)){
  if (client_contacted[i] == 999){
    client_contacted[i] <- "no"
    bank_data$pdays[i] <- -1
  }
  else{
    client_contacted[i] <- "yes"
  }
}

# Add this new column as a factor
bank_data['client_contacted_previously'] <- as.factor(client_contacted)
# str(bank_data)    # Check the changes made

# Reorder columns
bank_data <- bank_data[c(1,2,3,4,5,6,7,8,9,10,12,13,22,14,15,16,17,18,19,20,21)] # Remove 11 { duration } because it is said in the data description not to use that variable as it confirms the y response variable to give close to 100% accuracy.
str(bank_data)  # NOTE : 1 is no in our y. Thus, the model will check probabilities for no.

prop.table(table(bank_data$y))    # Shows that our dependent response variable is highly imbalanced in nature, with 88.73% of "no" and only 11.26% of "yes"

# Thus to finalize a good model, we check  for models for both imbalanced and balanced dataset.

#### Imbalanced Dataset ####

# Train and test split of the imbalanced data
bank_imbal_train <- create_train_test(bank_data, 0.75, train = TRUE)
bank_imbal_test <- create_train_test(bank_data, 0.75, train = FALSE)

# Check if the ratio of nos and yeses are same in the train and test set
prop.table(table(bank_imbal_train$y))
prop.table(table(bank_imbal_test$y))

# Perform first Logistic model considering all the variables and Check variable significance using Wald statistic
set.seed(100) # This is important to achieve the same results
library(car)
logit_imbal_bank <- glm(y~., family = "binomial", data = bank_imbal_train)
summary(logit_imbal_bank)   # AIC = 17052, Residual Deviance 16942 and Null Deviance 21656
Anova(logit_imbal_bank, test.statistic = "Wald")  # Job, Marital, Education, Loan, Housing, Pdays, Previous and Euribor3 are insignificant.

# Step by step removal of insignificant variables with highest p-value first and check model summary until all the variables are significant
logit_imbal_bank_1 <- update(logit_imbal_bank,y~.-loan)
summary(logit_imbal_bank_1)
Anova(logit_imbal_bank_1, test.statistic = "Wald")  # Job, Marital, Education, Housing, Pdays, Previous and Euribor3 are insignificant.

logit_imbal_bank_2 <- update(logit_imbal_bank_1,y~.-euribor3m)
summary(logit_imbal_bank_2)   
Anova(logit_imbal_bank_2, test.statistic = "Wald")  # Job, Marital, Education, Housing, Pdays and Previous are insignificant.

logit_imbal_bank_3 <- update(logit_imbal_bank_2,y~.-previous)
summary(logit_imbal_bank_3)
Anova(logit_imbal_bank_3, test.statistic = "Wald")  # Job, Marital, Education, Housing and Pdays are insignificant.

logit_imbal_bank_4 <- update(logit_imbal_bank_3,y~.-marital)
summary(logit_imbal_bank_4)
Anova(logit_imbal_bank_4, test.statistic = "Wald")  # Job, Education, Housing and Pdays are insignificant.

logit_imbal_bank_5 <- update(logit_imbal_bank_4,y~.-pdays)
summary(logit_imbal_bank_5)
Anova(logit_imbal_bank_5, test.statistic = "Wald")  # Job, Education, and Housing are insignificant.

logit_imbal_bank_6 <- update(logit_imbal_bank_5,y~.-job)
summary(logit_imbal_bank_6)
Anova(logit_imbal_bank_6, test.statistic = "Wald")  # Housing is insignificant.

logit_imbal_bank_7 <- update(logit_imbal_bank_6,y~.-housing)
summary(logit_imbal_bank_7) # AIC measure is brought down to 17036 from 17052 Residual deviance is just increased from 16942 to 16966 compared to null deviance of 21656.
Anova(logit_imbal_bank_7, test.statistic = "Wald")  # All variables are significant.

# We check if the AIC increases if we remove any variable further.
logit_imbal_bank_8 <- update(logit_imbal_bank_7,y~.-education)
summary(logit_imbal_bank_8)   # We can see that by removing a significant variable which was insignificant before, increases the AIC by 1 and residual error by 15, thus the last model is the best we can get out of the provided variables in the data for binary classification.
Anova(logit_imbal_bank_8, test.statistic = "Wald")

# Final Imbalanced data logistic regression model
logit_imbal_bank_final <- logit_imbal_bank_7
# Null Accuracy of the model
null_logit_p <- list(rep("no",length(bank_imbal_test$y)))
crosstab_1 <- table(prediction = null_logit_p[[1]], actual = bank_imbal_test$y)
crosstab_1   # Null Accuracy => 88.50%
# Prediction on test data with imbalanced data model
final_imbal_model_p <- predict(logit_imbal_bank_final, newdata = bank_imbal_test[, -c(3,6,7,12,14,19)], type = "response")
final_imbal_model_p <- ifelse(final_imbal_model_p > 0.5, "yes", "no")
confusionMatrix(as.factor(final_imbal_model_p), bank_imbal_test$y, positive = "yes")  # Model with removed variables # Accuracy = 89.74%, Kappa = 0.302, sensi = 0.2353, speci = 0.9832, detection rate = 0.027

# Test this model on entire bank data
final_imbal_model_p_1 <- predict(logit_imbal_bank_final, newdata = bank_data[, -c(3,6,7,12,14,19)], type = "response")
final_imbal_model_p_1 <- ifelse(final_imbal_model_p_1 > 0.5, "yes", "no")
confusionMatrix(as.factor(final_imbal_model_p_1), bank_data$y, positive = "yes")  # Model with removed variables # Accuracy = 90%, Kappa = 0.3031, sensi = 0.23238, speci = 0.98489, detection rate = 0.02618


#### End Imbalanced Data ####

#### Balanced Data ####

# Check no. of yeses in the response variable and select random rows of no's of the length of yeses and merge it to get a balanced data
library(prettyR)
describe.factor(bank_data$y)  # 4639 are No of yeses in bank_data$y 

total_no_rows_needed <- nrow(bank_data[bank_data$y == "yes",]) #4639 # No of yeses in bank_data$y
set.seed(21)
balance_idx <- row.names(bank_data[bank_data$y == "no",]) # get indexes of all no rows
sampled_balance_idx_no <- sample(balance_idx, total_no_rows_needed) # get indexes of all no rows randomly sampled with # of rows = to that of yeses
total_balance_idx_yes <- row.names(bank_data[bank_data$y == "yes",])  # get indexes of all yes rows
balance_idx <- c(sampled_balance_idx_no, total_balance_idx_yes) # Concatenate both nos and yes
balance_idx_shuffle <- balance_idx[sample(1:length(balance_idx))] # Shuffle randomly for good split
balanced_bank_data <- bank_data[balance_idx_shuffle, ]  # Get those rows from the main dataset
table(balanced_bank_data$y) # Check the balance => Correct! # Total observations 9278

# Train and test split of the balanced data
bal_bank_train <- create_train_test(balanced_bank_data, 0.75, train = TRUE)
bal_bank_test <- create_train_test(balanced_bank_data, 0.75, train = FALSE)

# Check if the ratio of split is equal
prop.table(table(bal_bank_train$y))
prop.table(table(bal_bank_test$y))

set.seed(100)
logit_bal_bank <- glm(y~., family = binomial, data = bal_bank_train)
summary(logit_bal_bank) # AIC = 7534.5, Null deviance is 9645.7 and Residual deviance is 7424.5
Anova(logit_bal_bank, test.statistic = "Wald") # Age, Job, Marital, Education, Housing, Loan, Pdays, Client Contacted Previously, Previous, Poutcome, Cons.Conf.Idx, Nr.Employed are insignificant variables.

# Step by step removal of insignificant variables with highest p-value first and check model summary until all the variables are significant
logit_bal_bank_1 <- update(logit_bal_bank, y~.-cons.conf.idx)
summary(logit_bal_bank_1)
Anova(logit_bal_bank_1, test.statistic = "Wald")  # Age, Job, Education, Housing, Loan, Day_of_week, Pdays, Previous, Poutcome, Cons.Conf.Idx, Nr.Employed are insignificant variables.

logit_bal_bank_2 <- update(logit_bal_bank_1, y~.-pdays)
summary(logit_bal_bank_2)  
Anova(logit_bal_bank_2, test.statistic = "Wald")  # Age, Education, Housing, Loan, Day_of_week, Pdays, Previous, Poutcome, Cons.Conf.Idx, Nr.Employed are insignificant variables.

logit_bal_bank_3 <- update(logit_bal_bank_2, y~.-loan)
summary(logit_bal_bank_3)
Anova(logit_bal_bank_3, test.statistic = "Wald") # Age, Housing, Loan, Day_of_week, Pdays, Previous, Poutcome, Cons.Conf.Idx, Nr.Employed are insignificant variables.

logit_bal_bank_4 <- update(logit_bal_bank_3, y~.-housing)
summary(logit_bal_bank_4)
Anova(logit_bal_bank_4, test.statistic = "Wald")  # Age, Loan, Day_of_week, Pdays, Previous, Poutcome, Cons.Conf.Idx, Nr.Employed are insignificant variables.

logit_bal_bank_5 <- update(logit_bal_bank_4, y~.-education)
summary(logit_bal_bank_5)
Anova(logit_bal_bank_5, test.statistic = "Wald")  # Age, Day_of_week, Pdays, Previous, Poutcome, Cons.Conf.Idx, Nr.Employed are insignificant variables.

logit_bal_bank_6 <- update(logit_bal_bank_5, y~.-nr.employed)
summary(logit_bal_bank_6)
Anova(logit_bal_bank_6, test.statistic = "Wald")  # Age, Day_of_week, Previous, Poutcome, Cons.Conf.Idx, Nr.Employed are insignificant variables.

logit_bal_bank_7 <- update(logit_bal_bank_6, y~.-previous)
summary(logit_bal_bank_7)
Anova(logit_bal_bank_7, test.statistic = "Wald") # Age, Day_of_week, Previous, Poutcome, Cons.Conf.Idx are insignificant variables.

logit_bal_bank_8 <- update(logit_bal_bank_7, y~.-marital)
summary(logit_bal_bank_8)
Anova(logit_bal_bank_8, test.statistic = "Wald")  # Age, Day_of_week, Previous, Poutcome are insignificant variables.

logit_bal_bank_9 <- update(logit_bal_bank_8, y~.-age)
summary(logit_bal_bank_9)   # AIC measure is brought down to 7507.4 from 7534.5 Residual deviance is just increased from 7424.5 to 7437.4 compared to null deviance of 9645.7
Anova(logit_bal_bank_9, test.statistic = "Wald")  # All variables are significant!

# Try removing age which was insignificant before just to check AIC and residual deviance increase or not
logit_bal_bank_10 <- update(logit_bal_bank_9, y~.-default) 
summary(logit_bal_bank_10)  # AIC = 7513 a jump of 5.6, Residual deviance = 7447.0 a jump of 9.6
Anova(logit_bal_bank_10, test.statistic = "Wald") # Age is insignificant variables.

# Finalize the balanced data logistic model
logit_bal_bank_final <- logit_bal_bank_9
# Null accuracy of the model is 50% as the data is balanced
# Prediction on the test data using the final balanced model
bal_final_model_p <- predict(logit_bal_bank_final, newdata = bal_bank_test[, -c(1, 3, 4, 6, 7, 12, 14, 18, 20, 21)], type = "response")
bal_final_model_p <- ifelse(bal_final_model_p > 0.5, "yes", "no")
confusionMatrix(as.factor(bal_final_model_p), bal_bank_test$y, positive = "yes")  # Accuracy = 0.728, Kappa = 0.4573, Sensi = 0.6232, Speci = 0.8352, Detection Rate = 0.3151

# Test this model on entire bank data
bal_final_model_p_1 <- predict(logit_bal_bank_final, newdata = bank_data[, -c(1, 3, 4, 6, 7, 12, 14, 18, 20, 21)], type = "response")
bal_final_model_p_1 <- ifelse(bal_final_model_p_1 > 0.5, "yes", "no")
confusionMatrix(as.factor(bal_final_model_p_1), bank_data$y, positive = "yes")  # Accuracy = 0.819, Kappa = 0.3472, Sensi = 0.63979, Speci = 0.84175, Detection Rate = 0.07208

# By comparing the evaluation metrics of balanced and imbalanced data, we finalize the balanced data model as our final model for classification using logistic regression

#### END Balanced Data ####

# Analyzing final LR model
# Psuedo Rˆ2 for knowing the relationship between the response variable and explanatory variables
library(pscl)
pR2(logit_bal_bank_final) # rˆ2ML(Maximum likelihood pseudo r-squared) => 0.2719513 | rˆ2CU(Cragg and Uhler's pseudo r-squared) => 0.3626034 | McFadden(McFadden's pseudo r-squared) => 0.2289489


# ROC Plot and AUC measure
library(ROCR)
library(Amelia)
bal_final_pr <- prediction(bal_final_model_p, bal_bank_test$y)
bal_final_prf <- performance(bal_final_pr, measure = "tpr", x.measure = "fpr")

bank_roc_colors <- c("#f9c74f", "#8ecae6")
plot(bal_final_prf, main="ROC Curve for best Logistic Regression and Naïve Bayes Model", col=bank_roc_colors[2], lwd=3)  # After this plot, plot the below NB ROC plot to get both the plots together.

lr_auc <- performance(bal_final_pr, measure = "auc")
lr_auc <- lr_auc@y.values[[1]]
lr_auc # 0.7758027

# Plotting Variable importance graph to know which variables are contributing more to response variable
varimp_lr <- varImp(logit_bal_bank_final)
ggplot(varimp_lr, aes(row.names(varimp_lr), Overall, fill = row.names(varimp_lr))) + geom_col() + coord_flip() # Variable Importance of the final model with removed insignificant variables


##### END LOGISTIC REGRESSION #####

##### NAIVE BAYES #####

# Get the data
bank_data <- read.csv("Bank Subscription/bank_full.csv", header = T, sep = ";")

# Categorizing Age into 4 age brackets for better prediction
age <- bank_data$age
unique(sort(age)) # => We see that min and max age range from 17 - 98 :: We then put age category from 15 - 30, 31 - 50, 51 - 70, 71 - 100
age_brackets <- c('[15 - 30]', '[31 - 50]', '[51 - 70]', '[71 - 100]')

for (i in 1:length(age)){
  if (age[i] <31){
    age[i] <- age_brackets[1]
  }
  if (age[i] >30 & age[i] < 51){
    age[i] <- age_brackets[2]
  }
  if (age[i] > 50 & age[i] < 71){
    age[i] <- age_brackets[3]
  }
  if (age[i] > 70){
    age[i] <- age_brackets[4]
  }
}

# Replacing with the age brackets in the original dataset
bank_data$age <- as.factor(age)
# str(bank_data$age)    # Check for the changes made
# Remove Duration as it will predict more accurately. It is a variable that is clear indication of yes or no.
bank_data <- bank_data[c(1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21)]

# Building a model #
# split data into training and test data sets #

# We use the same edited bank data for naive bayes
nb_bank_train <- create_train_test(bank_data, 0.75, train = TRUE)
nb_bank_test <- create_train_test(bank_data, 0.75, train = FALSE) 

# Check if the ratio of nos and yeses are same in the train and test set
prop.table(table(nb_bank_train$y))
prop.table(table(nb_bank_test$y))

library(e1071)
set.seed(200)
nb_model <- naiveBayes(y~., data = nb_bank_train)
print(nb_model)

nb_model_cv <- train(nb_bank_train[, -20], nb_bank_train$y, 'nb', trControl=trainControl(method='cv',number=10))
print(nb_model_cv)

nb_model_bootstrap <- train(nb_bank_train[, -20], nb_bank_train$y, 'nb')   # Used bootstrapping with 25 reps
print(nb_model_bootstrap)

nb_model_factor <- train(nb_bank_train[, -c(11, 12, 13, 14, 16, 17, 18, 19, 20)], nb_bank_train$y, 'nb', trControl=trainControl(method='cv',number=10))
print(nb_model_factor)


# Model Evaluation #

# Predict testing set #
bank_nb_p <- predict(nb_model, newdata = nb_bank_test[, -20])  
confusionMatrix(bank_nb_p, nb_bank_test$y, positive = "yes")   # Accuracy = 0.836, Kappa = 0.3234, Sensi = 0.51596, Speci = 0.87656, Detection Rate = 0.05808

bank_nb_cv_p <- predict(nb_model_cv, newdata = nb_bank_test[, -20])  
confusionMatrix(bank_nb_cv_p, nb_bank_test$y, positive = "yes")   # Accuracy = 0.8758, Kappa = 0.368, Sensi = 0.42968, Speci = 0.93237, Detection Rate = 0.04836

bank_nb_bs_p <- predict(nb_model_bootstrap, newdata = nb_bank_test[, -20])  
confusionMatrix(bank_nb_bs_p, nb_bank_test$y, positive = "yes")   # Accuracy = 0.8758, Kappa = 0.368, Sensi = 0.42968, Speci = 0.93237, Detection Rate = 0.04836

bank_nb_factor_p <- predict(nb_model_factor, newdata = nb_bank_test[, -c(11, 12, 13, 14, 16, 17, 18, 19, 20)])
confusionMatrix(bank_nb_factor_p, nb_bank_test$y, positive = "yes")   # Accuracy = 0.8783, Kappa = 0.2767, Sensi = 0.27869, Speci = 0.95437, Detection Rate = 0.03137

# From above we select optimum model as normal naive model without any resampling technique.

# Plot ROC and check AUC
bank_nb_p_num <- vector()   # To do that we first convert our factor predictions into numeric list as the function accepts numeric values only
for (i in 1:length(bank_nb_p)){
  if (bank_nb_p[i] == "no"){
    bank_nb_p_num[i] = 0
  }
  if (bank_nb_p[i] == "yes"){
    bank_nb_p_num[i] = 1
  }
}

bank_nb_pr <- prediction(bank_nb_p_num, nb_bank_test$y)
bank_nb_prf <- performance(bank_nb_pr, measure = "tpr", x.measure = "fpr")

plot(bank_nb_prf, main="ROC Curve for best Logistic Regression and Naïve Bayes Model", add=TRUE, col=bank_roc_colors[1], lwd=3)  
legend("bottomright", legend = c("Naïve Bayes  |  AUC: 0.6962", "Logistic Regression  |  AUC: 0.7758"), fill=bank_roc_colors, title = "Model")

auc_nb <- performance(bank_nb_pr, measure = "auc")
auc_nb <- auc_nb@y.values[[1]]
auc_nb # 0.6962607


# Understand which categorical features can result in a "yes" or a "no"
roww_names <- c("Job", "Marital", "Education", "Default", "Housing", "Loan", "Contact", "Age")
yes <- vector()
yes_name <- vector()
no <- vector()
no_name <- vector()

yes[1] <- max(nb_model$tables$job[2 ,])
yes_name[1] <- names(nb_model$tables$job[2 ,])[max.col(nb_model$tables$job[2 ,])][1]
no[1] <- max(nb_model$tables$job[1 ,])
no_name[1] <- names(nb_model$tables$job[1 ,])[max.col(nb_model$tables$job[1 ,])][1]

yes[2] <- max(nb_model$tables$marital[2 ,])
yes_name[2] <- names(nb_model$tables$marital[2 ,])[max.col(nb_model$tables$marital[2 ,])][1]
no[2] <- max(nb_model$tables$marital[1 ,])
no_name[2] <- names(nb_model$tables$marital[1 ,])[max.col(nb_model$tables$marital[1 ,])][1]

yes[3] <- max(nb_model$tables$education[2 ,])
yes_name[3] <- names(nb_model$tables$education[2 ,])[max.col(nb_model$tables$education[2 ,])][1]
no[3] <- max(nb_model$tables$education[1 ,])
no_name[3] <- names(nb_model$tables$education[1 ,])[max.col(nb_model$tables$education[1 ,])][1]

yes[4] <- max(nb_model$tables$default[2 ,])
yes_name[4] <- names(nb_model$tables$default[2 ,])[max.col(nb_model$tables$default[2 ,])][1]
no[4] <- max(nb_model$tables$default[1 ,])
no_name[4] <- names(nb_model$tables$default[1 ,])[max.col(nb_model$tables$default[1 ,])][1]

yes[5] <- max(nb_model$tables$housing[2 ,])
yes_name[5] <- names(nb_model$tables$housing[2 ,])[max.col(nb_model$tables$housing[2 ,])][1]
no[5] <- max(nb_model$tables$housing[1 ,])
no_name[5] <- names(nb_model$tables$housing[1 ,])[max.col(nb_model$tables$housing[1 ,])][1]

yes[6] <- max(nb_model$tables$loan[2 ,])
yes_name[6] <- names(nb_model$tables$loan[2 ,])[max.col(nb_model$tables$loan[2 ,])][1]
no[6] <- max(nb_model$tables$loan[1 ,])
no_name[6] <- names(nb_model$tables$loan[1 ,])[max.col(nb_model$tables$loan[1 ,])][1]

yes[7] <- max(nb_model$tables$contact[2 ,])
yes_name[7] <- names(nb_model$tables$contact[2 ,])[max.col(nb_model$tables$contact[2 ,])][1]
no[7] <- max(nb_model$tables$contact[1 ,])
no_name[7] <- names(nb_model$tables$contact[1 ,])[max.col(nb_model$tables$contact[1 ,])][1]

yes[8] <- max(nb_model$tables$age[2 ,])
yes_name[8] <- names(nb_model$tables$age[2 ,])[max.col(nb_model$tables$age[2 ,])][1]
no[8] <- max(nb_model$tables$age[1 ,])
no_name[8] <- names(nb_model$tables$age[1 ,])[max.col(nb_model$tables$age[1 ,])][1]

highest_probability_of_yes_no <- data.frame(row.names = roww_names, yes, yes_name, no, no_name) # Both for "yes" and "no" we get same answer => These are not good predictors to predict subscription value.


##### END NAIVE BAYES #####

###### -------------------------------------------- END BANK DATA -------------------------------------------- ######

