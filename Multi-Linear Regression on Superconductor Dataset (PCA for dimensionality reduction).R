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


###### -------------------------------------------- SUPERCONDUCTOR DATA -------------------------------------------- ######

# Get the data
superconductor_data <- read.csv("Super Conductor Temperature/Super_conductor.csv", header = T)

# Check the Structure
str(superconductor_data) # There are a lot of variables for atomic mass, fie, atomic radius, Density, Electron Affinity, Fusion Heat, Thermal Conductivity, Valence. These can be multicorrelated. Thus, we will need to remove such dependent variables.

# Check missing values
sapply(superconductor_data, function(x) sum(is.na(x))) # => No missing Values

# Checking Duplicates in datasets!
library(dplyr)
superconductor_data <- distinct(superconductor_data)  # Removed 66 duplicates.

# EDA / Descriptive Statistics
library(corrplot)
corrplot(cor(superconductor_data), type = "upper") # We can see that there are high positive and negative corr

#### MULTIPLE LINEAR REGRESSION ####

# We can't perform linear regression on data which has variables highly correlated with each other, as the assumption of absence of multicollinearity is violated. 

# But, still we build a linear regression model first, to check the performance of it.
## Also, Check significance of variables in superConductor Data ##
superc_og_model <- lm(critical_temp~., data = superconductor_data)
summary(superc_og_model)  # Adjusted Rˆ2 = 0.7361 | Residual Standard Error = 17.61
# We note that out of 81 variables, only 10 variables show insignificant. This, clearly indicates the high correlation between all variables with each other.

# We thus first remove all the weighted variables of other variables as they are related.
library(tidyverse)
superconductor_data_new <- superconductor_data %>% select(-contains("wtd"))   # We remove wtd. variables for reducing complexity.
str(superconductor_data_new)

# We check again the fit on the above wtd removed data, just for own purpose of checking the performance of the model
superc_og_no_wts_model <- lm(critical_temp~., data = superconductor_data_new)
summary(superc_og_no_wts_model)   # Adjusted Rˆ2 = 0.6404 | Residual Standard Error = 20.55. {Error increased by 2.94}

# We again check for correlations between all variables
corrplot(cor(superconductor_data_new), type = "upper") # We can see that there are high positive and negative corr
# Still there is high multi-collinearity between many variables and it needs to be dealt with to get a generalized linear model.

# To deal with many and highly correlated variables we perform PCA and then perform Regression on it

library(psych)
library(leaps)
library(car)

## PCA ##
superconductor_data_new_pca <- superconductor_data_new[, -c(42)]  # Get only Variables
str(superconductor_data_new_pca)  # 21197 obs and 41 variables => Suitable for PCA

# Perform checks on data to confirm we can perform PCA on this data.
KMO(superconductor_data_new_pca)  # Overall KMO = 0.83
bartlett.test(superconductor_data_new_pca)  # p-value = 2.2e-16
# Thus, we move forward with PCA

# Check for number of components
fa.parallel(superconductor_data_new_pca, fa="pc",n.iter = 100, main = "Scree Plot for PCA and Eigen Value Threshold") # 8 PCs to be formed according to EV Rule

#Extract components with rotated axes for better interpretation of components
rc.superconductor_data<-principal(superconductor_data_new_pca,nfactors = 8,rotate="varimax")

rc.superconductor_data$loadings

## END PCA ##

rcs_predictors <- data.frame(rc.superconductor_data$scores) # Create new data with these components as our predictors
rcs_predictors["critical_temp"] <- superconductor_data_new[, 42]  # Add our dependent variable to this new data
rc_train <- create_train_test(rcs_predictors, 0.75, train = TRUE)   # Create train and test set for this new data
rc_test <- create_train_test(rcs_predictors, 0.75, train = FALSE)
# str(rcs_predictors)

# First, we check relationship between our variables by simply plotting them.
corrplot(cor(rc_train), type = "upper") # We can see that there are high positive and negative corr
pairs.panels(rc_train,
             panel = panel.smooth,
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE  # show density plots
)   # We see that there expect for RC1 and RC5, others follow non-linear trend with our dependent variable


# Build new model on the RCs  with full data, just to check
superc_rc_model <- lm(critical_temp~., data = rcs_predictors)
summary(superc_rc_model)   # Adjusted Rˆ2 = 0.5488 | Residual Standard Error = 23.02

# Build Model on train and Test it on test set
superc_rc_model_train <- lm(critical_temp~., data = rc_train)
summary(superc_rc_model_train)   # Adjusted Rˆ2 = 0.5487 | Residual Standard Error = 22.91
predict_rc_lm <- predict(superc_rc_model_train, newdata = rc_test)
rmse <- sqrt(sum((predict_rc_lm - rc_test$critical_temp)^2)/length(rc_test$critical_temp))    
c(RMSE = rmse, R2=summary(superc_rc_model_train)$adj.r.squared)   # Test Set RMSE = 23.37 | Adjusted Rˆ2 = 0.5487
par(mfrow=c(2,2))   # It plots 4 different plot in a 2,2 matrix format as a single plot. 
plot(superc_rc_model_train)   # In this model, we see that Residual plot has a U shaped curve. This means there may be non-linear relationship between our dependent variable and our RCs. We first deal with it by using Log of our dependent variable. We also see that scale location plot also has a pattern in residuals. In Normal plot, we see a good normal distribution between variables. Also, no observation has a significant cooks distance.
# Thus, we first try to deal with Residual plot to have random residual plot.
par(mfrow=c(1,1))

# We also check for residual plot vs individual variables below to get more understanding in it.
##Plot the residual plot with all predictors.
attach(rc_train)
require(gridExtra)
plot1 = ggplot(rc_train, aes(RC1, residuals(superc_rc_model_train))) + geom_point() + geom_smooth()
plot2 = ggplot(rc_train, aes(RC2, residuals(superc_rc_model_train))) + geom_point() + geom_smooth()
plot3 = ggplot(rc_train, aes(RC3, residuals(superc_rc_model_train))) + geom_point() + geom_smooth()
plot4 = ggplot(rc_train, aes(RC4, residuals(superc_rc_model_train))) + geom_point() + geom_smooth()
plot5 = ggplot(rc_train, aes(RC5, residuals(superc_rc_model_train))) + geom_point() + geom_smooth()
plot6 = ggplot(rc_train, aes(RC6, residuals(superc_rc_model_train))) + geom_point() + geom_smooth()
plot7 = ggplot(rc_train, aes(RC7, residuals(superc_rc_model_train))) + geom_point() + geom_smooth()
plot8 = ggplot(rc_train, aes(RC8, residuals(superc_rc_model_train))) + geom_point() + geom_smooth()
grid.arrange(plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,ncol=4,nrow=2)
# We see RC1, RC4, RC5, RC6, RC7, RC8 have a slight curve in it. 
# We add polynomials of these terms and then check for the best subset using these variables and then check the residual plot again.

superc_rc_assump_1 <- update(superc_rc_model_train, critical_temp~.+ I(RC1^2)+ I(RC4^2)+ I(RC5^2)+ I(RC6^2)+ I(RC7^2)+ I(RC8^2))
summary(superc_rc_assump_1)   # Adjusted Rˆ2 = 0.5596 | Residual Standard Error = 22.63
par(mfrow=c(2,2))
plot(superc_rc_assump_1)
par(mfrow=c(1,1))
predict_rc_lm_assump_1 <- predict(superc_rc_assump_1, newdata = rc_test)
rmse <- sqrt(sum((predict_rc_lm_assump_1 - rc_test$critical_temp)^2)/length(rc_test$critical_temp)) 
c(RMSE = rmse, R2=summary(superc_rc_assump_1)$adj.r.squared)   # Test Set RMSE = 23.09 | Adjusted Rˆ2 = 0.5596


superc_rc_assump_1_best_subset <- regsubsets(critical_temp~RC1 + RC2 + RC3 + RC4 + RC5 + RC6 + RC7 + RC8 +
                                               I(RC1^2)+ I(RC4^2)+ I(RC5^2)+ I(RC6^2)+ I(RC7^2)+ I(RC8^2), data = rc_train)
summary(superc_rc_assump_1_best_subset)   # We see that having RC1, RC3, RC4, RC5, RC6, RC7, RC8 and I(RC7^2) gives best combination of variables based on AIC and Rˆ2 which are significant.
subsets(superc_rc_assump_1_best_subset, statistic="adjr2")
# Thus, we now use these as our variables for our assumption one correction.

superc_rc_assump_1_best_subset_model <- update(superc_rc_assump_1, critical_temp~.+ I(RC7^2) - RC2 - I(RC1^2) - I(RC4^2) - I(RC5^2) - I(RC6^2) - I(RC8^2))
summary(superc_rc_assump_1_best_subset_model)  # Adjusted Rˆ2 = 0.5554 | Residual Standard Error = 22.73
# Now, we again plot and check for residual errors randomness
par(mfrow=c(2,2))
plot(superc_rc_assump_1_best_subset_model)  # This is the best we can do for now for random residual vs fitted... We can see that residuals still follows a normal distribution.
par(mfrow=c(1,1))
predict_rc_lm_assump_1_best_subset_model <- predict(superc_rc_assump_1_best_subset_model, newdata = rc_test)
rmse <- sqrt(sum((predict_rc_lm_assump_1_best_subset_model - rc_test$critical_temp)^2)/length(rc_test$critical_temp)) 
c(RMSE = rmse, R2=summary(superc_rc_assump_1_best_subset_model)$adj.r.squared)   # Test Set RMSE = 23.20 | Adjusted Rˆ2 = 0.5554
durbinWatsonTest(superc_rc_assump_1_best_subset_model)  # DW Statistic => 2.02 => Means no Autocorrelation. => GOOD

# Now, we look for heteroscadasticity
ncvTest(superc_rc_assump_1_best_subset_model)   # p-value < 0.05 => Means we also have heteroscadasticity existing in our model. We try to remove it by taking sqrt of our dependent variable.

superc_rc_assump_2 <- update(superc_rc_assump_1_best_subset_model, sqrt(critical_temp)~.)
summary(superc_rc_assump_2)   # Adjusted Rˆ2 = 0.6328 | Residual Standard Error = 1.834
par(mfrow=c(2,2))
plot(superc_rc_assump_2)
par(mfrow=c(1,1))
predict_rc_lm_assump_2 <- predict(superc_rc_assump_2, newdata = rc_test)
rmse <- sqrt(sum((predict_rc_lm_assump_2 - rc_test$critical_temp)^2)/length(rc_test$critical_temp)) 
c(RMSE = rmse, R2=summary(superc_rc_assump_2)$adj.r.squared)   # Test Set RMSE = 44.93 | Adjusted Rˆ2 = 0.6328

ncvTest(superc_rc_assump_2)   # We still get a significant NCV Test. It means there's still heteroscadasticity, but the plot doesn't suggest that in scale location. This might be because of the large sample size we are dealing with.
durbinWatsonTest(superc_rc_assump_2)    # DW Statistic => 2.01 => Means no Autocorrelation. => GOOD
vif(superc_rc_assump_2)   # All values around 1. Thus, no multicollinearity.

# We again check for residuals for our final model superc_rc_assump_2.2
attach(rc_train)
require(gridExtra)
plot1 = ggplot(rc_train, aes(RC1, residuals(superc_rc_assump_2.2))) + geom_point() + geom_smooth()
plot3 = ggplot(rc_train, aes(RC3, residuals(superc_rc_assump_2.2))) + geom_point() + geom_smooth()
plot4 = ggplot(rc_train, aes(RC4, residuals(superc_rc_assump_2.2))) + geom_point() + geom_smooth()
plot5 = ggplot(rc_train, aes(RC5, residuals(superc_rc_assump_2.2))) + geom_point() + geom_smooth()
plot6 = ggplot(rc_train, aes(RC6, residuals(superc_rc_assump_2.2))) + geom_point() + geom_smooth()
plot7 = ggplot(rc_train, aes(RC7, residuals(superc_rc_assump_2.2))) + geom_point() + geom_smooth()
plot8 = ggplot(rc_train, aes(RC8, residuals(superc_rc_assump_2.2))) + geom_point() + geom_smooth()
grid.arrange(plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,ncol=4,nrow=2)

# Variable Importance in our regression model
x <- varImp(superc_rc_assump_2)
ggplot(x, aes(reorder(row.names(x), -Overall), Overall, fill = row.names(x))) + geom_col() + coord_flip() + 
  geom_text(aes(label=sprintf("%0.2f", round(Overall, digits = 2))), vjust=0.45, hjust=-0.15, size=3.5) +
  labs(x="Variables", y="Variable Importance", fill="Variables")

# What are the actual underlying influential factors in these components
fa.diagram(rc.superconductor_data)  # RC1 seems to be an entropy statistic of all the variables and RC6 is a combination of density, atomic radius, valence, atomic mass, thermal conductvity and first ionization energy(fie) with more loading weightage on density, atomic radius, fie and valence.

# Find out Bias vs Variance for all our built models
train_rmse <- c(22.91, 22.63, 22.73, 1.834)
test_rmse <- c(23.37, 23.09, 23.20, 44.93)
adj_r2 <- c(0.5487, 0.5596, 0.5554, 0.6328)
model_name <- c("superc_rc_model_train", "superc_rc_assump_1", "superc_rc_assump_1_best_subset_model", "superc_rc_assump_2")
model_type <- c("multiple linear model", "multiple polynomial", "linear + single polynomial", "Sqrt_response + linear + single polynomial")
no_of_predictors <- c(8, 14, 8, 8)

bias_variance_df <- data.frame(train_rmse, test_rmse, adj_r2, model_name, model_type, no_of_predictors)
bias_variance_df$model_name <- factor(bias_variance_df$model_name, levels = c("superc_rc_model_train", "superc_rc_assump_1_best_subset_model", "superc_rc_assump_1", "superc_rc_assump_2"))

ggplot(bias_variance_df, aes(x=model_name, group = 1)) +
  geom_line(aes(y = train_rmse, color = "darkred")) + 
  geom_line(aes(y = test_rmse, color="steelblue")) + 
  labs(x="Model Names (Increasing Complexity)", y="RMSE") +
  scale_colour_manual(name = 'Model testing', 
                      values =c('darkred'='darkred','steelblue'='steelblue'), labels = c('Train RMSE','Test RMSE'))

ggplot(bias_variance_df, aes(model_name, adj_r2, group = 1)) + geom_line(linetype="twodash") +
  labs(x="Model Name", y="Adjusted Rˆ2")
layer(geom = "line")

#### END MULTIPLE LINEAR REGRESSION ####


###### -------------------------------------------- END SUPERCONDUCTOR DATA -------------------------------------------- ######
