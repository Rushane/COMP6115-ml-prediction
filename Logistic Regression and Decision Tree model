rm(list=ls())
options(scipen=99999)

#install.packages("ROSE")
# install.packages("plyr")
# install.packages("ggplot2")
#install.packages("randomForest") 
#install.packages("caTools")
#install.packages("tidyverse")
#install.packages("hrbrthemes")
#install.packages("corrplot")
#install.packages("RColorBrewer")
#install.packages("PRROC")
#install.packages("varImpPlot")
library(caTools)
library(plyr)
library(ggplot2)
library(ROSE)
library(randomForest)
library(pROC)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(hrbrthemes)
library(ranger)
library(caret)
library(data.table)
library(corrplot)
library(RColorBrewer)
library(dplyr) 
library(PRROC)
library(psych)


# Load data
creditcarddata <- read.csv(file.choose())

# Data Understanding 
head(creditcarddata) 
str(creditcarddata)
nrow(creditcarddata)
summary(creditcarddata)# summary of credit card data
View(creditcarddata$Class) # should this be converted to other type of column

apply(creditcarddata,2, function(p) sum(is.na(p))) # get na by column
apply(creditcarddata,2, function(p) sum(p == "")) # get number of missing values by column

hist(creditcarddata$Time) # use a distribution plot instead
## hist(creditcarddata$Amount)

## checking fraud vs non fraud transactions
# 1 for fraudulent transactions, 0 non-fraudulent
summary(creditcarddata$Class==1) ## There are 492 fraudulent transactions compared to 284315
# Given that the fraud transactions are much lesser in number than non-fraudulent transactions
# this causes an imbalance in the data.

ggplot(creditcarddata, aes(x=Class)) + geom_histogram(binwidth=.5) ## shows imbalance in the data

#checking for correlation with class feature
creditcarddata_Cor=cor(creditcarddata)
corrplot(creditcarddata_Cor) #from plot the strongest correlation is between Class and Amount

creditcardFraud <- creditcarddata[creditcarddata$Class==1,]

# shows that majority of fraudulent transactions are small in amount
ggplot(creditcardFraud, aes(x=Amount)) + geom_histogram(binwidth=30)

hist(creditcarddata$V1)
hist(creditcarddata$V2)
hist(creditcarddata$V3)
hist(creditcarddata$V4)
hist(creditcarddata$V5)
hist(creditcarddata$V6)
hist(creditcarddata$V7)
hist(creditcarddata$V8)
hist(creditcarddata$V9)
hist(creditcarddata$V10)
hist(creditcarddata$V11)
hist(creditcarddata$V12)
hist(creditcarddata$V13)
hist(creditcarddata$V14)
hist(creditcarddata$V15)
hist(creditcarddata$V16)
hist(creditcarddata$V17)
hist(creditcarddata$V18)
hist(creditcarddata$V19)
hist(creditcarddata$V20)
hist(creditcarddata$V21)
hist(creditcarddata$V22)
hist(creditcarddata$V23)
hist(creditcarddata$V24)
hist(creditcarddata$V25)
hist(creditcarddata$V26)
hist(creditcarddata$V27)
hist(creditcarddata$V28)

#Data Preparation
# possibly standardize time and amount columns?

#Normalizing Amount column 
#normalization function 
normalize <- function(x){
  return ((x - mean(x, na.rm =TRUE))/sd(x, na.rm =TRUE))}

creditcarddata$Amount <- normalize(creditcarddata$Amount)

################################ Both Undersampling ##############################
## used undersampling technique to solve imbalance problem (TBD)
## changed n observations to 2100 
## because the amount of fraudulent transactions is equal to 492 
data_balanced_under <- ovun.sample(Class ~ ., data = creditcarddata, method = "under", N = 1200, seed = 1)$data
summary(data_balanced_under$Class==1)

# Structure of Class 
str(data_balanced_under$Class)

# converting class feature(target variable) to factor
data_balanced_under$Class <- as.factor(data_balanced_under$Class)

# shows that it has been changed to factor
str(data_balanced_under$Class)


### Split data into training and testing data 
set.seed(120)
splitCreditCardData <-sample.split(Y=data_balanced_under$Class, SplitRatio = 0.7)
trainData <- data_balanced_under[splitCreditCardData,]
testData <- data_balanced_under[!splitCreditCardData,]

dim(trainData)
dim(testData)

# Fitting Random Forest to the train dataset
classifier_RF = randomForest(x = trainData[-31],
                             y = trainData$Class,
                             ntree = 500)
classifier_RF #error rare: 5.24%

# Predicting the Test set results
y_pred = predict(classifier_RF, newdata = testData[-31])

# Confusion Matrix
confusion_mtx = table(testData[, 30], y_pred)
confusion_mtx

# Plotting model
plot(classifier_RF)

# Importance plot
importance(classifier_RF)

# Variable importance plot
varImpPlot(classifier_RF)

################################ Oversampling####################
## Using oversampling as technique
## use undersampling data on dataset because we want data with the initial observations for minority class

data_balanced_over <- ovun.sample(Class ~ ., data = data_balanced_under, method = "over", N = 28000, seed = 1)$data
table(data_balanced_over$Class) ## still unbalanced. No need to run a model


################################ Both Undersampling and Oversampling####################
## using both undersampling and oversampling
data_balanced_both <- ovun.sample(Class ~ ., data = creditcarddata, method = "both", p=0.4, N=28000, seed = 1)$data
table(data_balanced_both$Class) ## more records of both classes when this is used. Makes case for a more accurate model

# converting class feature(target variable) to factor
data_balanced_both$Class <- as.factor(data_balanced_both$Class)

splitCreditCardDataBalancedSampling <-sample.split(Y=data_balanced_both$Class, SplitRatio = 0.6)
trainDataBalancedSampling <- data_balanced_both[splitCreditCardDataBalancedSampling,]
testDataBalancedSampling <- data_balanced_both[!splitCreditCardDataBalancedSampling,]


# Fitting Random Forest to the train Balanced Sampling dataset
classifier_RF_BalancedSampling = randomForest(x = trainDataBalancedSampling[-31],
                                              y = trainDataBalancedSampling$Class,
                                              ntree = 500)
classifier_RF_BalancedSampling # error rate: 0.04% 

# Predicting the Test set results
y_pred_BalancedSampling = predict(classifier_RF_BalancedSampling, newdata = testDataBalancedSampling[-31])

# Confusion Matrix
confusion_mtx__BalancedSampling = table(testDataBalancedSampling[, 30], y_pred_BalancedSampling)
confusion_mtx__BalancedSampling


# Plotting model
plot(classifier_RF_BalancedSampling)


# Using AOC (Area Under Curve)
## undersampling
underSamplingRoc <- roc(as.numeric(testData$Class), y_pred, plot = TRUE, col = "blue")
underSamplingRoc # roc
auc(underSamplingRoc) # auc

## both under and over sampling
underBothSamplingRoc <- roc(as.numeric(testDataBalancedSampling$Class), as.numeric(y_pred_BalancedSampling), plot = TRUE, col = "blue")
underBothSamplingRoc # roc
auc(underBothSamplingRoc) # auc







##Logistic Regression with underSampled dataset

log_mod= glm (Class~., data=trainData, family = "binomial")
summary(log_mod)
plot(log_mod)
logLik(log_mod)

coef(log_mod)

## View Test dataset to view count of 0 and 1
plot(trainData$Class)


#using model to predict probability 
probTest=predict(log_mod, testData, type = "response", probility=TRUE)  # Predict probabilities

plot(probTest) ## View Test probtest to determine split coeff



## to obtain different predictions using (0.01,0.02,0.05) as cutoff

get_logistic_pred = function(mod, data, res = "y", pos = 1, neg = 0, cut = 0.5) {
  probs = predict(mod, newdata = data, type = "response")
  
  
  ifelse(probs > cut, pos, neg)
}

test_pred_02 = get_logistic_pred(log_mod, data = testData, res = "default", 
                                  pos = "1", neg = "0", cut = 0.02)
test_pred_2 = get_logistic_pred(log_mod, data = testData, res = "default", 
                                 pos = "1", neg = "0", cut = 0.2)
test_pred_5 = get_logistic_pred(log_mod, data = testData, res = "default", 
                                 pos = "1", neg = "0", cut = 0.5)

#evaluate accuracy 
test_pred_02 = table(predicted = test_pred_02, actual = testData$Class)
test_pred_2 = table(predicted = test_pred_2, actual = testData$Class)
test_pred_5 = table(predicted = test_pred_5, actual = testData$Class)

test_con_mat_02 = confusionMatrix(test_pred_02, positive = "1")
test_con_mat_2 = confusionMatrix(test_pred_2, positive = "1")
test_con_mat_5 = confusionMatrix(test_pred_5, positive = "1")

metrics = rbind(
  
  c(test_con_mat_02$overall["Accuracy"], 
    test_con_mat_02$byClass["Sensitivity"], 
    test_con_mat_02$byClass["Specificity"]),
  
  c(test_con_mat_2$overall["Accuracy"], 
    test_con_mat_2$byClass["Sensitivity"], 
    test_con_mat_2$byClass["Specificity"]),
  
  c(test_con_mat_5$overall["Accuracy"], 
    test_con_mat_5$byClass["Sensitivity"], 
    test_con_mat_5$byClass["Specificity"])
  
)

#We see then sensitivity decreases as the cutoff is increased. Conversely, specificity increases as the cutoff increases.
rownames(metrics) = c("c = 0.02", "c = 0.2", "c = 0.5")
metrics

# Now we recode probability to classification with c=0.5
predVal <- ifelse(probTest >= 0.5, 1, 0)
predTest <- factor(predVal, levels = c(0,1))
actualTest <-testData$Class

#Confusion Matrix
confusionMatrix(as.factor(actualTest), predTest)

#Aera Under the Curve
roc_val=roc(actualTest ~ probTest, plot=TRUE,print.auc =TRUE,col="red" )




###Decision Tree Model using underSample dataset

#fitting model
decisionTree_model <- rpart(Class ~ . , data= trainData, method = 'class')

predicted_val <- predict(decisionTree_model, testData, type = 'class')
probability <- predict(decisionTree_model, testData, type = 'prob')
rpart.plot(decisionTree_model)

summary(decisionTree_model) # detailed summary of splits
decisionTree_model #prints the rules


##Decision trees are also useful for examining feature importance, ergo, how much 
##predictive power lies in each feature. You can use the varImp() function to find out. 
##The following snippet calculates the importances and sorts them descendingly:
importances <- varImp(decisionTree_model)
importances %>%
  arrange(desc(Overall))

hist(importances)

###Use the fitted model to do predictions for the test data

predTest_d <- predict(decisionTree_model, testData, type="class")
predTest_d


confusionMatrix(as.factor(testData$Class), predTest_d)

str(testDataset$Class)

str(predTest_d)

probTest_d <- predict(decisionTree_model, testData, type="prob") #predicting probabily 

actualTest_d <- testData$Class


###Create Confusion Matrix and compute the misclassification error
confusionMatrix(as.factor(actualTest_d), predTest_d)

## Visualization of probabilities
hist(probTest_d[,2], breaks = 100)

## ROC and Area Under the Curve (AUC)
ROC <- roc(actualTest_d, probTest_d[,2])
plot(ROC, col="blue",print.auc=TRUE)
AUC <- auc(ROC)






