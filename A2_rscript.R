rm(list = ls())
set.seed(33520615)
WD = read.csv("WinnData.csv")
WD = WD[sample(nrow(WD), 5000, replace = FALSE),]
WD = WD[,c(sort(sample(1:30,20, replace = FALSE)), 31)]

# Import necessary packages
library(caret)
library(tree)
library(e1071)
library(ROCR)
library(randomForest)
library(adabag)
library(rpart)
library(glmnet)
library(neuralnet)
library(pROC)
library(xgboost)

# Q1 exploring the data
table(WD$Class)
prop.table(table(WD$Class))
summary(WD[, -ncol(WD)])
sapply(WD[, -ncol(WD)], sd)

# Q2 data preprocessing
sum(is.na(WD))
WD$Class <- factor(WD$Class, levels=c(0,1),
                         labels=c("Other","Oats"))
attribute_to_omit = nearZeroVar(WD[, -ncol(WD)], saveMetrics = TRUE)
attribute_to_omit[attribute_to_omit$nzv == TRUE, ]
index_to_omit = nearZeroVar(WD, saveMetrics = FALSE)
print(names(WD)[index_to_omit])
WD = WD[, -index_to_omit]
head(WD)

#Q3 spliting train and test data
set.seed(33520615) 
train.row = sample(1:nrow(WD), 0.7*nrow(WD))
WD.train = WD[train.row,]
WD.test = WD[-train.row,]

#Q4 fitting the machine learning models
# Decision tree
WD.dc.fit = tree(Class ~ ., data=WD.train)
WD.dc.fit
summary(WD.dc.fit)
plot(WD.dc.fit)
text(WD.dc.fit, pretty = 0)

# Naive Bayess
WD.nb.fit = naiveBayes(Class ~ . , data = WD.train)
WD.nb.fit

# Bagging
WD.bag.fit = adabag::bagging(Class ~ . , data = WD.train, mfinal = 5)
WD.bag.fit

# Boosting
WD.boost.fit = adabag::boosting(Class ~ . , data = WD.train, mfinal=10)
WD.boost.fit

# Random Forest
WD.rf.fit = randomForest(Class ~ . , data = WD.train)
WD.rf.fit

#Q5 Finding each models' confusion matrix and other metrics
# decision tree
WD.pred.tree = predict(WD.dc.fit, WD.test, type="class")
confusionMatrix(WD.pred.tree, WD.test$Class, mode = "everything", positive= "Oats")

# Naive bayes
WD.pred.bayes = predict(WD.nb.fit, newdata = WD.test)
confusionMatrix(WD.pred.bayes, WD.test$Class, mode = "everything", positive="Oats")

# Bagging
WD.pred.bag = predict.bagging(WD.bag.fit, WD.test)
WD.pred.bag$confusion
predicted_classes = factor(WD.pred.bag$class, levels = c("Other", "Oats"))
confusionMatrix(predicted_classes, WD.test$Class, mode = "everything", positive = "Oats")

# Boosting 
WD.pred.boost = predict.boosting(WD.boost.fit, newdata=WD.test)
WD.pred.boost$confusion
predicted_classes = factor(WD.pred.boost$class, levels = c("Other", "Oats"))
confusionMatrix(predicted_classes, WD.test$Class, mode = "everything", positive = "Oats")

# Random Forest
WD.pred.rf = predict(WD.rf.fit, WD.test)
confusionMatrix(WD.pred.rf, WD.test$Class, mode = "everything", positive="Oats")

# Q6 Finding AUC and plotting ROC curve for each model
# Decision Tree 
dtree.pred = predict(WD.dc.fit, WD.test, type = "vector")
colnames(dtree.pred)
labels = ifelse(WD.test$Class == "Oats", 1, 0)
dtree.prediction = ROCR::prediction(dtree.pred[, 2], labels)
dtree.perf = performance(dtree.prediction,"tpr","fpr")
plot(dtree.perf, main = "ROC-AUC", col="blue")
abline(0,1,lty=2)
dtree.auc = performance(dtree.prediction, "auc")@y.values[[1]]
dtree.auc
 
# Naive Bayes
nb.pred = predict(WD.nb.fit, WD.test, type = 'raw')
nb.prediction = ROCR::prediction(nb.pred[, "Oats"], labels)
nb.perf = performance(nb.prediction,"tpr","fpr")
plot(nb.perf, add=TRUE, col = "blueviolet")
nb.auc = performance(nb.prediction, "auc")@y.values[[1]]
nb.auc

# Bagging
bag.prediction = ROCR::prediction(WD.pred.bag$prob[,2], labels)
bag.perf = performance(bag.prediction,"tpr","fpr")
plot(bag.perf, add=TRUE, col = "green")
bag.auc = performance(bag.prediction, "auc")@y.values[[1]]
bag.auc

# Boosting
boost.prediction = ROCR::prediction(WD.pred.boost$prob[,2], labels)
boost.perf = performance(boost.prediction,"tpr","fpr")
plot(boost.perf, add=TRUE, col = "red")
boost.auc = performance(boost.prediction, "auc")@y.values[[1]]
boost.auc

# Random Forest
rf.pred = predict(WD.rf.fit, WD.test, type="prob")
rf.prediction = ROCR::prediction( rf.pred[,2], labels)
rf.perf = performance(rf.prediction,"tpr","fpr")
plot(rf.perf, add=TRUE, col = "lightblue",)
rf.auc = performance(rf.prediction, "auc")@y.values[[1]]
rf.auc

# Adding legends
legend("bottomright", legend = c("Decision Tree",
                                 "Naive Bayes",
                                 "Bagging",
                                 "Boosting",
                                 "Random Forest"),
      col = c("blue", "blueviolet", "green", "red", "lightblue"),
      lwd = 2)

# Q 8 
# Showing the importance of each attribute in each model
cat("\n#Decision Tree Attribute Importance\n")
print(summary(WD.dc.fit))
cat("\n#Naive Bayes Attribute Importance\n")
print(varImp(caret::train(Class ~ ., data = WD.train, method = "naive_bayes")))
cat("\n#Baging Attribute Importance\n")
print(WD.bag.fit$importance)
cat("\n#Boosting Attribute Importance\n")
print(WD.boost.fit$importance)
cat("\n#Random Forest Attribute Importance\n")
print(WD.rf.fit$importance)

#LASSO regression for penalising 
x = model.matrix(Class ~ ., WD.train)[,-1]
y = as.numeric(WD.train$Class == "Oats")
lasso = cv.glmnet(x, y, family = "binomial", alpha = 1)
lasso_model = glmnet(x, y, family = "binomial", alpha = 1, lambda = lasso$lambda.min)
print(coef(lasso_model))

# Q 10 
#create the simple model (single-split decision tree) to predict
simple.pred = factor(ifelse(WD.test$A26 < 15.624, "Other", "Oats"), levels = c("Other", "Oats"))

# AUC and other metrics for the simple model
confusionMatrix(simple.pred, WD.test$Class, mode = "everything", positive = "Oats")
labels = ifelse(WD.test$Class=="Oats", 1, 0)
simple.prediction = ROCR::prediction(WD.test$A26, labels)
performance(simple.prediction, "auc")@y.values[[1]]

# Q 11 
# create a best tree-based classifier
set.seed(33520615)
tune_results = tuneRF(x = WD.train[, setdiff(names(WD.train), "Class")],
  y = WD.train$Class, stepFactor = 1.5, ntreeTry = 250, improve = 0.01)

# get the best mtry
best_mtry = tune_results[which.min(tune_results[, "OOBError"]), "mtry"]
best_mtry

# balance the dataset ie undersample the 'Other' class
class_counts = table(WD.train$Class)
sampsize = rep(min(class_counts), length(class_counts))
names(sampsize) = names(class_counts)

# fit the final best rf
set.seed(33520615)
best.rf.fit = randomForest(Class ~ ., data = WD.train, ntree = 500,
                            mtry = best_mtry, sampsize = sampsize, importance =TRUE)

#Performance metrics
best.rf.pred = predict(best.rf.fit, WD.test)
confusionMatrix(best.rf.pred, WD.test$Class, mode = "everything", positive = "Oats")
best.rf.prob = predict(best.rf.fit, WD.test, type="prob")
best.rf.prediction = ROCR::prediction(best.rf.prob[, 2], labels)
best.rf.auc = performance(best.rf.prediction, "auc")@y.values[[1]]
best.rf.auc #auc


# Q12 Implementing ANN and its results
# Select top 10 variables by RF importance
rf_imp = varImp(best.rf.fit)
imp_df = as.data.frame(rf_imp)
imp_df$Variable = rownames(imp_df)
imp_df = imp_df[order(imp_df$Oats, decreasing = TRUE), ]
top10 = imp_df$Variable[1:10]
top10
train_top = WD.train[, c(top10, "Class")]
colnames(train_top)

# Oversample the minority class 'Oats'
set.seed(33520615)
sampled_train = upSample(x = train_top[, top10], y = train_top$Class, yname = "Class")
table(sampled_train$Class)

# Scale inputs and convert Class to numeric function
scale_vars = function(df, vars) {
  df_scaled = as.data.frame(scale(df[, vars]))
  df_scaled$Class = ifelse(df$Class == "Oats", 1, 0)
  df_scaled
}
train_ann = scale_vars(sampled_train, top10)
test_ann = scale_vars(WD.test, top10)

# Build the formula for the neuralnet
ann_formula = as.formula(paste("Class ~", paste(top10, collapse = " + ")))

# Train the network
set.seed(33520615)
WD.ann.fit = neuralnet(formula = ann_formula, data = train_ann, hidden = 4)

# Make predictions on the original test set
ann.raw.pred = compute(WD.ann.fit, test_ann[, top10])
ann.probs = ann.raw.pred$net.result[,1]
ann.pred = factor(ifelse(ann.probs > 0.5, "Oats", "Other"),
                   levels = c("Other","Oats"))

# Performance metrics
confusionMatrix(ann.pred, WD.test$Class, mode = "everything", positive  = "Oats")
auc.ann = auc(roc(WD.test$Class, ann.probs, levels = c("Other", "Oats")))
auc.ann

# Q13
?xgboost #documentation


x = WD.train[, setdiff(names(WD.train), "Class")]
y = WD.train$Class

# 2. upsample the oats to match others
set.seed(33520615)
sampled_train_xgb <- upSample(x = x, y = y, yname = "Class")
table(sampled_train_xgb$Class)

# Convert factors to numeric labels
label_train <- ifelse(sampled_train_xgb$Class == "Oats", 1, 0)
label_test <- ifelse(WD.test$Class == "Oats",   1, 0)

# data preprocessing
train_xgb = xgb.DMatrix(data = as.matrix(sampled_train_xgb[, setdiff(names(sampled_train_xgb),"Class")]),
                     label = label_train)
test_xgb = xgb.DMatrix(data = as.matrix(WD.test[,  setdiff(names(WD.test),"Class")]),
                    label = label_test)
# specify parameters
parameters = list(objective = "binary:logistic", eval_metric = "auc")

# train
set.seed(33520615)
wd.xgb.fit = xgb.train(params = parameters, data = train_xgb, nrounds = 200, watchlist = list(train = train_xgb),
                       early_stopping_rounds = 10, verbose = 0)

# predict
xgb.probs = predict(wd.xgb.fit, test_xgb)
xgb.pred = factor(ifelse(xgb.probs > 0.5, "Oats", "Other"), levels = c("Other","Oats"))

# performance metrics
confusionMatrix(xgb.pred, WD.test$Class, mode = "everything", positive = "Oats")
auc.xgb = auc(roc(WD.test$Class, xgb.probs, levels = c("Other","Oats")))
auc.xgb


