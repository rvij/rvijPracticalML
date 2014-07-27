library(caret); library(ggplot2)
set.seed(5469)

# Load the test file:
dataset  <- read.csv("../pml-training.csv", na.strings=c("NA",""), strip.white=TRUE)
dim(dataset)
dataset_test <- read.csv("../pml-testing.csv", na.strings=c("NA",""), strip.white=T)

# Check the summary of the data frame.
summary(dataset)

# There are large number of attributes that have NAs, we will ignore those variable as they may not contribute to 
# any predictive capability, also there are some time variables that will not be used

createTidyDataset = function(data){
  # Cleaning the data:
  isNAcolumns <- apply(data, 2, function(x) { sum(is.na(x)) })
  
  tidyData <- subset(data[, which(isNAcolumns == 0)], 
                     select=-c(X, user_name, new_window, num_window, 
                               raw_timestamp_part_1, raw_timestamp_part_2, 
                               cvtd_timestamp))
}

dataset <- createTidyDataset(dataset)
  
dim(dataset)

dataset_test <- createTidyDataset(dataset_test)

dim(dataset_test)

## Dimensionality of the new dataset that will be used for prediction
# Spliting the dataset:
inTrain <- createDataPartition(dataset$classe, p=0.8, list=FALSE)
train_set <- dataset[inTrain,]
validation_set <- dataset[-inTrain,]

# Test 1: Random forest classifier using cross validation contol
ctrl <- trainControl(allowParallel=TRUE, method="cv")
rfmodel <- train(classe ~ ., data=train_set, model="rf", trControl=ctrl)
predictor <- predict(rfmodel, newdata=validation_set)

# Error on valid_set:
sum(predictor == validation_set$classe) / length(predictor)
confusionMatrix(validation_set$classe, predictor)$table

# Classification for test_set:

rfpredictions <- predict(rfmodel, newdata=dataset_test)

# Test 3: Implementing Support Vector Machine most important variable for the ramdom forest predictor:
impvar <- varImp(rfmodel)

impvar$importance$var <- rownames(impvar$importance)

top10var <- impvar$importance[order(impvar$importance$Overall, decreasing = TRUE),c("var")][1:10]



# Train the SVM on the dataset reduce to Top ten variables
dataset_topten <- subset(dataset, select=c(top10var, "classe"))

dataset_topten_test <- subset(dataset_test, select=c(top10var))

dataset_topten_test$classe <- NA

dim(dataset_test)

svm <- train(classe ~ ., data=dataset_topten[inTrain,], model="svm", trControl=ctrl)

predictor_svm <- predict(svm, newdata=validation_set)


sum(predictor_svm == validation_set$classe) / length(predictor_svm)
confusionMatrix(validation_set$classe, predictor_svm)$table

results <- predict(svm, newdata=dataset_test)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(rfpredictions)

library(RWeka)

write.arff(dataset_topten, file="datasetTop10.arff")
write.arff(dataset_topten_test, file="dataset_test.arff")
