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


## Dimensionality of the new dataset that will be used for prediction
# Spliting the dataset:
inTrain <- createDataPartition(dataset$classe, p=0.8, list=FALSE)
train_set <- dataset[inTrain,]
validation_set <- dataset[-inTrain,]

# Cross-Validation folds 
folds <- createFolds(y=dataset$classe,k=10, list=TRUE,returnTrain=TRUE)
sapply(folds,length)




