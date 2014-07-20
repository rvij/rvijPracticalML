library(caret); library(ggplot2)
set.seed(5469)

# Load the test file:
dataset  <- read.csv("../pml-training.csv", na.strings=c("NA",""), strip.white=TRUE)
dim(dataset)

# Check the summary of the data frame.
summary(dataset)

# There are large number of attributes that have NAs, we will ignore those variable as they may not contribute to 
# any predictive capability, also there are some time variables that will not be used

# Cleaning the data:
isNAcolumn <- apply(dataset, 2, function(x) { sum(is.na(x)) })

dataset <- subset(dataset[, which(isNA == 0)], 
                  select=-c(X, user_name, new_window, num_window, 
                            raw_timestamp_part_1, raw_timestamp_part_2, 
                            cvtd_timestamp))
dim(dataset)

## Dimensionality of the new dataset that will be used for prediction
# Spliting the dataset:
inTrain <- createDataPartition(dataset$classe, p=0.8, list=FALSE)
train_set <- dataset[inTrain,]
validation_set <- dataset[-inTrain,]

# Cross-Validation folds 
folds <- createFolds(y=dataset$classe,k=10, list=TRUE,returnTrain=TRUE)
sapply(folds,length)


