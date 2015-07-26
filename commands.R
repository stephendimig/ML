library(caret)
require(gridExtra)

df <- read.csv("pml-training.csv", stringsAsFactors=FALSE, na.strings=c("NA", "#DIV/0!"))
inTrain <- createDataPartition(df$classe, p=0.7, list=FALSE)
training <- df[inTrain, ]
testing <- df[-inTrain, ]

plot1 <- qplot(X, roll_belt, data=training, colour=classe)
plot2 <- qplot(X, total_accel_dumbbell, data=training, colour=classe)
plot3 <- qplot(X, pitch_forearm, data=training, colour=classe)
plot4 <- qplot(X, roll_arm, data=training, colour=classe)
grid.arrange(plot1, plot2, plot3, plot4, ncol=2)


nzf <- nearZeroVar(training)
training <- training[, -nzf]
training <- training[names(training) != "raw_timestamp_part_1" ]
training <- training[names(training) != "raw_timestamp_part_2" ]
training <- training[names(training) != "user_name" ]
training <- training[names(training) != "cvtd_timestamp" ]
training <- training[names(training) != "X" ]
training <- training[names(training) != "num_window" ]
training <- training[,colSums(is.na(training))<13000]
training$classe <- as.factor(training$classe)

modelFit <- train(as.numeric(training$classe) ~., data=training[, -55], method="lm")
pred <- sapply(predict(modelFit, newdata=testing), round)
confusionMatrix(as.numeric(as.factor(testing$classe)), pred)

modelFit <- train(as.numeric(training$classe) ~., data=training[, -55], method="glm")
pred <- sapply(predict(modelFit, newdata=testing), round)
confusionMatrix(as.numeric(as.factor(testing$classe)), pred)


modelFit <- train(training$classe ~., data=training[, -55], method="gbm")
pred <- predict(modelFit, newdata=testing)
confusionMatrix(as.factor(testing$classe), pred)

ctrl <- trainControl(method="cv", number = 10, repeats = 4)
modelFit <- train(training$classe ~., data=training[, -55], method="rf", trControl = ctrl, ntree=50)
pred <- predict(modelFit, newdata=testing)
confusionMatrix(as.factor(testing$classe), pred)
varImp(modelFit)

rtesting <- read.csv("pml-testing.csv", stringsAsFactors=FALSE, na.strings=c("NA", "#DIV/0!"))
pred <- predict(modelFit, newdata=rtesting)
pml_write_files = function(x)
{
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
