Predicting Quality of Weight Lifting Exercises
========================================================
author: Stephen Dimig
date: Sun Jul 26 13:32:22 2015

Introduction
========================================================
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 


Introduction (continued)
========================================================
The goal of this project is to predict the manner in which they did a weight lifting exercise (the "classe" variable) from a subset of features whose values are taken from readings on accelerometers on the belt, forearm, arm, and dumbell of 6 participants who ere asked to perform barbell lifts correctly and incorrectly in 5 different ways. This report will:

* Examine several models and pick the best one
* Describe how cross validation was used
* Describe the sample error
* Use the model to predict 20 different test cases


Reading the Data
========================================================
The data was downloaded from a link on the Coursera website and read in as a csv file.

* "#DIV/0!" strings were considered NA values.


```r
library(ggplot2)
library(caret)
require(gridExtra)

df <- read.csv("pml-training.csv", stringsAsFactors=FALSE, na.strings=c("NA", "#DIV/0!"))
inTrain <- createDataPartition(df$classe, p=0.7, list=FALSE)
training <- df[inTrain, ]
testing <- df[-inTrain, ]
```

Exploratory Data Analysis
========================================================
This data is tough to get a feel for due to the large number of features involved. What I tried to do was plot a subset of the features against the X value which is a monotomically increasing integer that is unique for each case, and color the results by the classe variable. The features on the left showed some correlation while the ones on the right appear uncorrelated with classe.

Exploratory Data Analysis (continued)
========================================================
![plot of chunk unnamed-chunk-2](WeightLiftingExercises-figure/unnamed-chunk-2-1.png) 

Cleaning the Data
========================================================
Several features were removed from the set of predictors after an examination of the data.

* All features with near zero variance were removed. These columns have little variance so they cannot contribute to a prediction.
* The timestamp related variables were removed (raw_timestamp_part_1, raw_timestamp_part_2, and cvtd_timestamp).
* The X column was removed.
* The num_window column was removed.
* All columns with over 13000 NA values in the training set were removed. These might have good information but they were too sparse for a predictor.

Cleaning the Data (code)
========================================================

```r
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
```

Prediction Models 
========================================================
The following models were examined to determine the most accurate.

* Linear
* Boosting
* Random Forest


Cross Validation
========================================================
* The pml-training data was split into a a training and test set.
* The training set was created using roughly 70% of the pml-training data
* The testing set was created using roughly 30% of the pml-training data
* K-Fold cross validation with 10 folds and 4 repeats was used for all of the models.


Linear Model
========================================================
The first type of model attempted was a linear model. A linear model attempts to fit a straight line of the form:

* y = B0 + (B1 * X1) + (B2 * X2) + ... + (Bn * Xn) + e

In this case it is a multi-variate linear regression where the X values are the features and the Y value is the predicted value for the classe variable.

Linear Model (code)
========================================================

```r
set.seed(3311995)
ctrl <- trainControl(method="cv", number = 10, repeats = 4)
modelFit <- train(as.numeric(training$classe) ~., data=training[, -55], method="lm", trControl = ctrl)
pred <- sapply(predict(modelFit, newdata=testing), round)
```

Linear Model (confusion matrix table)
========================================================

```
          Reference
Prediction  -1   0   1   2   3   4   5   6   7
        -1   0   0   0   0   0   0   0   0   0
        0    0   0   0   0   0   0   0   0   0
        1    3  31 639 715 274  12   0   0   0
        2    0   3  50 429 555 101   1   0   0
        3    0   0   5 216 669 134   2   0   0
        4    0   0   0 117 461 364  22   0   0
        5    1   1   2  44 335 430 204  59   6
        6    0   0   0   0   0   0   0   0   0
        7    0   0   0   0   0   0   0   0   0
```

Boosting
========================================================
Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of (possibly) weak prediction models, typically decision trees. It builds the model in stage-wise fashion, by weighting the results and adding them up to form a stronger predictor from the compostion.


Boosting (code)
========================================================

```r
set.seed(3311995)
ctrl <- trainControl(method="cv", number = 10, repeats = 4)
modelFit <- train(training$classe ~., data=training[, -55], method="gbm", trControl = ctrl)
pred <- predict(modelFit, newdata=testing)
```


```
Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1302
     2        1.5232             nan     0.1000    0.0880
     3        1.4644             nan     0.1000    0.0672
     4        1.4205             nan     0.1000    0.0523
     5        1.3854             nan     0.1000    0.0494
     6        1.3528             nan     0.1000    0.0432
     7        1.3252             nan     0.1000    0.0387
     8        1.3010             nan     0.1000    0.0353
     9        1.2781             nan     0.1000    0.0320
    10        1.2555             nan     0.1000    0.0296
    20        1.1033             nan     0.1000    0.0190
    40        0.9292             nan     0.1000    0.0077
    60        0.8212             nan     0.1000    0.0069
    80        0.7431             nan     0.1000    0.0050
   100        0.6808             nan     0.1000    0.0023
   120        0.6291             nan     0.1000    0.0031
   140        0.5851             nan     0.1000    0.0029
   150        0.5662             nan     0.1000    0.0016

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1886
     2        1.4886             nan     0.1000    0.1314
     3        1.4051             nan     0.1000    0.1049
     4        1.3380             nan     0.1000    0.0774
     5        1.2881             nan     0.1000    0.0716
     6        1.2420             nan     0.1000    0.0580
     7        1.2043             nan     0.1000    0.0706
     8        1.1611             nan     0.1000    0.0575
     9        1.1264             nan     0.1000    0.0465
    10        1.0966             nan     0.1000    0.0405
    20        0.8888             nan     0.1000    0.0243
    40        0.6800             nan     0.1000    0.0106
    60        0.5472             nan     0.1000    0.0065
    80        0.4635             nan     0.1000    0.0044
   100        0.3994             nan     0.1000    0.0039
   120        0.3478             nan     0.1000    0.0027
   140        0.3097             nan     0.1000    0.0026
   150        0.2907             nan     0.1000    0.0016

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2264
     2        1.4625             nan     0.1000    0.1629
     3        1.3600             nan     0.1000    0.1234
     4        1.2816             nan     0.1000    0.1141
     5        1.2102             nan     0.1000    0.0877
     6        1.1540             nan     0.1000    0.0802
     7        1.1026             nan     0.1000    0.0603
     8        1.0640             nan     0.1000    0.0708
     9        1.0205             nan     0.1000    0.0634
    10        0.9817             nan     0.1000    0.0455
    20        0.7519             nan     0.1000    0.0233
    40        0.5242             nan     0.1000    0.0117
    60        0.4025             nan     0.1000    0.0059
    80        0.3232             nan     0.1000    0.0022
   100        0.2660             nan     0.1000    0.0022
   120        0.2230             nan     0.1000    0.0017
   140        0.1910             nan     0.1000    0.0022
   150        0.1767             nan     0.1000    0.0012

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1271
     2        1.5218             nan     0.1000    0.0869
     3        1.4634             nan     0.1000    0.0691
     4        1.4186             nan     0.1000    0.0541
     5        1.3827             nan     0.1000    0.0473
     6        1.3512             nan     0.1000    0.0458
     7        1.3217             nan     0.1000    0.0376
     8        1.2976             nan     0.1000    0.0318
     9        1.2762             nan     0.1000    0.0278
    10        1.2576             nan     0.1000    0.0350
    20        1.1036             nan     0.1000    0.0158
    40        0.9304             nan     0.1000    0.0096
    60        0.8209             nan     0.1000    0.0068
    80        0.7419             nan     0.1000    0.0058
   100        0.6808             nan     0.1000    0.0052
   120        0.6274             nan     0.1000    0.0031
   140        0.5853             nan     0.1000    0.0031
   150        0.5646             nan     0.1000    0.0025

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1880
     2        1.4895             nan     0.1000    0.1320
     3        1.4042             nan     0.1000    0.1015
     4        1.3386             nan     0.1000    0.0846
     5        1.2842             nan     0.1000    0.0757
     6        1.2367             nan     0.1000    0.0667
     7        1.1945             nan     0.1000    0.0571
     8        1.1581             nan     0.1000    0.0608
     9        1.1203             nan     0.1000    0.0460
    10        1.0909             nan     0.1000    0.0450
    20        0.8886             nan     0.1000    0.0245
    40        0.6721             nan     0.1000    0.0101
    60        0.5541             nan     0.1000    0.0056
    80        0.4692             nan     0.1000    0.0054
   100        0.4015             nan     0.1000    0.0026
   120        0.3515             nan     0.1000    0.0038
   140        0.3075             nan     0.1000    0.0024
   150        0.2902             nan     0.1000    0.0011

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2341
     2        1.4628             nan     0.1000    0.1601
     3        1.3581             nan     0.1000    0.1168
     4        1.2808             nan     0.1000    0.1151
     5        1.2098             nan     0.1000    0.0965
     6        1.1503             nan     0.1000    0.0724
     7        1.1051             nan     0.1000    0.0657
     8        1.0640             nan     0.1000    0.0633
     9        1.0244             nan     0.1000    0.0605
    10        0.9861             nan     0.1000    0.0464
    20        0.7581             nan     0.1000    0.0230
    40        0.5311             nan     0.1000    0.0130
    60        0.4087             nan     0.1000    0.0089
    80        0.3277             nan     0.1000    0.0038
   100        0.2689             nan     0.1000    0.0036
   120        0.2249             nan     0.1000    0.0020
   140        0.1903             nan     0.1000    0.0022
   150        0.1762             nan     0.1000    0.0008

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1305
     2        1.5221             nan     0.1000    0.0895
     3        1.4632             nan     0.1000    0.0651
     4        1.4191             nan     0.1000    0.0558
     5        1.3832             nan     0.1000    0.0507
     6        1.3501             nan     0.1000    0.0403
     7        1.3242             nan     0.1000    0.0388
     8        1.2989             nan     0.1000    0.0345
     9        1.2767             nan     0.1000    0.0309
    10        1.2542             nan     0.1000    0.0269
    20        1.0992             nan     0.1000    0.0165
    40        0.9298             nan     0.1000    0.0105
    60        0.8229             nan     0.1000    0.0063
    80        0.7398             nan     0.1000    0.0051
   100        0.6771             nan     0.1000    0.0036
   120        0.6257             nan     0.1000    0.0033
   140        0.5820             nan     0.1000    0.0015
   150        0.5634             nan     0.1000    0.0031

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1840
     2        1.4887             nan     0.1000    0.1327
     3        1.4043             nan     0.1000    0.1060
     4        1.3365             nan     0.1000    0.0855
     5        1.2827             nan     0.1000    0.0735
     6        1.2355             nan     0.1000    0.0653
     7        1.1946             nan     0.1000    0.0627
     8        1.1560             nan     0.1000    0.0593
     9        1.1192             nan     0.1000    0.0533
    10        1.0863             nan     0.1000    0.0415
    20        0.8834             nan     0.1000    0.0217
    40        0.6710             nan     0.1000    0.0104
    60        0.5504             nan     0.1000    0.0084
    80        0.4611             nan     0.1000    0.0042
   100        0.3981             nan     0.1000    0.0038
   120        0.3466             nan     0.1000    0.0026
   140        0.3065             nan     0.1000    0.0021
   150        0.2873             nan     0.1000    0.0026

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2349
     2        1.4609             nan     0.1000    0.1651
     3        1.3568             nan     0.1000    0.1315
     4        1.2742             nan     0.1000    0.1025
     5        1.2089             nan     0.1000    0.0876
     6        1.1527             nan     0.1000    0.0737
     7        1.1067             nan     0.1000    0.0706
     8        1.0619             nan     0.1000    0.0615
     9        1.0232             nan     0.1000    0.0623
    10        0.9831             nan     0.1000    0.0481
    20        0.7483             nan     0.1000    0.0233
    40        0.5262             nan     0.1000    0.0108
    60        0.4084             nan     0.1000    0.0087
    80        0.3240             nan     0.1000    0.0042
   100        0.2662             nan     0.1000    0.0028
   120        0.2240             nan     0.1000    0.0016
   140        0.1905             nan     0.1000    0.0019
   150        0.1763             nan     0.1000    0.0014

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1293
     2        1.5230             nan     0.1000    0.0894
     3        1.4640             nan     0.1000    0.0677
     4        1.4194             nan     0.1000    0.0562
     5        1.3830             nan     0.1000    0.0508
     6        1.3497             nan     0.1000    0.0397
     7        1.3235             nan     0.1000    0.0390
     8        1.2989             nan     0.1000    0.0336
     9        1.2778             nan     0.1000    0.0315
    10        1.2569             nan     0.1000    0.0328
    20        1.0995             nan     0.1000    0.0192
    40        0.9298             nan     0.1000    0.0087
    60        0.8246             nan     0.1000    0.0077
    80        0.7440             nan     0.1000    0.0039
   100        0.6846             nan     0.1000    0.0039
   120        0.6304             nan     0.1000    0.0026
   140        0.5878             nan     0.1000    0.0019
   150        0.5693             nan     0.1000    0.0018

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1835
     2        1.4892             nan     0.1000    0.1362
     3        1.4039             nan     0.1000    0.1027
     4        1.3379             nan     0.1000    0.0844
     5        1.2829             nan     0.1000    0.0725
     6        1.2358             nan     0.1000    0.0720
     7        1.1919             nan     0.1000    0.0569
     8        1.1560             nan     0.1000    0.0567
     9        1.1192             nan     0.1000    0.0434
    10        1.0914             nan     0.1000    0.0463
    20        0.8908             nan     0.1000    0.0191
    40        0.6799             nan     0.1000    0.0125
    60        0.5526             nan     0.1000    0.0076
    80        0.4699             nan     0.1000    0.0058
   100        0.4036             nan     0.1000    0.0032
   120        0.3534             nan     0.1000    0.0033
   140        0.3102             nan     0.1000    0.0022
   150        0.2915             nan     0.1000    0.0026

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2324
     2        1.4603             nan     0.1000    0.1573
     3        1.3584             nan     0.1000    0.1174
     4        1.2844             nan     0.1000    0.1075
     5        1.2164             nan     0.1000    0.0898
     6        1.1606             nan     0.1000    0.0810
     7        1.1103             nan     0.1000    0.0695
     8        1.0665             nan     0.1000    0.0659
     9        1.0253             nan     0.1000    0.0579
    10        0.9895             nan     0.1000    0.0573
    20        0.7512             nan     0.1000    0.0311
    40        0.5303             nan     0.1000    0.0108
    60        0.4044             nan     0.1000    0.0060
    80        0.3266             nan     0.1000    0.0042
   100        0.2683             nan     0.1000    0.0036
   120        0.2231             nan     0.1000    0.0016
   140        0.1911             nan     0.1000    0.0012
   150        0.1760             nan     0.1000    0.0008

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1289
     2        1.5228             nan     0.1000    0.0916
     3        1.4639             nan     0.1000    0.0676
     4        1.4195             nan     0.1000    0.0516
     5        1.3847             nan     0.1000    0.0533
     6        1.3509             nan     0.1000    0.0417
     7        1.3246             nan     0.1000    0.0395
     8        1.2996             nan     0.1000    0.0356
     9        1.2772             nan     0.1000    0.0325
    10        1.2562             nan     0.1000    0.0267
    20        1.0996             nan     0.1000    0.0176
    40        0.9285             nan     0.1000    0.0096
    60        0.8206             nan     0.1000    0.0064
    80        0.7413             nan     0.1000    0.0056
   100        0.6775             nan     0.1000    0.0049
   120        0.6262             nan     0.1000    0.0028
   140        0.5808             nan     0.1000    0.0024
   150        0.5617             nan     0.1000    0.0020

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1867
     2        1.4874             nan     0.1000    0.1253
     3        1.4056             nan     0.1000    0.1111
     4        1.3369             nan     0.1000    0.0876
     5        1.2814             nan     0.1000    0.0710
     6        1.2357             nan     0.1000    0.0691
     7        1.1925             nan     0.1000    0.0625
     8        1.1522             nan     0.1000    0.0485
     9        1.1210             nan     0.1000    0.0467
    10        1.0917             nan     0.1000    0.0493
    20        0.8833             nan     0.1000    0.0191
    40        0.6811             nan     0.1000    0.0111
    60        0.5521             nan     0.1000    0.0074
    80        0.4653             nan     0.1000    0.0034
   100        0.4003             nan     0.1000    0.0048
   120        0.3468             nan     0.1000    0.0019
   140        0.3066             nan     0.1000    0.0020
   150        0.2885             nan     0.1000    0.0020

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2336
     2        1.4612             nan     0.1000    0.1605
     3        1.3601             nan     0.1000    0.1235
     4        1.2815             nan     0.1000    0.1094
     5        1.2134             nan     0.1000    0.0937
     6        1.1540             nan     0.1000    0.0728
     7        1.1075             nan     0.1000    0.0771
     8        1.0594             nan     0.1000    0.0538
     9        1.0242             nan     0.1000    0.0623
    10        0.9850             nan     0.1000    0.0520
    20        0.7470             nan     0.1000    0.0237
    40        0.5217             nan     0.1000    0.0118
    60        0.4008             nan     0.1000    0.0065
    80        0.3177             nan     0.1000    0.0034
   100        0.2622             nan     0.1000    0.0038
   120        0.2199             nan     0.1000    0.0027
   140        0.1870             nan     0.1000    0.0013
   150        0.1741             nan     0.1000    0.0012

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1270
     2        1.5233             nan     0.1000    0.0889
     3        1.4653             nan     0.1000    0.0677
     4        1.4206             nan     0.1000    0.0516
     5        1.3853             nan     0.1000    0.0492
     6        1.3531             nan     0.1000    0.0395
     7        1.3277             nan     0.1000    0.0396
     8        1.3012             nan     0.1000    0.0368
     9        1.2783             nan     0.1000    0.0353
    10        1.2556             nan     0.1000    0.0282
    20        1.1030             nan     0.1000    0.0174
    40        0.9320             nan     0.1000    0.0087
    60        0.8232             nan     0.1000    0.0073
    80        0.7433             nan     0.1000    0.0051
   100        0.6797             nan     0.1000    0.0048
   120        0.6256             nan     0.1000    0.0034
   140        0.5818             nan     0.1000    0.0020
   150        0.5628             nan     0.1000    0.0018

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1885
     2        1.4870             nan     0.1000    0.1304
     3        1.4039             nan     0.1000    0.1030
     4        1.3365             nan     0.1000    0.0775
     5        1.2855             nan     0.1000    0.0733
     6        1.2384             nan     0.1000    0.0707
     7        1.1945             nan     0.1000    0.0532
     8        1.1595             nan     0.1000    0.0561
     9        1.1246             nan     0.1000    0.0456
    10        1.0947             nan     0.1000    0.0444
    20        0.8921             nan     0.1000    0.0252
    40        0.6776             nan     0.1000    0.0122
    60        0.5529             nan     0.1000    0.0094
    80        0.4653             nan     0.1000    0.0056
   100        0.3977             nan     0.1000    0.0039
   120        0.3480             nan     0.1000    0.0027
   140        0.3055             nan     0.1000    0.0023
   150        0.2875             nan     0.1000    0.0017

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2394
     2        1.4600             nan     0.1000    0.1618
     3        1.3581             nan     0.1000    0.1303
     4        1.2765             nan     0.1000    0.1044
     5        1.2105             nan     0.1000    0.0853
     6        1.1557             nan     0.1000    0.0735
     7        1.1101             nan     0.1000    0.0764
     8        1.0624             nan     0.1000    0.0568
     9        1.0265             nan     0.1000    0.0625
    10        0.9886             nan     0.1000    0.0521
    20        0.7467             nan     0.1000    0.0231
    40        0.5263             nan     0.1000    0.0111
    60        0.4026             nan     0.1000    0.0081
    80        0.3197             nan     0.1000    0.0039
   100        0.2625             nan     0.1000    0.0026
   120        0.2188             nan     0.1000    0.0025
   140        0.1862             nan     0.1000    0.0023
   150        0.1729             nan     0.1000    0.0008

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1293
     2        1.5241             nan     0.1000    0.0882
     3        1.4654             nan     0.1000    0.0644
     4        1.4232             nan     0.1000    0.0517
     5        1.3880             nan     0.1000    0.0515
     6        1.3557             nan     0.1000    0.0441
     7        1.3265             nan     0.1000    0.0365
     8        1.3030             nan     0.1000    0.0342
     9        1.2813             nan     0.1000    0.0292
    10        1.2620             nan     0.1000    0.0263
    20        1.1074             nan     0.1000    0.0183
    40        0.9345             nan     0.1000    0.0080
    60        0.8272             nan     0.1000    0.0061
    80        0.7455             nan     0.1000    0.0042
   100        0.6818             nan     0.1000    0.0029
   120        0.6298             nan     0.1000    0.0027
   140        0.5881             nan     0.1000    0.0028
   150        0.5674             nan     0.1000    0.0019

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1880
     2        1.4875             nan     0.1000    0.1285
     3        1.4058             nan     0.1000    0.1033
     4        1.3384             nan     0.1000    0.0796
     5        1.2868             nan     0.1000    0.0746
     6        1.2394             nan     0.1000    0.0625
     7        1.1996             nan     0.1000    0.0644
     8        1.1597             nan     0.1000    0.0480
     9        1.1292             nan     0.1000    0.0448
    10        1.1003             nan     0.1000    0.0467
    20        0.8926             nan     0.1000    0.0201
    40        0.6836             nan     0.1000    0.0092
    60        0.5624             nan     0.1000    0.0072
    80        0.4702             nan     0.1000    0.0053
   100        0.4023             nan     0.1000    0.0040
   120        0.3477             nan     0.1000    0.0022
   140        0.3070             nan     0.1000    0.0017
   150        0.2889             nan     0.1000    0.0015

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2275
     2        1.4639             nan     0.1000    0.1588
     3        1.3616             nan     0.1000    0.1325
     4        1.2780             nan     0.1000    0.1045
     5        1.2112             nan     0.1000    0.0817
     6        1.1593             nan     0.1000    0.0802
     7        1.1084             nan     0.1000    0.0690
     8        1.0653             nan     0.1000    0.0678
     9        1.0240             nan     0.1000    0.0561
    10        0.9895             nan     0.1000    0.0577
    20        0.7535             nan     0.1000    0.0286
    40        0.5246             nan     0.1000    0.0116
    60        0.4015             nan     0.1000    0.0048
    80        0.3215             nan     0.1000    0.0040
   100        0.2660             nan     0.1000    0.0033
   120        0.2221             nan     0.1000    0.0026
   140        0.1898             nan     0.1000    0.0017
   150        0.1763             nan     0.1000    0.0013

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1292
     2        1.5230             nan     0.1000    0.0931
     3        1.4619             nan     0.1000    0.0693
     4        1.4166             nan     0.1000    0.0549
     5        1.3808             nan     0.1000    0.0444
     6        1.3506             nan     0.1000    0.0432
     7        1.3220             nan     0.1000    0.0407
     8        1.2966             nan     0.1000    0.0329
     9        1.2755             nan     0.1000    0.0320
    10        1.2550             nan     0.1000    0.0299
    20        1.0987             nan     0.1000    0.0170
    40        0.9267             nan     0.1000    0.0081
    60        0.8214             nan     0.1000    0.0062
    80        0.7405             nan     0.1000    0.0050
   100        0.6776             nan     0.1000    0.0037
   120        0.6238             nan     0.1000    0.0022
   140        0.5819             nan     0.1000    0.0023
   150        0.5626             nan     0.1000    0.0019

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1862
     2        1.4878             nan     0.1000    0.1335
     3        1.4009             nan     0.1000    0.1043
     4        1.3335             nan     0.1000    0.0873
     5        1.2775             nan     0.1000    0.0751
     6        1.2299             nan     0.1000    0.0613
     7        1.1905             nan     0.1000    0.0642
     8        1.1515             nan     0.1000    0.0533
     9        1.1177             nan     0.1000    0.0408
    10        1.0917             nan     0.1000    0.0478
    20        0.8869             nan     0.1000    0.0203
    40        0.6703             nan     0.1000    0.0118
    60        0.5490             nan     0.1000    0.0063
    80        0.4634             nan     0.1000    0.0053
   100        0.4006             nan     0.1000    0.0037
   120        0.3491             nan     0.1000    0.0037
   140        0.3054             nan     0.1000    0.0014
   150        0.2877             nan     0.1000    0.0012

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2352
     2        1.4614             nan     0.1000    0.1628
     3        1.3577             nan     0.1000    0.1241
     4        1.2794             nan     0.1000    0.1109
     5        1.2086             nan     0.1000    0.0847
     6        1.1555             nan     0.1000    0.0784
     7        1.1056             nan     0.1000    0.0735
     8        1.0595             nan     0.1000    0.0613
     9        1.0194             nan     0.1000    0.0532
    10        0.9859             nan     0.1000    0.0540
    20        0.7517             nan     0.1000    0.0235
    40        0.5286             nan     0.1000    0.0106
    60        0.4018             nan     0.1000    0.0083
    80        0.3210             nan     0.1000    0.0037
   100        0.2647             nan     0.1000    0.0042
   120        0.2203             nan     0.1000    0.0026
   140        0.1871             nan     0.1000    0.0012
   150        0.1730             nan     0.1000    0.0013

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1289
     2        1.5230             nan     0.1000    0.0881
     3        1.4637             nan     0.1000    0.0665
     4        1.4202             nan     0.1000    0.0523
     5        1.3845             nan     0.1000    0.0444
     6        1.3551             nan     0.1000    0.0433
     7        1.3265             nan     0.1000    0.0385
     8        1.3018             nan     0.1000    0.0337
     9        1.2792             nan     0.1000    0.0311
    10        1.2573             nan     0.1000    0.0329
    20        1.1043             nan     0.1000    0.0172
    40        0.9327             nan     0.1000    0.0096
    60        0.8252             nan     0.1000    0.0062
    80        0.7451             nan     0.1000    0.0052
   100        0.6820             nan     0.1000    0.0035
   120        0.6297             nan     0.1000    0.0023
   140        0.5876             nan     0.1000    0.0025
   150        0.5673             nan     0.1000    0.0032

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1843
     2        1.4901             nan     0.1000    0.1276
     3        1.4076             nan     0.1000    0.1065
     4        1.3407             nan     0.1000    0.0856
     5        1.2862             nan     0.1000    0.0840
     6        1.2340             nan     0.1000    0.0660
     7        1.1923             nan     0.1000    0.0558
     8        1.1574             nan     0.1000    0.0443
     9        1.1280             nan     0.1000    0.0448
    10        1.0999             nan     0.1000    0.0505
    20        0.8894             nan     0.1000    0.0207
    40        0.6783             nan     0.1000    0.0111
    60        0.5605             nan     0.1000    0.0066
    80        0.4734             nan     0.1000    0.0039
   100        0.4075             nan     0.1000    0.0039
   120        0.3512             nan     0.1000    0.0021
   140        0.3092             nan     0.1000    0.0028
   150        0.2901             nan     0.1000    0.0024

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2306
     2        1.4624             nan     0.1000    0.1618
     3        1.3587             nan     0.1000    0.1233
     4        1.2816             nan     0.1000    0.1166
     5        1.2099             nan     0.1000    0.0853
     6        1.1562             nan     0.1000    0.0770
     7        1.1067             nan     0.1000    0.0785
     8        1.0575             nan     0.1000    0.0566
     9        1.0206             nan     0.1000    0.0522
    10        0.9860             nan     0.1000    0.0564
    20        0.7505             nan     0.1000    0.0249
    40        0.5263             nan     0.1000    0.0130
    60        0.4083             nan     0.1000    0.0062
    80        0.3265             nan     0.1000    0.0055
   100        0.2651             nan     0.1000    0.0016
   120        0.2235             nan     0.1000    0.0020
   140        0.1914             nan     0.1000    0.0016
   150        0.1778             nan     0.1000    0.0012

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1306
     2        1.5227             nan     0.1000    0.0891
     3        1.4647             nan     0.1000    0.0660
     4        1.4200             nan     0.1000    0.0549
     5        1.3834             nan     0.1000    0.0511
     6        1.3500             nan     0.1000    0.0443
     7        1.3219             nan     0.1000    0.0352
     8        1.2993             nan     0.1000    0.0350
     9        1.2766             nan     0.1000    0.0305
    10        1.2559             nan     0.1000    0.0274
    20        1.1052             nan     0.1000    0.0179
    40        0.9298             nan     0.1000    0.0083
    60        0.8229             nan     0.1000    0.0063
    80        0.7425             nan     0.1000    0.0046
   100        0.6792             nan     0.1000    0.0036
   120        0.6279             nan     0.1000    0.0046
   140        0.5846             nan     0.1000    0.0023
   150        0.5642             nan     0.1000    0.0026

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1855
     2        1.4884             nan     0.1000    0.1285
     3        1.4065             nan     0.1000    0.1032
     4        1.3415             nan     0.1000    0.0862
     5        1.2853             nan     0.1000    0.0689
     6        1.2404             nan     0.1000    0.0770
     7        1.1940             nan     0.1000    0.0608
     8        1.1562             nan     0.1000    0.0523
     9        1.1235             nan     0.1000    0.0413
    10        1.0962             nan     0.1000    0.0474
    20        0.8943             nan     0.1000    0.0275
    40        0.6769             nan     0.1000    0.0097
    60        0.5532             nan     0.1000    0.0090
    80        0.4636             nan     0.1000    0.0056
   100        0.3971             nan     0.1000    0.0028
   120        0.3468             nan     0.1000    0.0029
   140        0.3050             nan     0.1000    0.0041
   150        0.2874             nan     0.1000    0.0017

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2339
     2        1.4634             nan     0.1000    0.1611
     3        1.3629             nan     0.1000    0.1298
     4        1.2801             nan     0.1000    0.1012
     5        1.2164             nan     0.1000    0.0897
     6        1.1591             nan     0.1000    0.0842
     7        1.1047             nan     0.1000    0.0714
     8        1.0601             nan     0.1000    0.0628
     9        1.0202             nan     0.1000    0.0622
    10        0.9821             nan     0.1000    0.0482
    20        0.7587             nan     0.1000    0.0257
    40        0.5325             nan     0.1000    0.0149
    60        0.4087             nan     0.1000    0.0059
    80        0.3258             nan     0.1000    0.0033
   100        0.2672             nan     0.1000    0.0033
   120        0.2224             nan     0.1000    0.0022
   140        0.1884             nan     0.1000    0.0021
   150        0.1759             nan     0.1000    0.0011

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2327
     2        1.4613             nan     0.1000    0.1593
     3        1.3616             nan     0.1000    0.1250
     4        1.2826             nan     0.1000    0.1176
     5        1.2098             nan     0.1000    0.0864
     6        1.1544             nan     0.1000    0.0765
     7        1.1063             nan     0.1000    0.0675
     8        1.0625             nan     0.1000    0.0654
     9        1.0226             nan     0.1000    0.0635
    10        0.9841             nan     0.1000    0.0475
    20        0.7568             nan     0.1000    0.0230
    40        0.5266             nan     0.1000    0.0100
    60        0.4020             nan     0.1000    0.0071
    80        0.3295             nan     0.1000    0.0037
   100        0.2701             nan     0.1000    0.0050
   120        0.2253             nan     0.1000    0.0018
   140        0.1921             nan     0.1000    0.0012
   150        0.1784             nan     0.1000    0.0018
```


Boosting (confusion matrix table)
========================================================

```
          Reference
Prediction    A    B    C    D    E
         A 1658    9    3    3    1
         B   38 1076   25    0    0
         C    0   24  984   15    3
         D    1    8   29  925    1
         E    0   12    3   13 1054
```


Random Forests
========================================================
Random forests are an ensemble learning method for classification and regression tasks, that operate by constructing a multitude of decision trees at training time through bootstrap variables, and aggreagating the output based on voting or averaging. Random forests correct for decision trees' habit of overfitting to their training set.

Random Forests (code)
========================================================

```r
set.seed(3311995)
ctrl <- trainControl(method="cv", number = 10, repeats = 4)
modelFit <- train(training$classe ~., data=training[, -55], method="rf", trControl = ctrl, ntree=50)
pred <- predict(modelFit, newdata=testing)
```

Random Forests (confusion matrix table)
========================================================

```
          Reference
Prediction    A    B    C    D    E
         A 1673    1    0    0    0
         B    4 1132    3    0    0
         C    0    2 1020    4    0
         D    0    1    2  961    0
         E    0    1    1    0 1080
```

Model Selection
========================================================
Model selection was done by determining which model provided the best accuracy on the testing set.

|  __Model__    | __Accuracy__  | __Upper__  | __Lower__  |
| --------------| ------------- | ---------- | ---------- |
| Linear        | 0.3916737 | 0.404281 | 0.3791739 |
| Boosting      | 0.9680544 | 0.972399 | 0.9632381 |
| Random Forest | 0.9967715 | 0.9980551 | 0.9949628 |



Conclusions
========================================================
The Random Forest Model was found to have the highest accuracy on the training data and was the model used to predict the values on the test set. The Boosting algorithm was also quite good but took a very long time to run. The Linear Model did not provide good accuracy for this case.

