# Python-Course-IBM
Python course from IBM covering topics like:
    data pre-processing, 
    data formatting,  
    data normalization
    bining (Group values into 'bins'),
    regression analysis, 
    model fitting, etc
    
    
## EXPLORATORY DATA ANALYSIS
  Preliminary step in data analysis to:
  * Summarize main xtics of the data
  * Gain better understanding of the dataset
  * Uncover relationships btw variables
  * Extract important variables
  
  
## The main question we are trying to answer is: 
"What are the xtics which have the most impact on the car price"


## Visualizing various distribution of data 
Box plots are great a way to visualize numeric data

the main features that the box plot shows are:

* median of the data, which represents where the middle data point is,
* the upper quartile shows where the 75th percentile is
* the lower quartile shows the where the 25th percentile is
* the data between the upper and lower quartile represents the interquartile range
* next you have the lower and upper extremes
* outliers

## Correlation 
Correlation is a statistical metric for measurung to what
extent different variables are interdependent.

In other words, when we look at two variables over time, 
if one variable changes how does this change affect change in the other variable

## Correlation-- Statistic

Pearson Correlation:
Measures the strength of the correlation btw features.

Correlation coefficient
P-value
Correlation coefficient

Close to +1: Large Positive relationship
Close to -1: Large Negative relationship
Close to 0: No relationship
P-value
The p-value will tell us how certain we are about the correlation that we calculated

P-value less than 0.001: Strong certainty in the result
P-value less than 0.05: Moderate certainty in the result
P-value less than 0.1: Weak certainty in the result
P-value greater than 0.1: No certainty in the result
Strong correlation
Correlation coefficient close to 1 or -1
P-value less than 0.001


## Model Development
In this module we'll learn about:

Simple and Multiple Linear Regression
Model Evaluation using Visualization
Polynomial Regression and Pipelines
R-squared and MSE for In-Sample Evaluation
Prediction and Dcesion Making


## Linear Regression and Multiple Linear Regression:
Introduction 
Linear regression will refer to one independent variable to make a prediction
Multiple Linear Regression will refer to multiple independent variables to make a prediction
Simple Linear Regression {S.L.R} 
The predictor (independent) variable- x
The target (dependent) variable- y
We would like to come up with a linear relationship such that: y = b0 + b1.x

b0: the intercept
b1: the slope

## Multiple Linear Regression (MLR) 
This method is used to explain the relationship between:

One continous target (Y) variable
Two or more predictor (X) variables
If we have four (4) predictor variables, 
our linear relationship will look like this: Y = b0 + b1X1 + b2X2 + b3X3 + b4X4

b0: intercept (X0)
b1: the coefficient of parameter X1
b2: the coefficient of parameter X2 and so on...
Considering that there are only two predictor variables, 
our linear regression could be: Y = 1 + 2X1 + 3X2



## Model Evaluation Using Visualization 
Regression Plot 
Why use regression plot

It gives us a good estimate of:

The relationship btw two variables
The strength of the correlation
The direction of the relationship (positive or negative)
Regression Plot shows us a combination of:

The scatterplot: where each point represents a different y
The fitted linear regression line

## Residual Plot 
The residual plot represents the error between the actual value. 
By examing the predicted value and actual value we see a differnce. 
We obtain that value by subtracting the predicted value from the actual value. 
We the plot that value on the vertical axis with an independent variable as the horizontal axis. 
Similarly, for the second sample we repeat the process. 
Looking at the new plot gives some insight into our data. We expect to see the results 
have zero mean, distributed evenly along the x axis with similar variance, 
there is no curvature. This type of residual plot suggests a linear plot is appropriate.

## Distribution Plot 
A distribution plot counts the predicted value versus the actual value. 
These plot are extremely useful in visualizing models with more than one independent variables


## Polynomial Regression and Pipelines 
Polynomial Regressions 
A special case of the general linear regression model
Useful for describing curvilinear relationships
Curvilinear relationships:
By squaring or setting higher-order terms of the predictor variables

The model can be:

Quadratic - 2nd order: Y = b0 + b1X1 + b2(X2)^2
Cubic - 3rd order: Y = b0 + b1X1 + b2(X2)^2 + b3(X3)^3

## Pre-processing 
As the dimension of the data becomes bigger, 
we may want to normalize multiple features in scikit learn. 
Instead we can use the pre-processing module to simplify many tasks.

For example we can standardize each feature simultaneously

## Pipelines 
We can simplify our code by using the pipelines library

There are many steps to getting a prediction:
Normalization
Polynomial transform
Linear Regression
We a simplify the process using a pipeline.
Pipelines sequencially perform a series of transformations and 
the last step carries out a prediction


## Measure for In-Sample Evaluation 
These measures are a way to numerically determine how good the model fits on dataset

Two important measures to determine the fit of a model
Mean Squared Error (MSE)
R-Squared (R^2)


### Mean Squared Error (MSE) 
To measure the mse we take the difference between the actual value and
the predicted value and square it

We do this for a sample we select. After which we will sum all the
squared differences and then divide by the size of the sample

To find the MSE in python, we import mean_squared_error library from sklearn.metrics

The MSE function takes in two parameters; 
1. The actual value of the target variable and 
2. the predicted value of the target variable


## R-Squared (R^2) 
It is also called the Coefficient of Determination
It is a measure to determine how close the data is to the fitted regression line
R^2: is the percentage of variation of the target variable (Y) that is 
explained by the linear model
Think about it as comparing a regression model to a simple model i.e 
the mean of the data points
R^2 = (1 - [MSE of regression line/MSE of yhat])


## Prediction and Decision Making 
Decision Making

To determine the best fit, we look at a combination of:

Do the predicted values make sense
Visualization
Numerical measures for evaluation
Comparing between different models

## Visualization 
We can observe if our model is a best fit by viaualizing the data

Regression plot
Residual plot
Distribution plot
Numerical measures for Evaluation 
MSE- Mean Squared Error
R-Squared (R^2)
Comparing MLR and SLR 
The MSE for a Multiple Linear Regression Model will be smaller than the MSE for a 
Simple Linear Regression Model,  since the errors of the data will decrease when more 
variables are included in the model.
Polynomial regression will also have a smaller MSE than the linear regular regression.

## Model Evaluation 
In-sample evaluation tells us how well our model will fit the data used to train it
Problem?
It does not tell us how well the trained model can be used to predict new data
Solution?
The solution is to spit our data up:
We use the In-sample data or training data to train the model
Out-of-sample evaluation data or test data to test the model

## Training/Testing Sets 
Seperating our data into training and testing set is an important part of Model evaluation. 
We use the test data to get an idea of how our model will perform in the real world.
Usually when we split a dataset, the larger portion of data is used for training and 
the smaller portion of data is used for testing. 
For example, we can use 70% of the data for training and the remaining 20% for testing.
We use the training set to build a model and discover predictive relationship
We the use a testing set to evaluate the performance of the predictive model.
When we have completed our model, we should use all the data to train the model
to get the best performance.

 ## Function train_test_split() 
A popular function of the scikit learn library can help us to split data
into random train and test subsets  from sklearn.model_selection import train_test_split 


## Generalization Performance 
Generilization error is a measure of how well our data does at predicting previously unseen data
The error we obtaing using our testing data is an approximation of this eror
Using lots of training data will give us an accurate means to determine how well our 
model will perform in the real world

If we run our model randomly over different randomly generated train and test data, 
we will notice that the generalization error of each trial are different but all close 
to a general error. The best way to tackle this problem is by using  Cross Validation 

Cross Validation 
Most common out-of-sample evaluation metrics
More effective use of data (each observation is used for both training and testing)
In this method, the dataset is split into k equal groups, each group is referred to as a fold. 
e.g 4 folds. Some of the folds can be used as training set which we use to train the model.
And the remaining parts are used as a test set which we use to test the model. 
In this example, we can use one fold for testing and three folds for training. 
This is repeated until each partition is used for both training and testing. 
At the end, we use the average result as the estimate of our out-of-sample error. 
The Evaluation metric depends on the model, for example the R^2

The simplest way to apply Cross Validation is to call the  cross_val_score() . This method is imported from scikit learn library

 from sklearn.model_selection import cross_val_score 

## Overfitting, Underfitting and Model Selection 

Underfitting: The model is too simple to fit the data

Overfitting: Here the model is to flexible that it fits the noice but those not fit the data.

The training error decreases with the order of the polynomial

The test MSE is a better means of estimating the error of a polynomial. 
We select the order that minimizes the test error. 
Anything on the left of that order is consider underfitting and anything on the right is 
considered overfitting.

## Ridge Regression 
RR prevents overfitting

When dealing with high order polynomials, we see that the estimated polynomial 
coefficients have a very high magnitude. Ridge Regression controls the 
magnitude of these polynomial coefficients by introducing the parameter alpha 

Alpha is a parameter we set before fitting of training the model. 
Different values of alpha changes the model

In order to select alpha we use cross validation. To make a prediction using Ridge Regression:


## Grid Search 
Grid search allows us to scan through multiple free parameters with few lines of code

## Hyperparameters 
Parameters like the alpha from the Ridge Regreeion topic is a Hyperparameter and 
they are not part of the fitting or training pocess.
SCikit learn has a means of automatically iterating over these hyperparameters using 
cross-validation called  GRID SEARCH 
Grid search takes the model or objects you would like to train and different values 
of the hyperparameters. It then calculates the Mean Square Error or R^2 for various 
hyperparameter values. Allowing you to choose the best values.

After obtaining different errors from the different hyperparameter values. We select the hyperparameter that minimizes the MSE or that maximizes the R^2

#Process 
To select the righte values for the hyperparameters, we split our dataset into three parts;

Training set: We train the model for different parameters.
Validation set: We select the hyperparameter that minimizes the MSE or maxamizes the R^2 on the validation set.
Test set: We finally test our model performance using the test data.
In the module, we will focus on the hyperparameters alpha and the normalization parameter.

The grid search takes on the scoring method(in this case R^2), number of folds, the model or object, and the free parameter value.
