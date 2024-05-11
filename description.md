# Description of submission

1. data cleaning:
drop columns having more than 30% missing values
drop columns having only one unique value
drop columns in date format
keep rows where "outcome_available' is 1
drop string columns
impute missing values with median/mean, median performs better
call SelectKBest in sklearn.feature_selection to find top 10 features in dataframe

2. modeling:
use cleaned data to train different models, including logistic regression, decision trees, random forests, gradient boosting, SVC, and linear discriminant
calculate and compare accuracy, precision, recall, f1-score of all models
select the best performed model