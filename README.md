# Premier League Machine Learning

## Overview
An end-to-end machine learning project analysing 6,840 English Premier League match records across 18 seasons using R. The project explored match patterns through unsupervised learning and evaluated predictive models for both regression and classification tasks.

## Dataset
The dataset contains historical English Premier League match records spanning 18 seasons. Match-level variables were used to investigate patterns in match outcomes and develop predictive models.

## Objectives
Identify distinct match patterns and outcome profiles using unsupervised learning.
Predict goal-related outcomes using regression models.
Predict match outcome categories using classification models.
Compare model performance and determine which approaches generalise most effectively to unseen data.

## Methods
### Unsupervised Learning

Principal Component Analysis (PCA),
K-means clustering,
Hierarchical clustering for cluster validation

### Regression

Ordinary Least Squares (OLS),
Ridge Regression,
Lasso Regression,
Random Forest,
5-fold cross-validation,
Temporal train-test split to reduce data leakage

### Classification

Logistic Regression,
Logistic Ridge,
K-Nearest Neighbours (KNN),
Random Forest,
ROC/AUC analysis

## Results
PCA reduced the feature space to two principal components explaining 55.7% of the variance, while K-means identified three interpretable match archetypes with different outcome profiles.

For regression, OLS and Ridge Regression achieved the best test RMSE of 1.229 goals, outperforming the null baseline by 5.2%.

For classification, Logistic Ridge achieved the highest test AUC of 0.695 and accuracy of 64.9%, outperforming the tree-based and distance-based models evaluated.

## Project Outcomes
The analysis found that relatively simple linear models consistently matched or outperformed more complex ensemble approaches across the supervised learning tasks. This suggests that the relationships captured by the selected features were predominantly linear and that greater model complexity did not necessarily translate into better predictive performance.

## Future Improvements
Potential extensions include incorporating additional match-level and team-level features, testing alternative feature engineering approaches, and evaluating models using more advanced time-series validation strategies.
