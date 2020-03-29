# FakeJobPosting-Classification
Creating different types of models to determine whether a job posting is fake or not

# Logistic Regression Results
![alt text](https://github.com/jbofill10/FakeJobPosting-Classification/blob/master/model_results/Results.png)

Not too bad, but not good either. The score fluctuates in the 80s mostly, sometimes dropping to 78%. Still need to add metric testing to this.

# KNN Classification Results
![alt text](https://github.com/jbofill10/FakeJobPosting-Classification/blob/master/model_results/KNN_results.png)

Not too great. Showing little to no improvement implementing grid search either. I tried not using PCA and the score didn't change much.

# Comparing the models

Logistic Regression beat KNN by 4-5% without any other metric testing (KNN went through cross validation kfold = 5 and grid search of 1,30 neighbors
