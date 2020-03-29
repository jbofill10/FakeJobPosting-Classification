# FakeJobPosting-Classification
Creating different types of models to determine whether a job posting is fake or not

# Logistic Regression Results
![alt text](https://github.com/jbofill10/FakeJobPosting-Classification/blob/master/model_results/Results.png)

This score is really great... too great. I was wondering whether there was possible data leakage. Overall, I filled one column containing 8000 NaNs, another one containing 7000 NaNs, and a third one containing 6000 NaNs, fourth containing 5000 NaNs, and lastly a fifth containing 3,400 NaNs. The data set is of size ~110,000 so I wonder if the data leakaged occured there.


# KNN Classification Results
![alt text](https://github.com/jbofill10/FakeJobPosting-Classification/blob/master/model_results/KNN_results.png)

Pretty good overall. I used GridSearch and Cross Validation to ensure my model was accurate. Using grid search to optimize K, I found that the best value for K is from 2-9. K=4 scored the highest with 96.31%.


# Comparing the models

Logistic Regression beat KNN by 0.08% without any other metric testing (KNN went through cross validation kfold = 5 and grid search of 1,30 neighbors (I got a snack and showered during that))
