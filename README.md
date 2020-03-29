# FakeJobPosting-Classification
Creating different types of models to determine whether a job posting is fake or not

# Logistic Regression Results
![alt text](https://github.com/jbofill10/FakeJobPosting-Classification/blob/master/model_results/Results.png)

Not too bad, but not good either. The score fluctuates in the 80s mostly, sometimes dropping to 78%. Still need to add metric testing to this.

# KNN Classification Results
![alt text](https://github.com/jbofill10/FakeJobPosting-Classification/blob/master/model_results/KNN_results.png)

Not too great. Showing little to no improvement implementing grid search either. I tried not using PCA and the score didn't change much.
 
# Random Forests Classification
![alt text](https://github.com/jbofill10/FakeJobPosting-Classification/blob/master/model_results/RandomForests.png)  
So far this is the best score of my model selection process. I suspect this is due to the randomization of the training set allows my model to generalize a bit more than the other models. I originally got 92% and was really excited, but I suspected overfitting despite having balanced the data set based on the target label. After applying 5 folds to my training data, my score shot down to 82%.
 
# Comparing the models

Logistic Regression beat KNN by 4-5% without any other metric testing (KNN went through cross validation kfold = 5 and grid search of 1,30 neighbors.

As I stated before, Random Forests so far has done the best of out the 3 models. I am actually happy with this score because it is a consistent 80+%. Every time I run the different models, scores variate between 76%-83%. The Random Forests model never drops below 80%.
