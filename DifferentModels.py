from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from  imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np


def compute(df):
    # Only 300 locations, plus really messy. Might find a way to clean it up and add in later
    df = df.drop(
        ['job_id', 'title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements',
         'benefits'], axis=1).sort_index()

    y = df['fraudulent']

    X = df.drop(['fraudulent'], axis=1)

    numerical_features = ['telecommuting', 'has_company_logo', 'has_questions']
    label_features = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']

    for feature in label_features:
        X[feature].replace(np.nan, X[feature].mode()[0], regex=True, inplace=True)

    '''
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])

    categorical_transformer = Pipeline(steps=[
        ('cat_imputer', OneHotEncoder())])

    preprocessing = ColumnTransformer(transformers=[
        ('numerical', numeric_transformer, numerical_features),
        ('categorical', categorical_transformer, label_features)
    ])

    log_reg = Pipeline(steps=[
        ('preprocessing', preprocessing),
        ('scaler', StandardScaler(with_mean=False)),
        ('log', LogisticRegression())
    ])
    '''

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    imputer.fit_transform(X[numerical_features])

    c_t = make_column_transformer((OneHotEncoder(), label_features), remainder='passthrough')

    scalar = StandardScaler()

    big_X = c_t.fit_transform(X).toarray()

    big_X = scalar.fit_transform(big_X)

    pca = PCA()

    rus = RandomUnderSampler()

    reduced_x = pca.fit_transform(big_X)

    undersampled_x, y= rus.fit_resample(reduced_x, y)

    x_train, x_test, y_train, y_test = train_test_split(undersampled_x, y, test_size=0.2, random_state=0)

    #logistic_regression(x_train,y_train,x_test,y_test)
    print()
    #kNN(undersampled_x, y)
    print()
    RandomForest(undersampled_x,y)


def kNN(x, y):
    print('KNN Results:')
    knn = KNeighborsClassifier()

    grid = GridSearchCV(knn, param_grid={'n_neighbors':range(1,31)}, scoring='accuracy')

    grid.fit(x,y)

    for i in range(0, len(grid.cv_results_['mean_test_score'])):
        print('N_Neighbors {}: {} '.format(i+1, grid.cv_results_['mean_test_score'][i]))


def logistic_regression(x_train, y_train, x_test, y_test):
    print("Logistic Regression results:")
    log_reg = LogisticRegression()

    log_reg.fit(x_train, y_train)

    y_pred = log_reg.predict(x_test)

    print(confusion_matrix(y_test, y_pred))
    print(f'Prediction score: {log_reg.score(x_test, y_test) * 100:.2f}%')
    print(f'MAE from Logistic Regression: {mean_absolute_error(y_test, y_pred) * 100:.2f}%')


def RandomForest(x,y):
    print("Random Forest results:")
    rf = RandomForestClassifier(bootstrap=True)

    rf.fit(x,y)
    # Cross validation of 5
    score = cross_val_score(rf, x, y)

    print(f'Prediction score: {np.mean(score) * 100:.2f}%')


