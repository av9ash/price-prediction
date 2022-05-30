import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


# Read the data
def read_data():
    X_train = pd.read_csv('data/home-data-for-ml-course/train.csv', index_col='Id')
    X_test = pd.read_csv('data/home-data-for-ml-course/test.csv', index_col='Id')
    return X_train, X_test


def get_X_y(data):
    # Remove rows with missing target, separate target from predictors
    if 'SalePrice' in data.columns:
        data.dropna(axis=0, subset=['SalePrice'], inplace=True)
        labels = data.SalePrice
        data.drop(['SalePrice'], axis=1, inplace=True)
    else:
        labels = []
    return data, labels


def get_validation_set(X_train, y):
    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, train_size=0.8, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid


def get_low_cardinality_categorical_cols(X_train):
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train.columns if
                        X_train[cname].nunique() < 10 and
                        X_train[cname].dtype == "object"]

    return categorical_cols


def get_numerical_columns(X_train):
    # Select numerical columns
    numerical_cols = [cname for cname in X_train.columns if
                      X_train[cname].dtype in ['int64', 'float64']]

    return numerical_cols


def get_score(X, y, n_cv, model_pipeline):
    """
    Return the average MAE over n CV folds of random forest model.
    :param X: X
    :param y: y
    :param n_cv: int number of folds
    :param n_estimators:
    :return: float
    """
    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(model_pipeline, X, y, cv=n_cv, scoring='neg_mean_absolute_error')
    return scores.mean()


def main():
    create_csv = True
    # Load data
    X_train, test_data = read_data()
    X_train, y_train = get_X_y(X_train)

    # Validation code
    # Get validation set
    X_train, X_valid, y_train, y_valid = get_validation_set(X_train, y_train)

    # Get columns to consider
    categorical_cols = get_low_cardinality_categorical_cols(X_train)
    numerical_cols = get_numerical_columns(X_train)

    # Keep selected columns only
    considered_cols = categorical_cols + numerical_cols
    X_train = X_train[considered_cols].copy()
    # Validation code
    X_valid = X_valid[considered_cols].copy()
    X_test = test_data[considered_cols].copy()

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='mean')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define model
    # model = RandomForestRegressor(n_estimators=100, random_state=0)

    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)

    clf = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=5, n_jobs=4)
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    print(clf.best_iteration)
    # Bundle preprocessing and modeling code in a pipeline
    # model = Pipeline(steps=[('preprocessor', preprocessor),
    #                       ('model', model)
    #                       ])

    # Validation code
    # Preprocessing of validation data, get predictions
    # preds = clf.predict(X_valid)
    # print('MAE:', mean_absolute_error(y_valid, preds))

    # Cross Validation code
    # avg_mae = get_score(X_train, y_train, n_cv=5, model_pipeline=clf)
    # print('Average MAE score: ', avg_mae)

    if create_csv:
        # Preprocessing of training data, fit model
        # clf.fit(X_train, y_train)
        # Preprocessing of testing data, get predictions
        preds_test = clf.predict(X_test)
        # Save test predictions to file
        output = pd.DataFrame({'Id': test_data.index,
                               'SalePrice': preds_test})
        output.to_csv('submission.csv', index=False)

    print('Done')


if __name__ == '__main__':
    main()
