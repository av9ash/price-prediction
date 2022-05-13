import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier


class Model:
    def __init__(self, x_train, y_train, x_valid, y_valid):
        self.rfc = RandomForestClassifier(criterion="gini", n_jobs=-1, random_state=9)
        self.knn = KNeighborsClassifier(n_neighbors=3, algorithm="auto", n_jobs=-1)
        self.svm_clf = svm.SVC(break_ties=True, kernel="linear")
        self.bc = BaggingClassifier(self.svm_clf, max_samples=0.5, n_estimators=100)
        self.X_train = x_train
        self.y_train = y_train
        self.X_valid = x_valid
        self.y_valid = y_valid

    def train_model(self, clf):
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print('{} Classifier Accuracy Score: '.format(clf.__class__.__name__), accuracy_score(self.y_valid, y_pred))
        cm = print_confusion_matrix(self.y_valid, y_pred, clf.__class__.__name__)
        return cm


def show_price_range_distribution(train_data):
    sns.set()
    price_plot = train_data['price_range'].value_counts().plot(kind='bar')
    plt.xlabel('price_range')
    plt.ylabel('Count')
    plt.show()


def show_battery_capacity_distribution(train_data):
    sns.set(rc={'figure.figsize': (5, 5)})
    ax = sns.displot(data=train_data["battery_power"])
    plt.show()


def show_mobile_depth_dstribution(train_data):
    sns.set(rc={'figure.figsize': (5, 5)})
    ax = sns.displot(data=train_data["m_dep"])
    plt.show()


def show_missing_values(X):
    # missing values
    X.isna().any()


def get_X_y(train_data):
    X = train_data.drop(['price_range'], axis=1)
    y = train_data['price_range']
    return X, y


def print_confusion_matrix(y_test, y_pred, plt_title):
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='BuPu')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(plt_title)
    plt.show()
    return cm


def load_data(file_name):
    data_raw = pd.read_csv(file_name)
    print(data_raw.head())
    print(data_raw.info())
    # Remove any bad samples where screen size is 0
    data = data_raw[data_raw['sc_w'] != 0]
    print(data.shape)
    return data


def explore_data(train_data, X):
    show_price_range_distribution(train_data)
    show_battery_capacity_distribution(train_data)
    show_mobile_depth_dstribution(train_data)
    show_missing_values(X)


def predict_labels(file_name, trained_clf):
    test_data = load_data(file_name)
    X_test = test_data.drop(['id'], axis=1)

    # missing values
    # show_missing_values(X_test)
    y_pred_svm = trained_clf.predict(X_test)
    print(list(y_pred_svm))


def main():
    train_data = load_data('cell_price_data/train.csv')
    X, y = get_X_y(train_data)
    # explore_data(train_data, X)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)

    model = Model(X_train, y_train, X_valid, y_valid)
    model.train_model(clf=model.rfc)
    model.train_model(clf=model.knn)
    model.train_model(clf=model.svm_clf)
    model.train_model(clf=model.bc)

    # predict_labels('cell_price_data/test.csv', model.rfc)
    # predict_labels('cell_price_data/test.csv', model.knn)
    # predict_labels('cell_price_data/test.csv', model.svm_clf)


if __name__ == '__main__':
    main()
