from preprocessing import preprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


class tree_classifiers():
    def __init__(self) :
        X, y = preprocessor().preprocess()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size = 0.3)


    def fit_predict(self, clf) :
        clf = clf.fit(self.X_train , self.y_train)
        y_pred = clf.predict(self.X_test)

        print(classification_report(self.y_test , y_pred))

    def fun(self) :
        classifiers = [LogisticRegression(random_state=22, n_jobs= 4), XGBClassifier(random_state=22,learning_rate=0.9, n_jobs=4), GaussianNB()]
        for clf in classifiers :
            print('------------'*4)
            self.fit_predict(clf)

tree_classifiers().fun()