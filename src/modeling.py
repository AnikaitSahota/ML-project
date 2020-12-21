from preprocessing import preprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import joblib


class tree_classifiers():
    """class to classifiy the data using tree based models (i.e. DTree, XGradient Boost and Naive bayes classifier)
    """
    def __init__(self , X, y) :
        """constructor for class to intialise the training and testing dataset

        Parameters
        ----------
        X : numpy 2D array
            features of datapoints
        y : numpy 1D array
            labels of datapoints
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size = 0.3 , stratify = y, random_state = 1)

    def test(self, load_from_file = True) :
        """build the models specified in class and test them

        Parameters
        ----------
        load_from_file : bool, optional
            flag to instruct about usage of pickle file, by default True
        """
        saved_model_names = ['DTree' , 'XGradientBoost' , 'GaussianNaiveBayes']
        classifiers = [DecisionTreeClassifier(random_state=22, max_depth = 90), XGBClassifier(random_state=22,learning_rate=0.9, n_jobs=5), GaussianNB()]

        for i in range(len(classifiers)) :
            if(load_from_file) :            # loading the model from pickle file
                clf = joblib.load('models/saved_model_' + saved_model_names[i])
            else :                          # computing the model by fitting it
                clf = classifiers[i].fit(self.X_train, self.y_train)
                joblib.dump(clf , 'models/saved_model_'+saved_model_names[i])           # saving the model

            print('------------' , saved_model_names[i] , '------------')
            y_pred = clf.predict(self.X_test)                                           # prediction

            print(classification_report(self.y_test , y_pred))                          # report metric of model over test dataset


class logistic_classifier():
    """class to classifiy the data using logistic classifier
    """
    def __init__(self , X, y) :
        """constructor for class to intialise the training and testing dataset

        Parameters
        ----------
        X : numpy 2D array
            features of datapoints
        y : numpy 1D array
            labels of datapoints
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size = 0.3 , stratify = y, random_state = 1)

    def test(self, load_from_file = True) :
        """build the models specified in class and test them

        Parameters
        ----------
        load_from_file : bool, optional
            flag to instruct about usage of pickle file, by default True
        """

        saved_model_names = ['LogisticRegression']
        classifiers = [LogisticRegression(C = 0.57, random_state=22, n_jobs= 5)]

        for i in range(len(classifiers)) :
            if(load_from_file) :            # loading the model from pickle file
                clf = joblib.load('models/saved_model_' + saved_model_names[i])
            else :                          # computing the model by fitting it
                clf = classifiers[i].fit(self.X_train, self.y_train)
                joblib.dump(clf , 'models/saved_model_'+saved_model_names[i])           # saving the model

            print('------------' , saved_model_names[i] , '------------')
            y_pred = clf.predict(self.X_test)                                           # prediction

            print(classification_report(self.y_test , y_pred))                          # report metric of model over test dataset

