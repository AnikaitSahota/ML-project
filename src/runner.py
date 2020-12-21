import preprocessing as preprocess
import modeling

if __name__ == "__main__":
    pre_proc = preprocess.preprocessor(from_file = True)
    X , y = pre_proc.preprocess()

    # preprocess.EDA().scatter_plot(X , y)
    modeling.tree_classifiers(X,y).test()
    modeling.logistic_classifier(X,y).test()
    # print(pre_proc)
