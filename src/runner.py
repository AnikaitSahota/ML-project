import preprocessing as preprocess
import modeling

if __name__ == "__main__":
    # modeling.tree_classifiers().fun()
    pre_proc = preprocess.preprocessor(from_file = True)
    X , y = pre_proc.preprocess()

    preprocess.EDA().scatter_plot(X , y)
    # print(pre_proc)
