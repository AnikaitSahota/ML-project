import preprocessing as preprocess
import modeling

if __name__ == "__main__":
    pre_proc = preprocess.preprocessor(from_file = True)
    X , y = pre_proc.preprocess()

    # preprocess.EDA().scatter_plot(X , y)
    modeling.tmp(X,y).test(False)
    # print(pre_proc)
