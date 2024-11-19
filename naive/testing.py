
import pandas as pd
from sklearn.metrics import accuracy_score

def k_fold_cross_validation(training_func, prediction_func, reset_func, X, y, k):

    def shuffle(X, y):

        X.reset_index()
        y.reset_index()
        
        df = X.join(pd.DataFrame(y)) # left join by index

        df = df.sample(frac=1)

        X = df.iloc[:, 0:-1]
        y = df.iloc[:, -1]

        return X, y


    if not isinstance(X, pd.DataFrame):
        raise Exception("X must be a Pandas Dataframe.")
        
    if not isinstance(y, pd.Series):
        raise Exception("y must be a Pandas Series.")
    
    if X.shape[0] != y.shape[0]:
        raise Exception("Number of rows in X and y must be same.")
    
    row_count = X.shape[0]

    if k < 2 or k > row_count:
        raise Exception(f"k must be in the range [2,{row_count}]")
    
    X, y = shuffle(X, y)
    
    # Splitting to K-Folds

    fold_size = row_count // k

    k_folds_X = []
    k_folds_y = []

    for i in range(k):

        if i == k-1:
            k_folds_X.append(X.iloc[(i * fold_size):row_count])
            k_folds_y.append(y.iloc[(i * fold_size):row_count])
        else:
            k_folds_X.append(X.iloc[(i * fold_size):((i+1) * fold_size)])
            k_folds_y.append(y.iloc[(i * fold_size):((i+1) * fold_size)])

    # Performing K-Fold Cross-Validation

    scores = []

    for count in range(k):

        # Combining all other folds for training

        test_X = k_folds_X.copy()
        del test_X[count]

        test_y = k_folds_y.copy()
        del test_y[count]

        test_X = pd.concat(test_X)
        test_y = pd.concat(test_y)

        training_func(test_X, test_y)

        # Testing

        pred_y = prediction_func(k_folds_X[count])

        scores.append(accuracy_score(k_folds_y[count], pred_y))
        reset_func()
    
    mean_score = sum(scores) / len(scores)
    return mean_score