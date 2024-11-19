
import pandas as pd
import math
import numpy as np
from pandas.api.types import is_numeric_dtype

class MultinomialNaiveBayesClassifier:

    def __init__(self, alpha=0):
        self.alpha = alpha
        self.reset()


    def reset(self):
        self.is_trained = False
        self.feature_columns = None
        self.target_values = None
        self.probabilities = {
            'priors': {},
            'conditionals': {},
            'unfamiliar_probs': {}
        }


    def fit(self, feature_vars, target_var):
        
        if not isinstance(feature_vars, pd.DataFrame):
            raise Exception("feature_vars must be a Pandas Dataframe.")
        
        if not isinstance(target_var, pd.Series):
            raise Exception("target_var must be a Pandas Series.")
        
        if feature_vars.shape[0] != target_var.shape[0]:
            raise Exception("Number of rows in feature_vars and target_var must be same.")
        
        self.feature_columns = feature_vars.columns
        self.target_values = target_var.value_counts().keys()

        # Calculating Prior probabilities

        target_value_counts = target_var.value_counts()

        for name in self.target_values:
            self.probabilities['priors'][name] = target_value_counts[name] / target_var.shape[0]

        # Calculating Conditional probabilities

        self.probabilities['conditionals'] = {}

        for feature in set(feature_vars.columns):

            self.probabilities['conditionals'][feature] = {}
            self.probabilities['unfamiliar_probs'][feature] = {}
            feature_vars_value_count = feature_vars.loc[:, feature].value_counts()

            for value in feature_vars_value_count.keys():

                self.probabilities['conditionals'][feature][value] = {}
                filtered = feature_vars.loc[feature_vars[feature] == value]

                for target_value in self.target_values:
                    
                    temp_prob = 0
                    for target_matched_index in filtered.index:
                        if target_var[target_matched_index] == target_value:
                            temp_prob += 1
                
                    self.probabilities['conditionals'][feature][value][target_value] = (temp_prob + self.alpha) / (target_value_counts[target_value] + self.alpha * len(feature_vars_value_count))
                    
                    # To deal with zero probabilities due to dealing with unfamiliar values
                    if target_value not in self.probabilities['unfamiliar_probs'][feature].keys():
                        self.probabilities['unfamiliar_probs'][feature][target_value] = self.alpha / (target_value_counts[target_value] + self.alpha * len(feature_vars_value_count))

        self.is_trained = True


    def predict(self, feature_vars):

        if self.is_trained == False:
            raise Exception("Model is not trained.")

        if not isinstance(feature_vars, pd.DataFrame):
            raise Exception("feature_vars must be a Pandas Dataframe.")
        
        if not (set(self.feature_columns.tolist()) & set(feature_vars.columns.tolist())):
            raise Exception("Feature names of feature_vars and trained data must be same.")
        
        # Calculating the probabilities for each outcome  

        target_probs = []

        for feature_index in feature_vars.index:
            
            temp_result = {}

            for target_value in self.target_values:
                
                temp_result[target_value] = self.probabilities['priors'][target_value]

                for feature_name, feature_value in feature_vars.loc[feature_index].items():

                    if feature_value in self.probabilities['conditionals'][feature_name].keys():
                        temp_result[target_value] *= self.probabilities['conditionals'][feature_name][feature_value][target_value]

                    else:
                        temp_result[target_value] *= self.probabilities['unfamiliar_probs'][feature_name][target_value]


            target_probs.append(temp_result)

        # Finding the predictor value from the calculated probabilities

        final_prediction = []
        
        for target in target_probs:

            max_k = None
            max_v = -1
            for key, value in target.items():

                if value > max_v:
                    max_v = value
                    max_k = key

            final_prediction.append(max_k)

        return final_prediction
    
class GaussianNaiveBayesClassifier:

    def __init__(self):
        self.reset()


    def reset(self, var_smoothing=1e-9):
        self.is_trained = False
        self.feature_columns = None
        self.target_values = None
        self.var_smoothing = var_smoothing
        self.probabilities = {
            'priors': {},
            'means': {},
            'variances': {},
        }


    def fit(self, feature_vars, target_var):
        
        if not isinstance(feature_vars, pd.DataFrame):
            raise Exception("feature_vars must be a Pandas Dataframe.")
        
        if not isinstance(target_var, pd.Series):
            raise Exception("target_var must be a Pandas Series.")
        
        if feature_vars.shape[0] != target_var.shape[0]:
            raise Exception("Number of rows in feature_vars and target_var must be same.")
        
        for col_name in feature_vars.columns:
            if not is_numeric_dtype(feature_vars[col_name]):
                raise Exception("All columns in feature_vars must be numeric dtype.")
            
        self.feature_columns = feature_vars.columns
        self.target_values = target_var.value_counts().keys()
            
        # Calculating Prior probabilities

        target_value_counts = target_var.value_counts()

        for name in self.target_values:
            self.probabilities['priors'][name] = target_value_counts[name] / target_var.shape[0]

        # Calculating Mean and Variance for each column in feature_vars

        for feature in set(feature_vars.columns):

            # Calculating Mean

            self.probabilities['means'][feature] = {}

            for target_index in feature_vars.index:

                if target_var[target_index] in self.probabilities['means'][feature].keys():
                    self.probabilities['means'][feature][target_var[target_index]] += feature_vars[feature][target_index]
                else:
                    self.probabilities['means'][feature][target_var[target_index]] = feature_vars[feature][target_index]

            for target_value in self.target_values:
                self.probabilities['means'][feature][target_value] /= target_value_counts[target_value]

            # Calculating Variance 

            self.probabilities['variances'][feature] = {}

            for target_index in feature_vars.index:

                if target_var[target_index] in self.probabilities['variances'][feature].keys():
                    self.probabilities['variances'][feature][target_var[target_index]] += (self.probabilities['means'][feature][target_var[target_index]] - feature_vars[feature][target_index]) ** 2
                else:
                    self.probabilities['variances'][feature][target_var[target_index]] = (self.probabilities['means'][feature][target_var[target_index]] - feature_vars[feature][target_index]) ** 2

            for target_value in self.target_values:
                self.probabilities['variances'][feature][target_value] /= target_value_counts[target_value]

        self.is_trained = True


    def predict(self, feature_vars):

        def f(x, mu, variance):

            if variance == 0: variance = self.var_smoothing
            sigma = math.sqrt(variance)

            return -0.5 * np.log(2 * np.pi) - np.log(sigma) - ((x - mu)**2) / (2 * sigma**2)

        if self.is_trained == False:
            raise Exception("Model is not trained.")

        if not isinstance(feature_vars, pd.DataFrame):
            raise Exception("feature_vars must be a Pandas Dataframe.")
        
        if not (set(self.feature_columns.tolist()) & set(feature_vars.columns.tolist())):
            raise Exception("Feature names of feature_vars and trained data must be same.")
        
        for col_name in feature_vars.columns:
            if not is_numeric_dtype(feature_vars[col_name]):
                raise Exception("All columns in feature_vars must be numeric dtype.")

        # Calculating the probabilities for each outcome  

        target_probs = []

        for feature_index in feature_vars.index:

            final_probs = {}

            for target_value in self.target_values:
                final_probs[target_value] = np.log(self.probabilities['priors'][target_value])

                for feature_name in self.feature_columns:
                    final_probs[target_value] += f(
                        feature_vars.loc[feature_index][feature_name], 
                        self.probabilities['means'][feature_name][target_value],
                        self.probabilities['variances'][feature_name][target_value]
                    )

            target_probs.append(final_probs)

        # Finding the predictor value from the calculated values

        final_prediction = []

        for target in target_probs:

            max_k = None
            max_v = -1 * math.inf
            for key, value in target.items():

                if value > max_v:
                    max_v = value
                    max_k = key

            final_prediction.append(max_k)

        return final_prediction