
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class MultinomialNaiveBayesClassifier:

    def __init__(self):
        
        self.is_trained = False
        self.feature_columns = None
        self.target_values = None
        self.probabilities = {
            'priors': {},
            'conditionals': {}
        }

    def fit(self, feature_vars, target_var):
        
        if not isinstance(feature_vars, pd.DataFrame):
            raise Exception("feature_vars must be a Pandas Dataframe.")
        
        if not isinstance(target_var, pd.Series):
            raise Exception("feature_vars must be a Pandas Series.")
        
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
            feature_vars_value_count = feature_vars.loc[:, feature].value_counts()

            for value in feature_vars_value_count.keys():

                self.probabilities['conditionals'][feature][value] = {}
                filtered = feature_vars.loc[feature_vars[feature] == value]

                for target_value in self.target_values:
                    
                    temp_prob = 0
                    for target_matched_index in filtered.index:
                        if target_var[target_matched_index] == target_value:
                            temp_prob += 1
                
                    self.probabilities['conditionals'][feature][value][target_value] = temp_prob / target_value_counts[target_value]

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

                    temp_result[target_value] *= self.probabilities['conditionals'][feature_name][feature_value][target_value]

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



df = pd.read_csv('salary.csv')

mnb = MultinomialNaiveBayesClassifier()

X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]

mnb.fit(X, Y)

y_test = pd.DataFrame(
    [
        ["facebook","computer programmer","bachelors"],
        ["abc pharma","business manager","bachelors"],
    ], 
    columns=["company","job","degree"]
)


y_pred =  mnb.predict(y_test)
print(y_pred)


# df = pd.read_csv('data.csv')

# mnb = MultinomialNaiveBayesClassifier()

# X = df.iloc[:, 1:-1]
# Y = df.iloc[:, -1]

# mnb.fit(X, Y)

# y_test = pd.DataFrame(
#     [
#         ["Sunny", "Cool", "High", "Strong"],
#         ["Overcast", "Hot", "High", "Strong"],
#     ], 
#     columns=["Outlook", "Temperature", "Humidity", "Wind"]
# )

# y_pred =  mnb.predict(y_test)
# print(y_pred)