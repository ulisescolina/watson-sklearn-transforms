from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class EncodeCategoricalData(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_map):
        self.categorical_map = categorical_map
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        return data.replace(self.categorical_map)
