from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primero realizamos una copia del dataframe 'X' de entrada
        data = X.copy()

        # Nos quedamos con las columnas deseadas
        data = data[self.columns]

        # Generamos las columnas dummie de HOUSING_FREE, HOUSING_OWN
        data['HOUSING_FREE'] = np.where(data['HOUSING']=='FREE',1,0)
        data['HOUSING_OWN'] = np.where(data['HOUSING']=='OWN',1,0)
        
        return data.drop(labels="HOUSING", axis='columns')
