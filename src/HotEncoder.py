from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder

class HotEncoder(TransformerMixin, BaseEstimator):
    '''
    A class that selects certain columns of a dataframe.
    '''
    def __init__(self):
        self._encoder = OneHotEncoder()

    def fit(self, X, y=None):
        ''' Does nothing'''
        n_rows = X.shape[0]
        self._encoder.fit(X.reshape([n_rows,1]))
        return self

    def transform(self, X):
        ''' Applies methods and returns clean feature matrix'''
        n_rows = X.shape[0]
        X = self._encoder.transform(X.reshape([n_rows,1]))
        return X.todense()

if __name__ == '__main__':
    pass
