from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder

class Labeler(TransformerMixin, BaseEstimator):
    '''
    A class that selects certain columns of a dataframe.
    '''
    def __init__(self):
        self._labeler = LabelEncoder()

    def fit(self, X, y=None):
        ''' Does nothing'''
        n_rows = X.shape[0]
        self._labeler.fit(X.reshape([n_rows,]))
        return self

    def transform(self, X):
        ''' Applies methods and returns clean feature matrix'''
        n_rows = X.shape[0]
        X = self._labeler.transform(X.reshape([n_rows,]))
        return X

if __name__ == '__main__':
    pass
