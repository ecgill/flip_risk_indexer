from sklearn.base import TransformerMixin, BaseEstimator

class DFSelector(TransformerMixin, BaseEstimator):
    '''
    A class that selects certain columns of a dataframe.
    '''
    def __init__(self, cols_to_select):
        self.cols_to_select = cols_to_select

    def fit(self, X, y=None):
        ''' Does nothing'''
        return self

    def transform(self, X):
        ''' '''
        return X[self.cols_to_select].values


if __name__ == '__main__':
    pass
