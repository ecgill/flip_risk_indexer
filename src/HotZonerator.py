import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KernelDensity

class HotZonerator(TransformerMixin, BaseEstimator):
    '''
    A class that uses a KernelDensity estimator to feature engineer a hot zone
    feature that is indicative of how close a property is to a hot zone.
    '''
    def __init__(self, bandwidth=0.02):
        self.bandwidth = bandwidth
        self._kdestimator = KernelDensity(bandwidth=self.bandwidth)

    def fit(self, X, y=None):
        ''' Does nothing'''
        xy_pairs  = self._get_xy_pairs(X)
        self._kdestimator.fit(xy_pairs)
        return self

    def transform(self, X):
        ''' '''
        X = X.copy()
        xy_pairs  = self._get_xy_pairs(X)
        heat = np.exp(self._kdestimator.score_samples(xy_pairs))
        X.loc[:,'heat'] = heat
        return X

    def _get_xy_pairs(self, X):
        ylat = X['lat'].copy().values
        xlon = X['lng'].copy().values
        return np.vstack([ylat, xlon]).T

if __name__ == '__main__':
    pass
