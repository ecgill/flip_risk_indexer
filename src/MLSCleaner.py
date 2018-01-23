import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from sklearn.base import TransformerMixin, BaseEstimator

class MLSCleaner(TransformerMixin, BaseEstimator):
    '''
    A class that takes MLS .csv and cleans it.
    '''
    def __init__(self):
        self.cols_to_drop = ['id', 'listing_number', 'status_changed_on', 'listed_on',
               'contracted_on', 'original_list_price','above_grade_square_feet',
               'total_square_feet', 'finished_square_feet','derived_basement_square_feet',
               'garages', 'property_key', 'externally_last_updated_at', 'photos',
               'is_attached', 'year', 'month', 'basement_finished_status',
               'year_built', 'lat', 'lng', 'zip', 'deal_type', 'stories']

    def fit(self, X, y=None):
        ''' Does nothing'''
        return self

    def transform(self, X):
        ''' Applies methods and returns clean feature matrix'''
        X = self._make_deal_types(X)
        X = self._make_one_story(X)
        X = self._make_age(X)
        X = self._make_basement(X)
        X = self._make_lodo_dist(X)
        X = self._make_dummies(X)
        X = self._drop_cols_rows(X)
        return X

    def _make_deal_types(self, X):
        X.loc[:,'fnf'] = 0
        X.loc[X['deal_type'] == 'fnf-80', 'fnf'] = 1
        X.loc[:,'td'] = 0
        X.loc[X['deal_type'] == 'td-60', 'td'] = 1
        return X

    def _make_one_story(self, X):
        X.loc[X['stories'] == '1', 'stories'] = 1
        X.loc[X['stories'] == '2', 'stories'] = 2
        X.loc[X['stories'] == '3', 'stories'] = 3
        X.loc[:,'one_story'] = 0
        X.loc[X['stories'] == 1, 'one_story'] = 1
        return X

    def _make_age(self, X):
        X.loc[:,'age'] = X['year'] - X['year_built']
        return X

    def _make_basement(self, X):
        ''' this will be a function that uses sq ftage to create features about
        whether the house has no basement, unfinished basement, or finished basement'''
        X.loc[:,'basement'] = 'none'
        basement_codes = [(0.0, 'unfinished'), ('0', 'unfinished'),
                            ('O', 'unfinished'), ('F', 'finished'),
                            ('1', 'partial'), (1.0, 'partial')]
        for k, v in basement_codes:
            mask = X['basement_finished_status'] == k
            X.loc[mask, 'basement'] = v
        return X

    def _make_lodo_dist(self, X):
        X.loc[:,'lat'] = X['lat'].astype(float)
        X.loc[:,'lng'] = X['lng'].astype(float)
        X.loc[:,'lodo_dist'] = X.apply(lambda row: self._calc_lodo_dist(row['lat'], row['lng']), axis=1)
        return X

    def _calc_lodo_dist(self, pt_lat, pt_lon):
        lodo_lat = radians(39.742043)
        lodo_lon = radians(-104.991531)
        R = 6373.0
        dlon = pt_lon - lodo_lon
        dlat = pt_lat - lodo_lat
        a = sin(dlat/2)**2 + cos(lodo_lat) * cos(pt_lat) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def _make_dummies(self, X):
        X = X.copy()
        X['garages'].fillna(0, inplace=True)
        X.loc[:,'has_garage'] = pd.Series(np.where(X['garages'].values > 0, 'garage', 'none'), X.index)
        X.loc[X['has_garage'] == 'garage', 'has_garage'] = 1
        X.loc[X['has_garage'] == 'none', 'has_garage'] = 0
        X.loc[X['property_type'] == 'RES', 'property_type'] = 1
        X.loc[X['property_type'] != 'RES', 'property_type'] = 0
        X.loc[X['structural_type'] == 'DETSF', 'structural_type'] = 1
        X.loc[X['structural_type'] != 'DETSF', 'structural_type'] = 0
        return X

    def _drop_cols_rows(self, X):
        X = X.drop(self.cols_to_drop, axis=1)
        return X

if __name__ == '__main__':
    pass
