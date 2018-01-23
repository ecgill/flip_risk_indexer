import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class FlipRiskIndexer(object):
    '''
    A model that determines potential percentage gain on investment options.
    '''

    def __init__(self):
        self._regressor = RandomForestRegressor()

    def fit(self, X, y):
        '''
        Fit a text classifier model.

        Parameters
        ----------
        X (np array): A feature matrix, to be used as predictors.
        y (np array): An array of past percentage gains, to be used as responses.

        Returns
        -------
        self: The fit model object.
        '''
        self._regressor.fit(X, y)
        return self

    def predict(self, X):
        '''Make predictions on new data.'''
        return self._regressor.predict(X)

    def score(self, X, y):
        '''Return a classification accuracy score on new data.'''
        return self._regressor.score(X, y)

    def plot_feature_importances(self, X):
        importances = self._regressor.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self._regressor.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()



if __name__ == '__main__':
    pass
    # pkl = 'data/merged_sold.pickle'
    # X, y = get_data(pkl)
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #     test_size=0.3, random_state=42)
    #
    # fri = FlipRiskIndexer()
    # fri.fit(X_train, y_train)
    # y_pred = fri.predict(X_test)
    #
    # print('Training R2: {:.2f}'.format(fri.score(X_train, y_train)))
    # print('Test R2: {:.2f}'.format(fri.score(X_test, y_test)))
    #
    # # Plot feature importances
    # fri.plot_feature_importances(X_train)

    # Residual plots


    # missing_lns = df[df['listing_number_y'].isnull()]['listing_number_previous']
