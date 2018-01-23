import numpy as np
import src.library as lib
import _pickle as pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.HotZonerator import HotZonerator
from src.MLSCleaner import MLSCleaner
from src.DFSelector import DFSelector
from src.Labeler import Labeler
from src.HotEncoder import HotEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

def set_pipeline(final_estimator):
    num_cols = ['list_price', 'beds', 'baths', 'age', 'lodo_dist', 'heat']
    cat_cols = ['basement']
    bin_cols = ['property_type', 'structural_type', 'has_garage', 'fnf', 'td', 'one_story']

    # Set up pipelines:
    num_pipeline = Pipeline([
    	('select_num', DFSelector(num_cols)),
    	('scale', StandardScaler())
    	])
    cat_pipeline = Pipeline([
    	('select_cat', DFSelector(cat_cols)),
    	('label',  Labeler()),
        ('hot_encode', HotEncoder())
    	])
    bin_pipeline = Pipeline([
        ('select_bin', DFSelector(bin_cols))
        ])
    full_pipeline = Pipeline([
        ('make_heat', HotZonerator(bandwidth=0.2)),
    	('cleaner', MLSCleaner()),
    	('feat_union', FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline),
            ('bin_pipeline', bin_pipeline)
            ])),
        ('regress', final_estimator)
        ])
    return full_pipeline

def plot_actual_v_pred(y_train, y_pred_train, y_test, y_pred_test):
    fig = plt.figure(figsize=(12,5), dpi=100)
    ax_train = fig.add_subplot(121)
    ax_train.set_ylabel('Predicted target (train)')
    ax_train.set_xlabel('Actual target (train)')
    ax_train.plot(y_train, y_train, 'k-', alpha=0.5)
    ax_train.scatter(y_train, y_pred_train, alpha=0.5, color='red')

    ax_test = fig.add_subplot(122)
    ax_test.set_ylabel('Predicted target (test)')
    ax_test.set_xlabel('Actual target (test)')
    ax_test.plot(y_test, y_test, 'k-', alpha=0.5)
    ax_test.scatter(y_test, y_pred_test, alpha=0.5)
    plt.show()

def print_feat_importance(forest_pipe):
    forest = forest_pipe.steps[3][1]
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    print('Feature ranking:')
    for f in range(len(importances)):
        print('{} feature {} ({:.2f})'.format(f+1, indices[f], importances[indices[f]]))

def get_data():
    mls = 'data/emily-property-listings-20180116.csv'
    flips = 'data/denver-deals-clean.csv'
    print('--- Reading dirty MLS .csv file line by line ---')
    # lib.write_clean_mls(mls)

    target = 'status_price'
    print('--- Define target variable as {}'.format(target))

    print('--- Reading clean MLS .csv file to pandas DF ---')
    df_mls = lib.read_mls(mls)

    print('--- Reading flips .csv file to pandas DF ---')
    df_flips = lib.read_flips(flips)

    print('--- Merging MLS and flips to create training data ---')
    df_past_invest = lib.get_past_invest(df_mls, df_flips, target)
    y = df_past_invest.pop(target).values
    X = df_past_invest.copy()

    return X, y


if __name__ == '__main__':
    print('--- Get data -- ')
    X, y = get_data()

    print('--- Test train split data ---')
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3, random_state=42)

    print('--- Set up PIPELINE ---')
    final_estimator = RandomForestRegressor(n_estimators=1000, min_samples_leaf=5)
    full_pipeline = set_pipeline(final_estimator)

    print('--- Fit PIPELINE ---')
    full_pipeline.fit(X_train.copy(), y_train)

    print('--- Plot predictions ---')
    y_pred_train = full_pipeline.predict(X_train.copy())
    y_pred_test = full_pipeline.predict(X_test.copy())
    plot_actual_v_pred(y_train, y_pred_train, y_test, y_pred_test)

    print('--- Score best model ---')
    print('    --- Training Score - R2 - {:.2f}'.format(full_pipeline.score(X_train.copy(), y_train)))
    print('    --- Training Score - RMSE - {:.2f}'.format(np.sqrt(mean_squared_error(y_pred_train, y_train))))
    print('    --- Training Score - MAE - {:.2f}'.format(mean_absolute_error(y_pred_train, y_train)))
    print('    --- Training Score - MedAE - {:.2f}'.format(median_absolute_error(y_pred_train, y_train)))
    print('    --- Test Score - R2 - {:.2f}'.format(full_pipeline.score(X_test.copy(), y_test)))
    print('    --- Test Score - RMSE - {:.2f}'.format(np.sqrt(mean_squared_error(y_pred_test, y_test))))
    print('    --- Test Score - MAE - {:.2f}'.format(mean_absolute_error(y_pred_test, y_test)))
    print('    --- Test Score - MedAE - {:.2f}'.format(median_absolute_error(y_pred_test, y_test)))

    print('--- Printing feature importances ---')
    print_feat_importance(full_pipeline)

    print('--- Run on all data ---')
    y_pred_all = full_pipeline.predict(X.copy())
    print('    --- Full Score - R2 - {:.2f}'.format(full_pipeline.score(X.copy(), y)))
    print('    --- Full Score - RMSE - {:.2f}'.format(np.sqrt(mean_squared_error(y_pred_all, y))))
    print('    --- Full Score - MAE - {:.2f}'.format(mean_absolute_error(y_pred_all, y)))
    print('    --- Full Score - MedAE - {:.2f}'.format(median_absolute_error(y_pred_all, y)))

    # print('--- Pickle model ---')
    # with open('data/model.pkl', 'wb') as f:
    #     pickle.dump(full_pipeline, f)
