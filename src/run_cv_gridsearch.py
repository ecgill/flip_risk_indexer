import numpy as np
from src.run import get_data, set_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import median_absolute_error, make_scorer
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import GridSearchCV
import seaborn as sns

def cross_validate(X, y, models):
    medianae = make_scorer(median_absolute_error)
    results = {}
    names = []
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', medianae]
    for name, model in models:
        pipe = set_pipeline(model)
        names.append(name)
        results[name] = []
        for score in scoring:
            print('cv for {} and {}'.format(name, score))
            cv_results = cross_val_score(pipe, X, y, cv=10, scoring=score)
            results[name].append(cv_results)
    return results, names, scoring

def print_cv_results(results, names, scoring):
    print('\n \n --- Cross Validation Scores ---')
    for i, name in enumerate(names):
        res = results[name]
        print('{}:\n {}-{:.2f} ({:.2f})\n {}-{:.2f} ({:.2f})\n {}-{:.2f} ({:.2f})\n {}-{:.2f} ({:.2f})'.format(name,
                                                            scoring[0],
                                                            res[0].mean(),
                                                            res[0].std(),
                                                            scoring[1],
                                                            np.abs(res[1]).mean(),
                                                            np.abs(res[1]).std(),
                                                            scoring[2],
                                                            np.sqrt(np.abs(res[2])).mean(),
                                                            np.sqrt(np.abs(res[2])).std(),
                                                            scoring[3],
                                                            res[3].mean(),
                                                            res[3].std()
                                                            ))

def boxplot_cv(cv_results, names):
    scoring = ['R2', 'RMSE', 'Mean AE', 'Median AE']
    for i, score in enumerate(scoring):
        to_box =[]
        labels=[]
        for name in names:
            if name == 'AB':
                continue
            labels.append(name)
            if score == 'RMSE':
                res = np.sqrt(np.abs(cv_results[name][i]))/1000
            elif score == 'Mean AE':
                res = np.abs(cv_results[name][i])/1000
            elif score == 'Median AE':
                res = (cv_results[name][i])/1000
            else:
                res = cv_results[name][i]
            to_box.append(res)
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison Using {}'.format(score))
        ax = fig.add_subplot(111)
        sns.boxplot(data=to_box, orient='v', ax=ax)
        ax.set_ylabel(score)
        ax.set_xticklabels(labels)
        ax.set_xlabel('sklearn Regressor models')
        plt.savefig('static/plots/cv_model_comparison_' + score + '.pdf', transparent=True)

if __name__ == '__main__':
    print('--- Get data -- ')
    X, y = get_data()

    print('--- Test train split data ---')
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3, random_state=42)

    # print('--- Set up list of models ---')
    # models = []
    # models.append(('LR', LinearRegression()))
    # models.append(('RD', Ridge()))
    # models.append(('RF', RandomForestRegressor(n_estimators=500)))
    # models.append(('AB', AdaBoostRegressor(n_estimators=50)))
    # models.append(('GB', GradientBoostingRegressor(n_estimators=500)))
    # models.append(('KNN', KNeighborsRegressor(n_neighbors=3)))
    # models.append(('BR', BayesianRidge()))

    # print('--- Cross validate ---')
    # results, names, scoring = cross_validate(X_train, y_train, models)
    # print_cv_results(results, names, scoring)

    # print('--- Boxplot CV Results ---')
    # boxplot_cv(results, names)

    print('--- GridSearch AB ---')
    model = AdaBoostRegressor()
    ab_pipe = set_pipeline(model)
    ab_param_grid = {
        'regress__n_estimators': (50, 100, 200, 300, 700),
        }
    ab_grid = GridSearchCV(ab_pipe, cv=3, n_jobs=-1, param_grid=ab_param_grid,
                        scoring='neg_mean_absolute_error')
    ab_grid.fit(X_train, y_train)
    ab_bestmod = ab_grid.best_estimator_

    # print('--- GridSearch RF ---')
    # model = RandomForestRegressor()
    # rf_pipe = set_pipeline(model)
    # rf_param_grid = {
    #     'make_heat__bandwidth': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    #     'regress__n_estimators': (750, 1000),
    #     'regress__min_samples_leaf': (5, 10),
    #     }
    # rf_grid = GridSearchCV(rf_pipe, cv=3, n_jobs=-1, param_grid=rf_param_grid,
    #                     scoring='neg_mean_absolute_error')
    # rf_grid.fit(X_train.copy(), y_train)
    # rf_bestmod = rf_grid.best_estimator_

    # print('--- GridSearch GB ---')
    # model = GradientBoostingRegressor()
    # gb_pipe = set_pipeline(model)
    # gb_param_grid = {
    #     'make_heat__bandwidth': (0.2),
    #     'regress__n_estimators': (100, 150, 250),
    #     'regress__learning_rate': (0.4, 0.5, 0.6),
    #     'regress__min_samples_leaf': (4, 5, 6),
    #     }
    # gb_grid = GridSearchCV(gb_pipe, cv=3, n_jobs=-1, param_grid=gb_param_grid,
    #                     scoring='neg_mean_absolute_error')
    # gb_grid.fit(X_train, y_train)
