import pandas as pd
import src.library as lib
import _pickle as pickle
from src.GoogleMapPlotter import GoogleMapPlotter

def predict_investment_heatmap(best, black, gray, red):
    flip_fn = 'data/denver-deals-clean.csv'
    df_flips = lib.read_flips(flip_fn)
    df_sold = df_flips[df_flips['status'] == 'sold']
    df_plot = df_sold[['deal_type', 'lat', 'lng', 'perc_gain', 'status_changed_on','year', 'month']].copy()
    recent_heat = df_plot[(df_plot['year'] == 2017) &
                          (df_plot['month'].isin([7, 8, 9, 10, 11, 12]))]

    # Define paths for Google Map plotting:
    path_heat = [tuple(recent_heat['lat'].values), tuple(recent_heat['lng'].values)]
    pins_black = [tuple(black['lat'].values), tuple(black['lng'].values)]
    pins_gray = [tuple(gray['lat'].values), tuple(gray['lng'].values)]
    pins_red = [tuple(red['lat'].values), tuple(red['lng'].values)]
    pins_best = [tuple(best['lat'].values), tuple(best['lng'].values)]

    # Make map:
    api_key = 'AIzaSyDjQWALRhnei06NclCDO2NwMJKKCkIPtxY'
    gmap = GoogleMapPlotter(39.728, -104.963, 11, apikey=api_key)
    gmap.heatmap(path_heat[0], path_heat[1], radius=50, maxIntensity=5)
    gmap.scatter(pins_black[0], pins_black[1], black['listing_number'].values, color='black', alpha=1, s=60, marker=False)
    gmap.scatter(pins_gray[0], pins_gray[1], gray['listing_number'].values, color='gray', alpha=1, s=60, marker=False)
    gmap.scatter(pins_red[0], pins_red[1], red['listing_number'].values, color='red', alpha=1, s=60, marker=False)
    gmap.scatter(pins_best[0], pins_best[1], best['listing_number'].values, color='green', marker=True)

    # Save map:
    dir_name = 'images/maps/'
    map_name = 'predict_investments.html'
    gmap.draw(dir_name + map_name)

if __name__ == '__main__':
    # Google API: AIzaSyDjQWALRhnei06NclCDO2NwMJKKCkIPtxY

    print('--- Read in MLS data ---')
    mls = 'data/emily-property-listings-20180116.csv'
    df_mls = lib.read_mls(mls)

    print('--- Obtain currently active listings ---')
    df_active = lib.get_active_listings(df_mls)
    df_active = df_active.reset_index()
    df_active.drop(['index'], axis=1, inplace=True)

    print('--- Extract columns for model ---')
    cols_for_pipe = ['id', 'listing_number', 'status_changed_on',
        'listed_on','contracted_on', 'list_price', 'original_list_price',
       'above_grade_square_feet', 'total_square_feet', 'finished_square_feet',
       'derived_basement_square_feet', 'garages', 'beds', 'baths', 'zip',
       'property_key', 'externally_last_updated_at', 'photos', 'property_type',
       'year_built', 'basement_finished_status', 'lat', 'lng',
       'structural_type', 'is_attached', 'stories', 'year', 'month']
    X_new = df_active[cols_for_pipe].copy()

    print('--- Load pickled model ---')
    with open('data/model.pkl', 'rb') as f:
	       full_pipeline = pickle.load(f)

    print('--- User will pick a type of flip and enter reno budget ---')
    deal_type = 'fnf-80'  # 'pt-70', 'td-60'
    reno_budget = 80000

    print('--- Append deal type on to X_new df ---')
    X_new.loc[:,'deal_type'] = deal_type

    print('--- Predict on new data ---')
    y_pred_new = full_pipeline.predict(X_new.copy())
    gain = y_pred_new - X_new['list_price']
    X_predict = pd.concat([X_new[['deal_type','id','listing_number',
                                  'status_changed_on','list_price','property_key',
                                  'lat','lng','year_built','beds','baths']],
                           pd.Series(y_pred_new, name="y_pred"),
                           pd.Series(gain, name="gain"),
                           df_active[['street','city','state']]], axis=1)

    print('--- Define potential properties as black, gray, red ---')
    amae = 30000
    black = X_predict[X_predict['gain'] > (reno_budget + amae)]
    gray = X_predict[(X_predict['gain'] >= reno_budget) &
                     (X_predict['gain'] < (reno_budget + amae))]
    red = X_predict[(X_predict['gain'] < reno_budget)]

    print('--- Find top 5 investment options ---')
    best = X_predict.nlargest(20, 'gain')

    print('--- Plot on gmap ---')
    predict_investment_heatmap(best, black, gray, red)
