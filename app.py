import os
import uuid
import glob
import pickle
import pandas as pd
from flask import Flask, render_template, request
from flask_table import Table, Col
from src.run import set_pipeline
from GoogleMapPlotter import GoogleMapPlotter

app = Flask(__name__)

with open('data/model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

with open('data/active.pkl', 'rb') as f:
    df_active = pickle.load(f)

with open('data/flips.pkl', 'rb') as f:
    df_flips = pickle.load(f)

def remove_previous_html():
    html_files = glob.glob(os.path.join('static/', '*_.html'))
    for f in html_files:
        os.remove(f)

def predict_investment_heatmap(best, black, gray, red):
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
    api_key = 'AIzaSyD3vBwndDQ1bj2bfbUth1vxoch2S_HEKhA'
    gmap = GoogleMapPlotter(39.728, -104.963, 13, apikey=api_key)
    # gmap = GoogleMapPlotter(39.728, -104.963, 13)
    gmap.heatmap(path_heat[0], path_heat[1], radius=50, maxIntensity=5)
    gmap.scatter(pins_black[0], pins_black[1], black['listing_number'].values, color='black', alpha=1, s=60, marker=False)
    gmap.scatter(pins_gray[0], pins_gray[1], gray['listing_number'].values, color='gray', alpha=1, s=60, marker=False)
    gmap.scatter(pins_red[0], pins_red[1], red['listing_number'].values, color='red', alpha=1, s=60, marker=False)
    gmap.scatter(pins_best[0], pins_best[1], best['listing_number'].values, color='green', marker=True)

    # Save map:
    u_filename = './static/' + str(uuid.uuid4()) + '_.html'
    gmap.draw(u_filename)
    return u_filename

def predict_flip_risk(pkl, amae, deal_type, reno_budget):
    df_new = df_active.reset_index()
    df_new.drop(['index'], axis=1, inplace=True)

    print('--- Extract columns for model ---')
    cols_for_pipe = ['id', 'listing_number', 'status_changed_on',
        'listed_on','contracted_on', 'list_price', 'original_list_price',
       'above_grade_square_feet', 'total_square_feet', 'finished_square_feet',
       'derived_basement_square_feet', 'garages', 'beds', 'baths', 'zip',
       'property_key', 'externally_last_updated_at', 'photos', 'property_type',
       'year_built', 'basement_finished_status', 'lat', 'lng',
       'structural_type', 'is_attached', 'stories', 'year', 'month']
    X_new = df_new[cols_for_pipe].copy()

    print('--- Append deal type on to X_new df ---')
    X_new.loc[:,'deal_type'] = deal_type

    print('--- Predict on new data ---')
    y_pred_new = pkl.predict(X_new.copy())
    gain = y_pred_new - X_new['list_price']
    total_investment = X_new['list_price'] + reno_budget
    perc_roi = (y_pred_new - X_new['list_price'])/(X_new['list_price'] + reno_budget)
    X_predict = pd.concat([X_new[['deal_type','id','listing_number',
                                  'status_changed_on','list_price','property_key',
                                  'lat','lng','year_built','beds','baths']],
                           pd.Series(y_pred_new, name="y_pred"),
                           pd.Series(gain, name="gain"),
                           pd.Series(total_investment, name='total_investment'),
                           pd.Series(perc_roi, name='perc_roi'),
                           df_new[['street','city','state']]], axis=1)

    print('--- Define potential properties as black, gray, red ---')
    black = X_predict[X_predict['gain'] > (reno_budget + amae)]
    gray = X_predict[(X_predict['gain'] >= reno_budget) &
                     (X_predict['gain'] < (reno_budget + amae))]
    red = X_predict[(X_predict['gain'] < reno_budget)]

    print('--- Find top 20 investment options ---')
    best = X_predict.nlargest(20, 'gain')

    return best, black, gray, red

def prepare_top20table(best):
    best_table = best[['listing_number', 'street', 'city', 'year_built', 'beds',
                       'baths', 'list_price', 'y_pred', 'gain', 'total_investment', 'perc_roi']].copy()
    best_table['year_built'] = best_table['year_built'].astype(int)
    best_table['y_pred'] = best_table['y_pred'].astype(int)
    best_table['gain'] = best_table['gain'].astype(int)
    best_table['perc_roi'] = best_table['perc_roi'].round(2)
    best_table = best_table.reset_index().astype(str)
    best_table.drop(['index'], axis=1, inplace=True)
    return best_table

class Top20Table(Table):
    ''' Quick class to make a Flask table '''
    listing_number = Col('Listing Number')
    street = Col('Street')
    city = Col('City')
    year_built = Col('Yr Built')
    beds = Col('Beds')
    baths = Col('Baths')
    list_price = Col('List Price')
    y_pred = Col('Pred Sell Price')
    gain = Col('Gain after Inv')
    total_investment = Col('Total Investment')
    perc_roi = Col('Predicted ROI')

# home page
@app.route('/')
def home():
    return render_template('home.html')

# hotzones page
@app.route('/hotzones')
def get_heatmap():
    deal_types = ["fix n flip", "pop top", "scrape", "all types"]
    time_periods = ["past year", "past 6 months", "past 3 months"]
    return render_template('hotzones.html', deal_types=deal_types, time_periods=time_periods)

# hotzones page showing requested map
@app.route('/hotzones', methods=['GET','POST'])
def show_heatmap():
    deal_types = ["fix n flip", "pop top", "scrape", "all types"]
    time_periods = ["past year", "past 6 months", "past 3 months"]
    u_time_period = str(request.form['u_time_period'])
    u_deal_type = str(request.form['u_deal_type'])
    map_filename = '.' + '/static/' + u_deal_type.replace(" ", "") + '_' + u_time_period.replace(" ","") + '.html'
    return render_template('hotzones.html', deal_types=deal_types, time_periods=time_periods,
                            map_filename=map_filename, u_time_period=u_time_period, u_deal_type=u_deal_type)

# input page
@app.route('/input')
def results():
    types = ["fix n flip", "pop top", "scrape"]
    return render_template('input.html', deal_types = types )

# # prediction page
@app.route('/predict', methods=['POST'])
def predict():
    remove_previous_html()
    u_reno_budget = str(request.form['u_reno_budget'])
    u_deal_type = str(request.form['u_deal_type'])
    deal_type_dict = {"fix n flip": 'fnf-80',
                      "pop top": 'pt-70',
                      "scrape": 'td-60'}
    deal_type = deal_type_dict[u_deal_type]
    reno_budget = int(u_reno_budget)*1000
    amae = 20000
    best, black, gray, red = predict_flip_risk(pipeline, amae, deal_type, reno_budget)
    items = prepare_top20table(best)
    items = items.to_dict('records')
    table = Top20Table(items)
    u_filename = predict_investment_heatmap(best, black, gray, red)
    html_map = '.' + u_filename
    return render_template('predict.html', html_map=html_map,
                            u_deal_type=u_deal_type, u_reno_budget = u_reno_budget,
                            top20table=table)

# contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, debug=True, threaded=True)
