# resources: week9-day1-dsi-afternoon-webapparvos
# run webapparvos and look at how theyre pulling in data
# resources: solns-dsi-dataproducts-individual-myappsoln
# look at the app soln for simple example and unpickling model, etc.

from flask import Flask, render_template, request
from flask_table import Table, Col
import pickle

import sys,os
sys.path.append(os.path.join(os.path.expanduser('~'),'Documents', 'git_data_sci_proj', 'flip_risk_indexer','src'))
from run

# from ..src import run

app = Flask(__name__)

with open('../data/model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

class Top20Table(Table):
    ''' Quick class to make a Flask table '''
    listing = Col('Listing Number')
    list_price = Col('List Price')
    predicted_gain = Col('Predicted Selling Price')
    address = Col('Address')
    beds = Col('Beds')
    baths = Col('Baths')
    sqft = Col('Square Feet')

class Top20Item(object):
    ''' Quick class to define table objects, each will be a row '''
    def __init__(self, listing, list_price, predicted_gain, address, beds, baths, sqft):
        self.listing = listing
        self.list_price = list_price
        self.predicted_gain = predicted_gain
        self.address = address
        self.beds = beds
        self.baths = baths
        self.sqft = sqft

# home page
@app.route('/')
def home():
    return render_template('home.html')

# input page
@app.route('/input')
def results():
    types = ["fix 'n flip", "pop-top", "scrape"]
    return render_template('input.html', deal_types = types )

# # prediction page
@app.route('/predict', methods=['POST'])
def predict():
    u_reno_budget = str(request.form['u_reno_budget'])
    u_deal_type = str(request.form['u_deal_type'])
    # pipeline.predict...
    # make the map...

    # make the table from top 20...

    items = [Top20Item('1', '500K', '120K', 'blah blah', '2', '1', '2000'),
         Top20Item('1', '500K', '120K', 'blah blah', '2', '1', '2000'),
         Top20Item('1', '500K', '120K', 'blah blah', '2', '1', '2000')]

    # items = [dict(name='Name1', description='Description1'),
    #      dict(name='Name2', description='Description2'),
    #      dict(name='Name3', description='Description3')]

    # Populate the table
    table = Top20Table(items)

    html_map = '../static/maps/predict_investments.html'
    return render_template('predict.html', html_map=html_map,
                            u_deal_type=u_deal_type, u_reno_budget = u_reno_budget,
                            top20table=table)

# about page
@app.route('/about')
def about():
    return render_template('about.html')

# contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, debug=True, threaded=True)
