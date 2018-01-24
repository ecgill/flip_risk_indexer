from flask import Flask, render_template
import pickle

app = Flask(__name__)
#
# with open('static/model.pkl', 'rb') as f:
#     model = pickle.load(f)


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
# @app.route('/predict', methods=['POST'])
# def predict():
#     u_reno_budget = str(request.form['u_reno_budget'])
#     u_deal_type = str(request.form['u_deal_type'])
#
#
#     model.predict...
#
#
#     return render_template('predict.html', reno_budget=u_reno_budget, deal_type=u_deal_type)


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
