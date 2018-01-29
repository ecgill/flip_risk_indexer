import os
import glob
import pickle
import imageio
import numpy as np
import src.library as lib
from GoogleMapPlotter import GoogleMapPlotter

def make_one_map(df, api_key):
    lat = df['lat'].values
    lon = df['lng'].values
    path = [tuple(lat), tuple(lon)]
    mymap = GoogleMapPlotter(39.728, -104.963, 11, apikey=api_key)
    mymap.heatmap(path[0], path[1], radius=15, maxIntensity=10)
    dir_name = 'static/'
    map_name =  'all_flips_heat.html'
    mymap.draw(dir_name + map_name)


def make_recent_maps(df, api_key):
    df_sold = df[df['status'] == 'sold']
    deal_types = ['fix n flip', 'pop top', 'scrape', 'all types']
    deal_type_dict = {'fix n flip': 'fnf-80', 'pop top': 'pt-70', 'scrape': 'td-60'}
    time_periods = ['past year', 'past 6 months', 'past 3 months']
    for deal in deal_types:
        dealname = deal.replace(" ", "")
        if deal == 'all types':
            df_sub = df_sold.copy()
        else:
            df_sub = df_sold[df_sold['deal_type'] == deal_type_dict[deal]].copy()
        for period in time_periods:
            periodname = period.replace(" ", "")
            if period == 'past year':
                df_plot = df_sub[df_sub['year'] == 2017]
            elif period == 'past 6 months':
                df_plot = df_sub[(df_sub['year'] == 2017) & (df_sub['month'].isin([7,8,9,10,11,12]))]
            else:
                df_plot = df_sub[(df_sub['year'] == 2017) & (df_sub['month'].isin([10,11,12]))]

            n_flips = len(df_plot)
            lat = df_plot['lat'].values
            lon = df_plot['lng'].values
            path = [tuple(lat), tuple(lon)]
            mymap = GoogleMapPlotter(39.728, -104.963, 11, apikey=api_key)
            mymap.heatmap(path[0], path[1], radius=25, maxIntensity=10)
            mymap.text(39.728+0.14, -104.963+0.16, color='k', text=('Total deals done: ' + str(n_flips)))
            dir_name = 'static/'
            map_name =  dealname + '_' + periodname + '.html'
            mymap.draw(dir_name + map_name)

def make_imgs_for_gif(df, api_key):
    lat_m = df['lat'].mean()
    lon_m = df['lng'].mean()
    years = np.array(range(2010,2018))
    labels = [['JFMA'], ['MJJA'], ['SOND']]
    months = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

    for yr in years:
        print('\n \n {}: '.format(yr))
        for i, m in enumerate(months):
            sub_df = df[(df['year'] == yr) & (df['month'].isin(m))]
            sub_lat = sub_df['lat'].values
            sub_lon = sub_df['lng'].values
            print('{} ({}) - {} investments'.format(labels[i], m, len(sub_lat)))
            path = [tuple(sub_lat), tuple(sub_lon)]
            mymap = GoogleMapPlotter(39.728+.05, -104.963, 11, apikey=api_key)
            mymap.heatmap(path[0], path[1], radius=50, maxIntensity=5)
            mymap.text(lat_m+0.15, lon_m+0.16, color='k', text=str(yr) + ' ' + labels[i][0])
            dir_name = 'static/maps/html_for_gif/'
            map_name =  str(yr) + '_' + str(i) + '_' + labels[i][0] + '.html'
            mymap.draw(dir_name + map_name)


def make_gif(fps=1.0):
    filenames = glob.glob(os.path.join('static/maps/png_for_gif/', '*.png'))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('static/heatmap.gif', images, fps=fps)

def main():
    api_key = 'AIzaSyD3vBwndDQ1bj2bfbUth1vxoch2S_HEKhA'

    print('--- Get data for making heat maps ---')
    # flip_fn = 'data/denver-deals-clean.csv'
    # df_flips = lib.read_flips(flip_fn)
    with open('data/flips.pkl', 'rb') as f:
        df_flips = pickle.load(f)

    print('--- Make recent heatmaps for webpage ---')
    make_recent_maps(df_flips, api_key)

    print('--- Make a single Google map ---')
    # make_one_map(df_flips, api_key)

    print('--- Select only "sold" properties ---')
    # df_sold = df_flips[df_flips['status'] == 'sold']
    # df_plot = df_sold[['deal_type', 'lat', 'lng', 'perc_gain', 'status_changed_on','year', 'month']].copy()

    print('--- Making gmap images to be used in gif ---')
    # make_imgs_for_gif(df_plot, api_key)

    print('--- Making gif ---')
    # make_gif()

if __name__ == '__main__':
    # pass
    main()
    # make_gif()
