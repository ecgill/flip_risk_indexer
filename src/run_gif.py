import glob
import imageio
import numpy as np
import src.library as lib
from src.GoogleMapPlotter import GoogleMapPlotter

def make_one_map(df):
    lat = df['lat'].values
    lon = df['lng'].values
    path = [tuple(lat), tuple(lon)]
    mymap = GoogleMapPlotter(39.728, -104.963, 11)
    mymap.heatmap(path[0], path[1], radius=15, maxIntensity=10)
    dir_name = 'images/maps/'
    map_name =  'all_flips_heat.html'
    mymap.draw(dir_name + map_name)

def make_imgs_for_gif(df):
    lat_m = df['lat'].mean()
    lon_m = df['lng'].mean()
    years = np.array(range(2008,2018))
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
            mymap = GoogleMapPlotter(39.728, -104.963, 11)
            mymap.heatmap(path[0], path[1], radius=50, maxIntensity=5)
            mymap.text(lat_m+0.12, lon_m+0.12, color='k', text=str(yr) + ' ' + labels[i][0])
            dir_name = 'images/maps/'
            map_name =  str(yr) + '_' + str(i) + '_' + labels[i][0] + '.html'
            mymap.draw(dir_name + map_name)


def make_gif(filenames, fps=1.0):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('images/maps/pngs/test.gif', images, fps=fps)

def main():
    print('--- Get data for making heat maps ---')
    flip_fn = 'data/denver-deals-clean.csv'
    df_flips = lib.read_flips(flip_fn)

    print('--- Make a single Google map ---')
    make_one_map(df_flips)

    print('--- Select only "sold" properties ---')
    df_sold = df_flips[df_flips['status'] == 'sold']
    df_plot = df_sold[['deal_type', 'lat', 'lng', 'perc_gain', 'status_changed_on','year', 'month']].copy()

    print('--- Making gmap images to be used in gif ---')
    make_imgs_for_gif(df_plot)

if __name__ == '__main__':
    main()
