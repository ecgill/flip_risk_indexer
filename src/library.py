import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def write_clean_mls(filename):
    '''
    Reads in raw mls csv line by line, skips blank rows or rows that don't
        start with a numeric, and then writes the clean rows to a new csv.
    Args:
    ---------
        filename (str): Filename of raw MLS .csv
    Returns:
    ---------
        None. Writes a clean .csv to working directory.
    '''
    fidin = open(filename)
    reader = csv.reader(fidin)

    fn = filename.split('.')

    fidout = open(fn[0] + '_trunc.' + fn[1],'w')
    writer = csv.writer(fidout)

    fidout_skipped = open(fn[0] + '_skipped.' + fn[1],'w')
    writer_skipped = csv.writer(fidout_skipped)

    lines_empty = []
    lines_skipped = []
    lines_printed = 0
    long_lines = []
    for i, line in enumerate(reader):
        if i % 100000 == 0:
            print('Reading line # {}'.format(i))
        if i == 0:
            writer.writerow(line)

        if not line:
            lines_empty.append(i)   # all good. definitely blank lines.
            continue
        elif len(line) > 62:
            print(line)
            long_lines.append(i)
            continue
        elif line[0].isdigit():
            # writer.writerow(line[:50])
            writer.writerow(line)
            lines_printed += 1
        else:
            writer_skipped.writerow(line)
            lines_skipped.append(i)
    print('# Lines: Wrote-{}, Non-Numeric-{}, Empty-{}, Long-{}'.format(lines_printed,
            len(lines_skipped), len(lines_empty), len(long_lines)))

def read_mls(filename):
    '''
    Reads in clean mls csv, drops unneeded columns, turns listing status
        date into a datetime object, which is then used to create year and month
        columns.
    Args:
    ---------
        filename (str): Filename of clean MLS .csv
    Returns:
    ---------
        df (pandas DataFrame): A df of potential features for regression model.
    '''
    fn = filename.split('.')
    df = pd.read_csv(fn[0] + '_trunc.' + fn[1])
    cols_to_drop = ['off_market_on', 'previous_price', 'area', 'hoa_name',
                    'hoa_fee', 'school_district', 'county', 'architecture',
                    'lot_size_acres', 'lot_size_square_feet', 'basement_type',
                    'listing_agent', 'listing_brokerage', 'version',
                    'approval_condition', 'seller_concessions', 'car_spaces',
                    'car_storage', 'subdivision', 'structural_style',
                    'basement_finished_pct', 'basement_square_feet',
                    'basement_size', 'seller_type', 'zoned', 'showings_phone',
                    'parcel_number', 'sold_on', 'sold_price', 'created_at',
                    'updated_at','property_id']
    df.drop(cols_to_drop, axis=1, inplace=True)
    df['status_changed_on'] = pd.to_datetime(df['status_changed_on'],
        format='%Y/%m/%d')
    df.loc[:,'year'] = df['status_changed_on'].dt.year
    df.loc[:,'month'] = df['status_changed_on'].dt.month
    df['listing_number'] = df['listing_number'].astype(str)
    return df

def read_flips(filename):
    '''
    Reads in csv of deals already flagged by Privy as fix n flips (fnf), pop-tops
        (pt) or tear downs (td) with current listing number and previous listing
        number. It also creates year and month columns, lowercases some string
        columns and creates a perc_gain column based on difference between the
        re-sale value and the original cost of the fix.
    Args:
    ---------
        filename (str): Filename of deal csv
    Returns:
    ---------
        df (pandas DataFrame): A df of potential features for regression model.
    '''
    df = pd.read_csv(filename, encoding = 'ISO-8859-1')
    # df = df[(df['lat'] > 39.2) & (df['lat'] < 40.4) & (df['lng'] > -105.4) & (df['lng'] < -104.4)]
    df = df[(df['lat'] > 39.515) & (df['lat'] < 39.968) & (df['lng'] > -105.244) & (df['lng'] < -104.707)]
    df.loc[:,'perc_gain'] = df['last_price_change']/(df['status_price'] - df['last_price_change'])
    df['status_changed_on'] = pd.to_datetime(df['status_changed_on'],
        format='%m/%d/%y')
    df.loc[:,'year'] = df['status_changed_on'].dt.year
    df.loc[:,'month'] = df['status_changed_on'].dt.month
    df.loc[:,'city'] = df['city'].str.lower()
    df.loc[:,'county'] = df['county'].str.lower()
    df.loc[:,'perc_gain'] = df['last_price_change']/(df['status_price'] - df['last_price_change'])
    return df

def get_past_invest(df_mls, df_flips, y = 'perc_gain'):
    '''
    Reads in mls and flip dataframes and merges them based on the listing number
        from the the historical records (mls) and the listing_number_previous from
        the deal df. It only takes deals that have been sold the second time.
    Args:
    ---------
        df_mls (pandas DataFrame): dataframe of historical MLS listings
        df_mls (pandas DataFrame): dataframe with deal types
        y (str): name of column that will be the target variable
    Returns:
    ---------
        df_past_invest (pandas DataFrame): A df all houses in Denver that have been
            bought, flipped in one of those three main ways, and resold.
    '''
    df_sold = df_flips[df_flips['status'] == 'sold']
    df_past_invest = pd.merge(df_sold[['listing_number_previous', 'deal_type', y]],
                           df_mls,
                           left_on = 'listing_number_previous',
                           right_on = 'listing_number',
                           how = 'left',
                           suffixes=('_x', ''))
    # index = df_past_invest['listing_number'].index[df_past_invest['listing_number'].isnull() == True]
    df_past_invest = df_past_invest[df_past_invest['list_price'] < 500000]
    df_past_invest.drop(['listing_number_previous','status','street','city','state'], axis=1, inplace=True)
    return df_past_invest

def get_active_listings(df_mls):
    df_active = df_mls[(df_mls['status'] == 'active') & (df_mls['year'] == 2017)]
    df_active = df_active[df_active['list_price'] < 500000]
    df_active = df_active[(df_active['lat'] > 39.515) & (df_active['lat'] < 39.968)
                        & (df_active['lng'] > -105.244) & (df_active['lng'] < -104.707)]

    return df_active

def plot_kde2d(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""
    # create grid of sample locations (default: 100x100)
    X, Y = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]
    xy_sample = np.vstack([Y.ravel(), X.ravel()]).T
    xy_train  = np.vstack([y, x]).T
    kde = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde.fit(xy_train)
    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde.score_samples(xy_sample))
    Z = np.reshape(z, X.shape)
    plt.pcolormesh(X, Y, Z, cmap=plt.cm.RdYlGn_r)
    plt.title('2D Kernel Density Heat Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar()
    plt.savefig('images/2d_kernel_heatmap.png', transparent=True)
    return X, Y, Z


if __name__ == '__main__':
    # from src.run import get_data
    # from sklearn.model_selection import train_test_split
    #
    # print('--- Get data -- ')
    # X, y = get_data()
    #
    # print('--- Test train split data ---')
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #     test_size=0.3, random_state=42)
    #
    #
    # lat = X_train['lat'].values
    # lon = X_train['lng'].values
    #
    # xx, yy, zz = plot_kde2d(lon, ylat, 0.02)

    flips = 'data/denver-deals-clean.csv'
    df_flips = read_flips(flips)

    df_plot = df_flips[['deal_type', 'lat', 'lng', 'perc_gain', 'status_changed_on']].copy()
    lat_m = df_plot['lat'].mean()
    lon_m = df_plot['lng'].mean()

    lat = df_plot['lat'].values
    lon = df_plot['lng'].values

    xx, yy, zz = plot_kde2d(lon, lat, 0.02)
