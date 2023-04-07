import pandas as pd
from datetime import datetime
import numpy as np
from geopy.distance import geodesic as GD


#%% Import the data

train = pd.read_csv("train.csv")
train = train.iloc[:45593,:]
train = train.replace('NaN', float(np.nan), regex=True)
train = train.dropna().reset_index(drop=True)

#%% Clean and reformat the data where necessary

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

train = pd.read_csv("train.csv")
train = train.iloc[:45593,:]
train= train.replace('NaN', float(np.nan), regex=True)
train = train.dropna()
train = train.reset_index(drop=True)

def clean_reformat_data(df):
    # Remove Delivery and Driver IDs
    df = df.iloc[:, 2:]

    # Format Order Date to Datetime
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%m/%d/%Y')

    # Convert the geographical coordinates into a geodistance calculation
    df['Geo_Distance']=np.zeros(len(df))
    res_coordinates=df[['Restaurant_latitude','Restaurant_longitude']].to_numpy()
    del_coordinates=df[['Delivery_location_latitude','Delivery_location_longitude']].to_numpy()
    for i in range(len(df)):
        df['Geo_Distance'].loc[i]=GD(res_coordinates[i],del_coordinates[i])
        
    df['Geo_Distance']=df['Geo_Distance'].astype("str").str.extract('(\d+)')
    df['Geo_Distance']=df['Geo_Distance'].astype("float")
    df = df[df['Geo_Distance'] < 100].reset_index(drop=True) # Remove any outliers


    # Format order times to datetime
    df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S')
    df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'], format='%H:%M:%S')
    
    # Clean and reformat weather conditions column
    df['Weatherconditions'] = df['Weatherconditions'].str.split(" ", expand=True)[1]
    df['Sunny'] = np.where(df['Weatherconditions'] == 'Sunny', 1, 0)
    df['Fog'] = np.where(df['Weatherconditions'].isin(['Fog', 'Cloudy']), 1, 0)
    
    # Road traffic density
    df['Road_traffic_density'] = df['Road_traffic_density'].map({'Low ': 1, 'Medium ': 2, 'High ': 3, 'Jam ': 4})

    # Format Type of Vehicle
    df['Motorcycle'] = np.where(df['Type_of_vehicle'] == 'motorcycle ', 1, 0)

    # Format Festival
    df['Festival'] = np.where(df['Festival'] == 'Yes ', 1, 0)
    
    # Format City
    city_map = {'Semi-Urban ': 1, 'Urban ': 2, 'Metropolitian ': 3}
    df['Urban_Density'] = df['City'].map(city_map)

    # Remove the "(min)" from the time taken
    df['Time_taken(min)'] = df['Time_taken(min)'].str.split(" ", expand=True)[1]
    
    return df


def prelim_feature_engineering(df):
    
    # Create Time Features
    df['Hour'] = df['Time_Order_picked'].dt.hour
    df['Night'] = np.where(df['Hour'].isin([22, 23, 24, 1, 2, 3, 4]), 1, 0)
    df['Rush_Hour'] = np.where(df['Hour'].isin([7, 8, 9, 16, 17, 18]), 1, 0)
    
    # Create Date Features
    df["month"] = df.Order_Date.dt.month
    df["month_even"] = np.where(df['month'].isin([2, 4, 6, 8, 10, 12]), 1, 0)
    df['day_of_week'] = df.Order_Date.dt.dayofweek.astype(int)
    df = df.drop(columns=["month", "day_of_week"], axis=1) # Don't need these anymore

    return df

train_clean=clean_reformat_data(train)
train_clean=prelim_feature_engineering(train_clean)

# Drop redundant columns
cols_to_drop = ['Weatherconditions', 'Order_Date', 'Time_Orderd', 'Time_Order_picked',
                'Type_of_vehicle', 'City', 'Type_of_order', 'Delivery_location_latitude',
                'Delivery_location_longitude', 'Restaurant_latitude', 'Restaurant_longitude']

train_clean.drop(columns=cols_to_drop, inplace=True)
    
# Write into a csv 
train_clean.to_csv('train_clean.csv', index=False)