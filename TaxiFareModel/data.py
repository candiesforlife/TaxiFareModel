import pandas as pd

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def extract_time_features(df):
    timezone_name = 'America/New_York'
    time_column = 'pickup_datetime'
    df.index = pd.to_datetime(df[time_column])
    df.index = df.index.tz_convert(timezone_name)
    df["dow"] = df.index.weekday
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df.reset_index(drop=True)

if __name__ == '__main__':
    df = get_data()
