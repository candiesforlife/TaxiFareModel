# imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

from TaxiFareModel.encoders import  TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "CBenhaim"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_preprocess = Pipeline([
            ('transformer',DistanceTransformer()),
            ('scaler',RobustScaler())
        ])
        time_preprocess = Pipeline([
            ('transformer',TimeFeaturesEncoder()),
            ('encoder',OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        time_column = ['pickup_datetime']
        dist_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

        # create preprocessing pipeline
        preprocess = ColumnTransformer([
            ('time', time_preprocess, time_column),
            ('dist', dist_preprocess, dist_columns)
        ])

        self.pipeline = Pipeline([
            ('preprocess', preprocess),
            ('estimator', RandomForestRegressor())
        ])

        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline().fit(self.X, self.y)
        return self.pipeline


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        #print(rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



if __name__ == "__main__":

    # get data
    df = get_data(nrows=10_000)

    # clean data
    df = clean_data(df)

    # set X and y
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

    # train
    pipe = Trainer(X_train, y_train)
    pipe.run()

    # evaluate
    rmse = pipe.evaluate(X_test, y_test)
    print(rmse)

    pipe.mlflow_log_param("model", 'RandomForestRegressor')
    pipe.mlflow_log_metric('rmse', rmse)






