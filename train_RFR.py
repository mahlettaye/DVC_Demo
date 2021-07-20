

import warnings


import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

from urllib.parse import urlparse

import seaborn as sns
import matplotlib.pyplot as plt

import logging
import dvc.api 

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

path ='data/wine-quality.csv'
repo="C:/Users/25192/DVC/"
version="v1"


data_url= dvc.api.get_url(
        path=path,
        repo=repo,
        rev=version)
mlflow.set_experiment('dvc_RFR')


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the versioned data
    csv_url = (
        data_url
    )
    try:
        data = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception(
            "Unable to load dataset", e
        )

    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)
    
    #clean data 
    data= clean_dataset(data)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]


     # log artifacts columns 
    cols_x=pd.DataFrame(list(train_x.columns))
    cols_x.to_csv('features.csv', header=False, index=False)
    #mlflow.log_artifact('feature.csv')
    
    cols_y=pd.DataFrame(list(train_y.columns))
    cols_y.to_csv('target.csv', header=False, index=False)
    #mlflow.log_artifact('target.csv')
    
    mlflow.end_run()
    with mlflow.start_run(nested =True):
        
        # Fit a model on the train section
        regr = RandomForestRegressor(max_depth=2, random_state=40)
        regr.fit(train_x, train_y)

        # Report training set score
        train_score = regr.score(train_x, train_y) * 100
        # Report test set score
        test_score = regr.score(test_x, test_y) * 100

        predicted_qualities = regr.predict(test_x)
        
        


        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
 
        print("wine quality predictor using random forest model :")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        #mlflow.log_param("variance explained:", train_score)
        #mlflow.log_param("Test variance explained:", test_score)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # Calculate feature importance in random forest
        importances = regr.feature_importances_
        labels = data.columns
        feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
        feature_df = feature_df.sort_values(by='importance', ascending=False,)

        # image formatting
        axis_fs = 18 #fontsize
        title_fs = 22 #fontsize
        sns.set(style="whitegrid")

        ax = sns.barplot(x="importance", y="feature", data=feature_df)
        ax.set_xlabel('Importance',fontsize = axis_fs) 
        ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
        ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

        plt.tight_layout()
        plt.savefig("feature_importance_v1.png",dpi=120) 
        #mlflow.log_artifact("feature_importance.png") 
        plt.close()

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(regr, "model", registered_model_name="RandomForest Regressor")
        else:
            mlflow.sklearn.log_model(regr, "model")
    
    
    

    


