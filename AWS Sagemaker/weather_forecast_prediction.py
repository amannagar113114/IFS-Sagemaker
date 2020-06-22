import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import datetime
import boto3
import sagemaker
from sagemaker import get_execution_role
import pandas as pd

# Provide endpoint
with_categories = False
endpoint_name = 'deepar-weather-new-no-categories-2020-06-16-20-18-09-415'

freq='H' # Timeseries consists Hourly Data and we need to predict hourly temp and humidity

# how far in the future predictions can be made
# 12 days worth of hourly forecast 
prediction_length = 288 

# aws recommends setting context same as prediction length as a starting point. 
# This controls how far in the past the network can see
context_length = 288


dt_predict_max = pd.Timestamp("2020-06-18 23:00:00", freq=freq) # 2012-12-31 23:00 alt way..pd.datetime(2012,12,31,23,0,0)

dt_dataset_start_time = pd.Timestamp("2018-06-19 00:00:00", freq=freq)
dt_dataset_end_time = pd.Timestamp("2020-06-06 23:00:00", freq=freq)

# use for model training
# Start time is the first row provided by kaggle
# Training TS end time ensures some data is withheld for model testing
# 12 days worth of training data is withheld for testing
dt_train_range = (dt_dataset_start_time,
                  dt_dataset_end_time - datetime.timedelta(hours=12*24) )

# Use entire data for testing
# We can compare predicted values vs actual (i.e. last 12 days is withheld for testing and model hasn't seen that data)
dt_test_range = (dt_dataset_start_time, 
                 dt_dataset_end_time) 

sagemaker_session = sagemaker.Session()
role = get_execution_role()

def encode_target(ts):
    return [x if np.isfinite(x) else "NaN" for x in ts]  

def encode_dynamic_feat(dynamic_feat):  
    l = []
    for col in dynamic_feat:
        assert (not dynamic_feat[col].isna().any()), col  + ' has NaN'             
        l.append(dynamic_feat[col].tolist())
    return l

def series_to_obj(ts, cat=None, dynamic_feat=None):
    obj = {"start": str(ts.index[0]), "target": encode_target(ts)}
    if cat is not None:
        obj["cat"] = cat
    if dynamic_feat is not None:
        obj["dynamic_feat"] = encode_dynamic_feat(dynamic_feat)
    return obj

def series_to_jsonline(ts, cat=None, dynamic_feat=None):

class DeepARPredictor(sagemaker.predictor.RealTimePredictor):

    def set_prediction_parameters(self, freq, prediction_length):
        """Set the time frequency and prediction length parameters. This method **must** be called
        before being able to use `predict`.
        
        Parameters:
        freq -- string indicating the time frequency
        prediction_length -- integer, number of predicted time points
        
        Return value: none.
        """
        self.freq = freq
        self.prediction_length = prediction_length
        
    def predict(self, ts, cat=None, dynamic_feat=None, 
                encoding="utf-8", num_samples=100, quantiles=["0.1", "0.5", "0.9"]):
        """Requests the prediction of for the time series listed in `ts`, each with the (optional)
        corresponding category listed in `cat`.
        
        Parameters:
        ts -- list of `pandas.Series` objects, the time series to predict
        cat -- list of integers (default: None)
        encoding -- string, encoding to use for the request (default: "utf-8")
        num_samples -- integer, number of samples to compute at prediction time (default: 100)
        quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])
        
        Return value: list of `pandas.DataFrame` objects, each containing the predictions
        """
        #prediction_times = [x.index[-1]+1 for x in ts]
        prediction_times = [x.index[-1] + datetime.timedelta(hours=1) for x in ts] 
        
        req = self.__encode_request(ts, cat, dynamic_feat, encoding, num_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, prediction_times, encoding)
    
    def __encode_request(self, ts, cat, dynamic_feat, encoding, num_samples, quantiles):
        
        instances = [series_to_obj(ts[k], 
                                   cat[k] if cat else None,
                                   dynamic_feat if dynamic_feat else None) 
                     for k in range(len(ts))]
        
        configuration = {"num_samples": num_samples, "output_types": ["quantiles"], "quantiles": quantiles}
        http_request_data = {"instances": instances, "configuration": configuration}
        return json.dumps(http_request_data).encode(encoding)
    
    def __decode_response(self, response, prediction_times, encoding):
        response_data = json.loads(response.decode(encoding))
        list_of_df = []
        for k in range(len(prediction_times)):
            #prediction_index = pd.DatetimeIndex(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            prediction_index = pd.date_range(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            list_of_df.append(pd.DataFrame(data=response_data['predictions'][k]['quantiles'], index=prediction_index))
        return list_of_df

predictor = DeepARPredictor(
    endpoint=endpoint_name,
    sagemaker_session=sagemaker_session,
    content_type="application/json"
)
predictor.set_prediction_parameters(freq, prediction_length)



df = pd.read_csv('all_data_weather.csv', parse_dates=['datetime'],index_col=0)
df_test1 = pd.read_csv('/home/ec2-user/SageMaker/dataset/test.csv', parse_dates=['datetime'],index_col=0)
df_test = df_test1[['season', 'weather', 'windspeed']]
df = df.resample('1h').mean()

target_values = ['temp', 'humidity']
time_series_test = []
time_series_training = []

for t in target_values:
    time_series_test.append(df[dt_test_range[0]:dt_test_range[1]][t])
    time_series_training.append(df[dt_train_range[0]:dt_train_range[1]][t])

# Provide 0 based index for categories
list_of_df = predictor.predict(time_series_training,
                               cat=[[0],[1],[2]] if with_categories else None)
                               
#plot graph for visualisation
for k in range(len(list_of_df)):
    # print (-prediction_length-context_length) #120 = 72+48
    plt.figure(figsize=(12,6))
    
    time_series_test[k][-prediction_length-context_length:].plot(label='target')
    
    p10 = list_of_df[k]['0.1']
    p90 = list_of_df[k]['0.9']
    plt.fill_between(p10.index, p10, p90, color='y', alpha=0.5, label='80% confidence interval')
    list_of_df[k]['0.5'].plot(label='prediction median')
    plt.legend()
    plt.show()

predict_window = []
for i,x in df_test.groupby([df_test.index.year,df_test.index.month]):
    predict_window.append(x.index.min()-datetime.timedelta(hours=1))

for t in target_values:
    df_test[t] = np.nan
for window in predict_window:
    print(window)
    # If trained with categories, we need to send corresponding category for each training set
    # In this case
    for i in range(len(target_values)):
        list_of_df = predictor.predict([time_series_test[i][:window]],
                                       cat=[i] if with_categories else None)
        df_tmp = list_of_df[0]
        df_tmp.index.name = 'datetime'
        df_tmp.columns = ['0.1',target_values[i],'0.9']
        df_test.update(df_tmp[target_values[i]])
# Store the results
if with_categories:
    df_test[['temp', 'humidity']].to_csv('weather-prediction-with-categories.csv',index=True,index_label='datetime')
else:
    df_test[['temp', 'humidity']].to_csv('weather-prediction.csv',index=True,index_label='datetime')

df_main = pd.read_excel('/home/ec2-user/SageMaker/dataset/Disease_Weather_Forecast.xlsx', index_col='date_time', parse_dates=['date_time'])
df_input=df_main['2014-01-29':][['temp', 'humidity']]
#df = df.resample('1D').mean
target_values = ['temp','humidity']
time_series_test = []

for t in target_values:
    time_series_test.append(df_input[t])
    
    
# Provide 0 based index for categories
list_of_df = predictor.predict(time_series_test, None)

df_prediction = pd.DataFrame()
target_values = ['temp','humidity']
time_series_test = []

for t in target_values:
    time_series_test.append(df_input[t])

for i in range(len(target_values)):
    list_of_df = predictor.predict([time_series_test[i]], None)
    df_tmp = list_of_df[0]
    df_tmp.index.name = 'datetime'
    df_tmp.columns = ['0.1',target_values[i],'0.9']
    df_prediction[target_values[i]] = df_tmp[target_values[i]]


import random
def conditions(predicted_df):
    if (predicted_df['temp']>=15) & (predicted_df['temp']<=23):
        return round(random.uniform(4.31, 5.0),2)
    elif (predicted_df['temp']>23) & (predicted_df['temp']<=27):
        return round(random.uniform(0.0, 3.0),2)
    elif (predicted_df['temp']>27) & (predicted_df['temp']<=35):
        return round(random.uniform(3.01, 4.3),2)
    else:
        return round(random.uniform(4.76, 5.0),2)
   
df_prediction['predicted_disease_incidence'] = df_prediction.apply(conditions, axis=1)


def condi(predicted_df):
    if (predicted_df['predicted_disease_incidence'] <= 2) :
        return 'No Disease'
    elif (predicted_df['predicted_disease_incidence']>2) & (predicted_df['predicted_disease_incidence']<=3):
        return 'Downy Mildews'
    elif (predicted_df['predicted_disease_incidence']>3) & (predicted_df['predicted_disease_incidence']<=3.9):
        return 'Bacterial Wilt'
    elif (predicted_df['predicted_disease_incidence']>3.9) & (predicted_df['predicted_disease_incidence']<=4.3):
        return 'Leaf Blight'
    elif (predicted_df['predicted_disease_incidence']>4.3) & (predicted_df['predicted_disease_incidence']<=4.75):
        return 'Basal Rot'
    elif (predicted_df['predicted_disease_incidence']>4.75) & (predicted_df['predicted_disease_incidence']<=4.85):
        return 'Aster Yellows'
    else:
        return 'Powdery Mildews'
  
df_prediction['predicted_disease'] = df_prediction.apply(condi, axis=1)

# Store the results
df_prediction.to_csv('DeepAR_prediction.csv',index=True,index_label='date_time')







    