import os
import io
import boto3
import json
import csv
import pandas as pd
import random
import datetime
import numpy as np
import random
from io import StringIO
 
# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

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
    return json.dumps(series_to_obj(ts, cat, dynamic_feat))  
    
class DeepARPredictor():

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
        #res = super(DeepARPredictor, self).predict(req)
        res = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Body=req)
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
        #response_data = json.loads(response.decode(encoding))
        response_data =response['Body'].read().decode("utf-8")
        response_data = json.loads(response_data)
        #print(response_data)
        list_of_df = []
        for k in range(len(prediction_times)):
            #prediction_index = pd.DatetimeIndex(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            prediction_index = pd.date_range(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            list_of_df.append(pd.DataFrame(data=response_data['predictions'][k]['quantiles'], index=prediction_index))
        return list_of_df
        
def conditions(predicted_df):
    if (predicted_df['temp']>=15) & (predicted_df['temp']<=23):
        return round(random.uniform(4.31, 5.0),2)
    elif (predicted_df['temp']>23) & (predicted_df['temp']<=27):
        return round(random.uniform(0.0, 3.0),2)
    elif (predicted_df['temp']>27) & (predicted_df['temp']<=35):
        return round(random.uniform(3.01, 4.3),2)
    else:
        return round(random.uniform(4.76, 5.0),2)

def calculate_incidence(temp):
    if (temp>=15) & (temp<=23):
        return round(random.uniform(4.31, 5.0),2)
    elif (temp>23) & (temp<=27):
        return round(random.uniform(0.0, 3.0),2)
    elif (temp>27) & (temp<=35):
        return round(random.uniform(3.01, 4.3),2)
    else:
        return round(random.uniform(4.76, 5.0),2)
        
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

def predict_disease(incidence):
    if (incidence <= 2) :
        return 'No Disease'
    elif (incidence>2) & (incidence<=3):
        return 'Downy Mildews'
    elif (incidence>3) & (incidence<=3.9):
        return 'Bacterial Wilt'
    elif (incidence>3.9) & (incidence<=4.3):
        return 'Leaf Blight'
    elif (incidence>4.3) & (incidence<=4.75):
        return 'Basal Rot'
    elif (incidence>4.75) & (incidence<=4.85):
        return 'Aster Yellows'
    else:
        return 'Powdery Mildews'

predictor = DeepARPredictor()
    
predictor.set_prediction_parameters('H', 288)

def lambda_handler(event, context):

    df_input = pd.read_csv('s3://weather-pred/testingsanjeet.csv' , index_col='date_time', parse_dates=['date_time'])
    target_values = ['temp','humidity']
    time_series_test = []
    for t in target_values:
        time_series_test.append(df_input[t])
    #print(time_series_test)

    df_prediction = pd.DataFrame()
    df_prediction_final=pd.DataFrame()
    for i in range(len(target_values)):
        list_of_df = predictor.predict([time_series_test[i]], None)
        df_tmp = list_of_df[0]
        #print(df_tmp)
        df_tmp.index.name = 'date_time' 
        df_tmp.columns = ['0.1',target_values[i],'0.9']
        df_prediction[target_values[i]] = df_tmp[target_values[i]]
    
    #print(df_prediction)
    df_prediction=df_prediction.mean()
    #print(df_prediction)
    Temperature=df_prediction[['temp']][0]
    Temperature=round(Temperature,3)
    Humidity=df_prediction[['humidity']][0]
    Humidity=round(Humidity,3)
    disease_incidence=calculate_incidence(Temperature)
    #print(disease_incidence)
    predicted_disease_name=predict_disease(disease_incidence)
    
    csv_buffer = StringIO()    
    csv_buffer = df_prediction.to_csv(index=False)
    client = boto3.client('s3')
    client.put_object(Body = csv_buffer, Bucket = 'weather-timeseries-forecast' , Key = 'deepar/weather/output/predicted.csv')
    return {'Average_Temperature': Temperature ,
            'Average_Humidity': Humidity,
            'Predicted_Disease':predicted_disease_name
            
        }

