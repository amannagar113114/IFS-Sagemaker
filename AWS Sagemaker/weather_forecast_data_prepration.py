import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import datetime


#This function is to download files or folder from s3 to local notebook instance
def downloadDirectoryFroms3(bucketName,remoteDirectoryName):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName)
    for object in bucket.objects.filter(Prefix = remoteDirectoryName):
        if not os.path.exists(os.path.dirname(object.key)):
            os.makedirs(os.path.dirname(object.key))
        bucket.download_file(object.key,object.key)
		
#downloadDirectoryFroms3('weather-timeseries-forecast','deepar/train.csv')

target_values = ['temp','humidity']


# included in the training and test data
with_categories = False

# Set datetime column as index to work with data based on Date/Time
df1 = pd.read_csv('/home/ec2-user/SageMaker/dataset/train.csv', parse_dates=['datetime'],index_col=0)
df_test1 = pd.read_csv('/home/ec2-user/SageMaker/dataset/test.csv', parse_dates=['datetime'],index_col=0)

df1 = pd.read_csv('/home/ec2-user/SageMaker/dataset/train.csv', parse_dates=['datetime'],index_col=0)
df = df1[['season', 'weather', 'temp', 'humidity', 'windspeed']]
df_test = df_test1[['season', 'weather', 'windspeed']]

# We need to let DeepAR know how far in the future predictions can be made using prediction_length hyperparameter
# Let's look at how many hours we need to predict in a month using test.csv data file
hours_to_predict = []
print ('Check maximum hours we need to predict')
# Group by year,month
predict_window = df_test.groupby([df_test.index.year,df_test.index.month])
for i,x in predict_window:
    delta = x.index.max() - x.index.min() 
    hours = np.ceil(delta.total_seconds()/3600)
    hours_to_predict.append(hours)
    print ("{0}, Hours:{1}".format(i, hours))

print ("Maximum Prediction Length in Hours: ", np.max(hours_to_predict))



# https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_hyperparameters.html
    
freq='H' # Timeseries consists Hourly Data and we need to predict hourly temp and humidity

# how far in the future predictions can be made
# 12 days worth of hourly forecast 
prediction_length = 288 

# aws recommends setting context same as prediction length as a starting point. 
# This controls how far in the past the network can see
context_length = 288



import pandas as pd
dt_predict_max = pd.Timestamp("2020-06-18 23:00:00", freq=freq) # 2012-12-31 23:00 alt way..pd.datetime(2012,12,31,23,0,0)

dt_dataset_start_time = pd.Timestamp("2018-06-19 00:00:00", freq=freq)
dt_dataset_end_time = pd.Timestamp("2020-06-06 23:00:00", freq=freq)

# use for model training
# Training TS end time ensures some data is withheld for model testing
# 12 days worth of training data is withheld for testing
dt_train_range = (dt_dataset_start_time,
                  dt_dataset_end_time - datetime.timedelta(hours=12*24) )

# Use entire data for testing
# We can compare predicted values vs actual (i.e. last 12 days is withheld for testing and model hasn't seen that data)
dt_test_range = (dt_dataset_start_time, 
                 dt_dataset_end_time) 
				 
#dt_predict_max,dt_predict_max + datetime.timedelta(hours=1)

# Let's see if there are gaps in timesteps
def is_missing_steps(df,start,end,freq='D'):
    dt_range = pd.date_range(start=start,end=end,freq=freq)
    return not dt_range.equals(df[start:end].index)

def get_missing_steps(df,start,end,freq='D'):
    dt_range = pd.date_range(start=start,end=end,freq=freq)
    return dt_range.difference(df[start:end].index)    

# List timeseries with only NaNs
# They can be removed
def timeseries_with_only_nans(df):
    l = []
    for col in df.columns:
        if pd.isna(df[col].min()):
            #print (col)
            l.append(col)
    return l
	
time_series_test = []
time_series_training = []

for target in target_values:
    time_series_test.append(df[dt_test_range[0]:dt_test_range[1]][target])
    time_series_training.append(df[dt_train_range[0]:dt_train_range[1]][target])
	
#plot graph
time_series_test[0].plot(label='test')
time_series_training[0].plot(label='train')#, ls=':')
plt.legend()
plt.show()


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
	
encoding = "utf-8"
cat_idx = 0

train_file_name = "train.json"
test_file_name = "test.json"

if with_categories:
    train_file_name = "train_with_categories.json"
    test_file_name = "test_with_categories.json"

with open(train_file_name, 'wb') as fp:
    for ts in time_series_training:
        fp.write(series_to_jsonline(ts,[cat_idx] if with_categories else None).encode(encoding))
        fp.write('\n'.encode(encoding))
        cat_idx += 1
		
cat_idx = 0
with open(test_file_name, 'wb') as fp:
    for ts in time_series_test:
        fp.write(series_to_jsonline(ts,[cat_idx] if with_categories else None).encode(encoding))
        fp.write('\n'.encode(encoding))
        cat_idx += 1
		
df.to_csv('all_data_weather.csv',index=True,index_label='datetime')