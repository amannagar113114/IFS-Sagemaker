import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import datetime
import boto3
import sagemaker
from sagemaker import get_execution_role

# Set a good base job name when building different models
# It will help in identifying trained models and endpoints
with_categories = False
if with_categories:
    base_job_name = 'deepar-weather-new-with-categories'
else:
    base_job_name = 'deepar-weather-new-no-categories'
    
# Specify your bucket name
bucket = 'weather-timeseries-forecast'
prefix = 'deepar/weather'

# This structure allows multiple training and test files for model development and testing
if with_categories:
    s3_data_path = "{}/{}/data_with_categories".format(bucket, prefix)
else:
    s3_data_path = "{}/{}/data".format(bucket, prefix)
    

s3_output_path = "{}/{}/output".format(bucket, prefix)

# File name is referred as key name in S3
# Files stored in S3 are automatically replicated across
# three different availability zones in the region where the bucket was created.

def write_to_s3(filename, bucket, key):
    with open(filename,'rb') as f: # Read in binary mode
        return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(f)
# Upload one or more training files and test files to S3
if with_categories:
    write_to_s3('train_with_categories.json',bucket,'deepar/weather/data_with_categories/train/train_with_categories.json')
    write_to_s3('test_with_categories.json',bucket,'deepar/weather/data_with_categories/test/test_with_categories.json')
else:
    write_to_s3('train.json',bucket,'deepar/weather/data/train/train.json')
    write_to_s3('test.json',bucket,'deepar/weather/data/test/test.json')

sagemaker_session = sagemaker.Session()
role = get_execution_role()

# We no longer have to maintain a mapping of container images by region
# Simply use the convenience method provided by sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
image_name = get_image_uri(boto3.Session().region_name, 'forecasting-deepar')    
        
freq='H' # Timeseries consists Hourly Data and we need to predict hourly temp and humidity

# how far in the future predictions can be made
# 12 days worth of hourly forecast 
prediction_length = 288 

# aws recommends setting context same as prediction length as a starting point. 
# This controls how far in the past the network can see
context_length = 288    

# In this example, I am using ml.m5.xlarge for training
estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sagemaker_session,
    image_name=image_name,
    role=role,
    train_instance_count=1,
    train_instance_type='ml.m5.xlarge',
    base_job_name=base_job_name,
    output_path="s3://" + s3_output_path
)

hyperparameters = {
    "time_freq": freq,
    "epochs": "300",
    "early_stopping_patience": "40",
    "mini_batch_size": "64",
    "learning_rate": "5E-4",
    "context_length": str(context_length),
    "prediction_length": str(prediction_length),
    "cardinality" : "auto" if with_categories else ''
}

estimator.set_hyperparameters(**hyperparameters)

#Here, we are simply referring to train path and test path
# You can have multiple files in each path
# SageMaker will use all the files
data_channels = {
    "train": "s3://{}/train/".format(s3_data_path),
    "test": "s3://{}/test/".format(s3_data_path)
}

#starting the training
estimator.fit(inputs=data_channels)
job_name = estimator.latest_training_job.name
print ('job name: {0}'.format(job_name))

# Create an endpoint for real-time predictions
endpoint_name = sagemaker_session.endpoint_from_job(
    job_name=job_name,
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    deployment_image=image_name,
    role=role
)

print ('endpoint name: {0}'.format(endpoint_name))




       