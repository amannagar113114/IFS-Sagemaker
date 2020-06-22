import json
import numpy as np
import boto3
import os
import botocore

#Give the endpoint name in Environment variable 
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime = boto3.Session().client(service_name='sagemaker-runtime')
bucket = 'new-plant-disease-dataset'
key = 'training-and-validation/train/Tomato_Bacterial_spot/03707c27-7f95-4f19-a173-1c0c74653fdf___GCREC_Bact.Sp 3046.JPG'
bucket = 'sumerian-apptest'
key = 'aman_path/images.jfif'
object_categories = ['Not_a_leaf','Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy', 'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

def lambda_handler(event, context):
    os.chdir('/tmp/')
    session = boto3.session.Session(region_name='us-east-2')
    s3 = session.client('s3')
    #response = s3client.get_object(Bucket='new-plant-disease-dataset', Key='output-model/pepperbellbacterialspot.jpg')
    s3.download_file(bucket, key, '/tmp/image.jpg')
    res='/tmp/image.jpg'
    with open(res, 'rb') as f:
        payload = f.read()
        final_payload = bytearray(payload)
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/x-image', Body=final_payload)
    result = response['Body'].read()
    result = json.loads(result)
    pred_label_id = np.argmax(result)
    probability=str(result[pred_label_id])
    a=(result[pred_label_id])
    b=a*100
    return {'statusCode': 200,'body': json.dumps("Predicting image as - '" + object_categories[pred_label_id] + "' with surety of percentage - " + str(b))}
