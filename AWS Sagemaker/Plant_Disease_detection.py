%%time
import sagemaker
from sagemaker import get_execution_role
import mxnet
from sagemaker.amazon.amazon_estimator import get_image_uri
import os
import urllib.request

role = get_execution_role()
print(role)

sess = sagemaker.Session()
bucket='new-plant-disease-dataset' 
prefix = 'training-and-validation'

training_image = get_image_uri(sess.boto_region_name, 'image-classification', repo_version="latest")
print(training_image)

#this function is to download file from s3 to local notebook instance
def downloadDirectoryFroms3(bucketName,remoteDirectoryName):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName) 
    for object in bucket.objects.filter(Prefix = remoteDirectoryName):
        if not os.path.exists(os.path.dirname(object.key)):
            os.makedirs(os.path.dirname(object.key))
        bucket.download_file(object.key,object.key)
		
#downloadDirectoryFroms3('new-plant-disease-dataset','training-and-validation/Testing/To upload Sumerian new/')

%%bash
python im2rec.py --list --recursive plant-disease-train Plantvillage-train/
python im2rec.py --list --recursive plant-disease-val Plantvillage-test/

# Four channels: train, validation, train_lst, and validation_lst
s3train = 's3://{}/{}/train/'.format(bucket, prefix)
s3validation = 's3://{}/{}/validation/'.format(bucket, prefix)
s3train_lst = 's3://{}/{}/train_lst/'.format(bucket, prefix)
s3validation_lst = 's3://{}/{}/validation_lst/'.format(bucket, prefix)

# upload the image files to train and validation channels
!aws s3 cp Plantvillage-train $s3train --recursive --quiet
!aws s3 cp Plantvillage-test $s3validation --recursive --quiet

# upload the lst files to train_lst and validation_lst channels
!aws s3 cp plant-disease-train.lst $s3train_lst --quiet
!aws s3 cp plant-disease-val.lst $s3validation_lst --quiet


%%bash
python im2rec.py --resize 256 --quality 90 --num-thread 16 plant-disease-val Plantvillage-test/
python im2rec.py --resize 256 --quality 90 --num-thread 16 plant-disease-train Plantvillage-train/


s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)
ic = sagemaker.estimator.Estimator(training_image,
                                         role, 
                                         train_instance_count=1, 
                                         train_instance_type='ml.p2.xlarge',
                                         train_volume_size = 50,
                                         train_max_run = 360000,
                                         input_mode= 'File',
                                         output_path=s3_output_location,
                                         sagemaker_session=sess)

#define hyperparameters										
										
ic.set_hyperparameters(num_layers=18,
                             use_pretrained_model=1,
                             image_shape = "3,224,224",
                             num_classes=16,
                             mini_batch_size=128,
                             epochs=21,
                             learning_rate=0.1,
                             top_k=2,
                             num_training_samples=17675,
                             resize = 256,
                             early_stopping=True,
                             early_stopping_patience = 5,
                             precision_dtype='float32')

train_data = sagemaker.session.s3_input(s3train, distribution='FullyReplicated', 
                        content_type='application/x-image', s3_data_type='S3Prefix')
validation_data = sagemaker.session.s3_input(s3validation, distribution='FullyReplicated', 
                             content_type='application/x-image', s3_data_type='S3Prefix')
train_data_lst = sagemaker.session.s3_input(s3train_lst, distribution='FullyReplicated', 
                        content_type='application/x-image', s3_data_type='S3Prefix')
validation_data_lst = sagemaker.session.s3_input(s3validation_lst, distribution='FullyReplicated', 
                             content_type='application/x-image', s3_data_type='S3Prefix')

data_channels = {'train': train_data, 'validation': validation_data, 
                 'train_lst': train_data_lst, 'validation_lst': validation_data_lst}
				 
#start training
ic.fit(inputs=data_channels, logs=True)

#deploy the model
ic_classifier = ic.deploy(initial_instance_count = 1,
                                          instance_type = 'ml.t2.medium')
										  
#test few random disease picture on notebook instance
import json
import numpy as np

#file_name = '/home/ec2-user/SageMaker/Plantvillage-train/Tomato__Tomato_mosaic_virus/021accd9-bbb2-4777-8f94-93295e6de49e___PSU_CG 2075.JPG'

with open(file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)
    
ic_classifier.content_type = 'application/x-image'
result = json.loads(ic_classifier.predict(payload))

index = np.argmax(result)
object_categories = ['Not_a_leaf','Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___healthy','Potato___Late_blight','Tomato__Target_Spot','Tomato__Tomato_mosaic_virus','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_healthy','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite']
#It will predict the diseased image with the percentage of accuracy
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))