import boto3
from torchvision import datasets, transforms
import sagemaker
import argparse
import os

if __name__ == '__main__':
    bucket = os.environ['DATASETS_BUCKET']
    key_prefix = os.environ['DATASETS_KEY_PREFIX']

    sess = boto3.Session()
    if sess.region_name == "us-east-1":
        sess.client('s3').create_bucket(Bucket=bucket)
    else:
        sess.client('s3').create_bucket(Bucket=bucket, CreateBucketConfiguration={'LocationConstraint': sess.region_name})

    sagemaker.Session().upload_data(path='mnist', bucket=bucket, key_prefix=key_prefix)