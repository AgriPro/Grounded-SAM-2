import os
import boto3
import tempfile
from botocore.exceptions import NoCredentialsError

def download_s3_folder(data_dir: str) -> str:
    """
    Download a folder from S3 and return the local path.
    """
    _, _, bucket, *prefix_parts = data_dir.split("/")
    prefix = "/".join(prefix_parts).strip()
    s3 = boto3.client("s3")
    temp_dir = tempfile.mkdtemp()
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if "Contents" not in response:
            raise ValueError("No files found in the specified S3 folder.")

        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith(".csv"):  # only download CSVs
                local_file = os.path.join(temp_dir, os.path.basename(key))
                s3.download_file(bucket, key, local_file)
        return temp_dir
    except NoCredentialsError:
        raise ValueError("AWS credentials not found. Please configure your AWS credentials to access S3.")
    except Exception as e:
        raise ValueError(f"Error downloading from S3: {e}")