# server.py (abbrev)
import json
import os
import shutil
from fastapi import FastAPI, BackgroundTasks, Request
import boto3
import uuid
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from urllib.parse import urlparse

import torch
from utils.log_utils import setup_logger
from configs.defaults import RunConfig
from utils.setup_utils import setup_environment
from sam2_multi_object import process_video

logger = setup_logger(__name__)
app = FastAPI()
s3 = boto3.client("s3")
is_processing = False

class InferenceRequest(BaseModel):
    s3_dir: str
    point_prompt: Optional[List[int]] = Field(None, min_items=2, max_items=2)
    box_prompt: Optional[List[int]] = Field(None, min_items=4, max_items=4)
    class_names: Optional[List[str]] = None

@app.get("/ping")
def ping():
    return {"status": "ok"}

def upload_to_s3(file_path: str, bucket_name: str, object_name: str):
    """
    Upload a file to an S3 bucket.
    """
    s3 = boto3.client("s3")
    s3.upload_file(file_path, bucket_name, object_name)

def download_from_s3(s3_uri, local_path):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    object_path = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    s3.download_file(bucket, object_path, local_path)
    return local_path

def handle_video_job(payload: InferenceRequest, tmpdir: str):
    """The actual heavy lifting happens here, off the main thread."""
    global is_processing
    try:
        load_dotenv()
        
        # 1. Download Logic (Using your existing functions)
        s3_video_uri = os.path.join(payload.s3_dir, "scene_camera.mp4")
        s3_gaze_uri = os.path.join(payload.s3_dir, "gaze.csv")
        
        local_gaze = download_from_s3(s3_gaze_uri, os.path.join(tmpdir, "gaze.csv"))
        local_video = download_from_s3(s3_video_uri, os.path.join(tmpdir, "scene_camera.mp4"))
        
        # 2. Setup Configs
        configs = RunConfig()
        configs.video_path = local_video
        configs.gaze_path = local_gaze
        configs.boxes = payload.box_prompt
        configs.class_names = payload.class_names
        configs.temp_dir = tmpdir
        configs.output_video_path = os.path.join(tmpdir, configs.output_video_path)
        configs.SOURCE_VIDEO_FRAME_DIR = os.path.join(tmpdir, configs.SOURCE_VIDEO_FRAME_DIR)
        configs.SAVE_TRACKING_RESULTS_DIR = os.path.join(tmpdir, configs.SAVE_TRACKING_RESULTS_DIR)

        # 3. Processing
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            attention_over_time, attention_tracker_objects = process_video(configs)

        # 4. Upload Logic
        parsed = urlparse(payload.s3_dir)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        s3_object_path = os.path.join(prefix, "attention_output.mp4")

        upload_to_s3(configs.output_video_path, bucket, s3_object_path)
        logger.info(f"Successfully uploaded result to {s3_object_path}")
        results_data = {
            "attention_tracker_objects": attention_tracker_objects,
            "attention_over_time": attention_over_time.tolist(),
            "metadata": {
                "s3_input": payload.s3_dir,
                "frame_count": len(attention_over_time)
            }
        }
        results_json_path = os.path.join(tmpdir, "attention_results.json")
        with open(results_json_path, "w") as f:
            json.dump(results_data, f)
        s3_results_object_path = os.path.join(prefix, "attention_results.json")
        upload_to_s3(results_json_path, bucket, s3_results_object_path)

    except Exception as e:
        logger.error(f"Job failed in background: {str(e)}")
    finally:
        is_processing = False  # Always release the lock
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.post("/invocations")
async def invocations(payload: InferenceRequest, background_tasks: BackgroundTasks):
    global is_processing    
    # 1. Check if GPU is already busy
    if is_processing:
        return {
            "status": "busy", 
            "message": "Server is currently processing another request. Please try again later."
        }

    # 2. Mark as busy and prepare workspace
    is_processing = True
    tmpdir = f"/tmp/job_{uuid.uuid4().hex}"
    os.makedirs(tmpdir, exist_ok=True)
    s3_video_uri = os.path.join(payload.s3_dir, "attention_output.mp4")

    # 3. Hand off the VALIDATED Pydantic object to the background
    background_tasks.add_task(handle_video_job, payload, tmpdir)
    
    # 4. Return immediately to satisfy SageMaker's 60s timeout
    return {
            "status": "submitted", 
            "job_id": tmpdir,
            "message": f"Processing started. Results will be uploaded to {s3_video_uri} upon completion.",
            "output_video_s3_path": s3_video_uri
    }
