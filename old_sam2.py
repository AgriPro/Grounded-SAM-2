import os
import cv2
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2_video_predictor, build_sam2
#from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from dataclasses import dataclass
from configs.defaults import RunConfig
from utils.file_utils import download_s3_folder
from utils.plot_utils import draw_counter
from utils.video_utils import create_video_from_images, load_video_frames
from utils.setup_utils import setup_environment
from utils.track_utils import masks_to_boxes


def main(configs: RunConfig):
    torch.autocast(device_type=configs.DEVICE, dtype=torch.bfloat16).__enter__()
    configs.VIDEO_PATH = os.path.join(configs.data_dir, configs.VIDEO_PATH)
    configs.gaze_path = os.path.join(configs.data_dir, configs.gaze_path)
    cap = cv2.VideoCapture(configs.VIDEO_PATH)
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)//configs.frame_stride
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    scale_factor = configs.max_width / max(frame_height, frame_width)
    scaled_height = int(frame_height * scale_factor)
    scaled_width = int(frame_width * scale_factor)
    configs.scale_factor = scale_factor
    cap.release()
    frame_times = [i/input_fps for i in range(frame_count)]
    df_gaze = pd.read_csv(configs.gaze_path)
    df_gaze['timestamps'] = pd.to_datetime(df_gaze['timestamps'])
    t0 = df_gaze['timestamps'].iloc[0]
    df_gaze['t_sec'] = (df_gaze['timestamps'] - t0).dt.total_seconds()
    gx = (np.interp(frame_times, df_gaze['t_sec'], df_gaze['gaze_x'])).astype(np.int32)  # adjust for resized frame
    gy = (np.interp(frame_times, df_gaze['t_sec'], df_gaze['gaze_y'])).astype(np.int32)  # adjust for resized frame
    gx = (gx * scale_factor).astype(np.int32)  # Scale the x-coordinate
    gy = (gy * scale_factor).astype(np.int32)

    video_predictor = build_sam2_video_predictor(configs.model_cfg, configs.sam2_checkpoint)
    grounding_model = load_model(
        model_config_path=configs.GROUNDING_DINO_CONFIG,
        model_checkpoint_path=configs.GROUNDING_DINO_CHECKPOINT,
        device=configs.DEVICE
    )
    #sam2_image_model = build_sam2(configs.model_cfg, configs.sam2_checkpoint)
    #image_predictor = SAM2ImagePredictor(sam2_image_model)
    frame_names = load_video_frames(configs)
    inference_state = video_predictor.init_state(video_path=configs.SOURCE_VIDEO_FRAME_DIR)
    if not os.path.exists(configs.SAVE_TRACKING_RESULTS_DIR):
        os.makedirs(configs.SAVE_TRACKING_RESULTS_DIR)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(configs.OUTPUT_VIDEO_PATH, fourcc, 30, (scaled_width, scaled_height))
    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    img_path = os.path.join(configs.SOURCE_VIDEO_FRAME_DIR, frame_names[configs.ann_frame_idx])

    if configs.boxes is not None and len(configs.boxes) > 0:
        input_boxes = np.array(configs.boxes)
        class_names = configs.class_names
    else:
        image_source, image = load_image(img_path)
        h, w, _ = image_source.shape
        boxes, confidences, class_names = predict(
            model=grounding_model,
            image=image,
            caption=configs.TEXT_PROMPT,
            box_threshold=configs.BOX_THRESHOLD,
            text_threshold=configs.TEXT_THRESHOLD,
            device=configs.DEVICE
        )
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidences = np.array(confidences)
    #image_predictor.set_image(image_source)
    OBJECTS = class_names
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    with torch.no_grad():
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=configs.ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    attention_tracker = 0
    attention_tracker_objects = {i: 0 for i in range(1, len(OBJECTS) + 1)}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        frame_path = os.path.join(configs.SOURCE_VIDEO_FRAME_DIR, frame_names[out_frame_idx])
        img = cv2.imread(frame_path)
        out_mask_logits = out_mask_logits.to(torch.bfloat16)
        masks = (out_mask_logits > 0.0).cpu().numpy()
        torch.cuda.empty_cache()
        if masks.ndim == 4: # Handle (N, 1, H, W) if necessary
            masks = masks.squeeze(1)
        mask_all_bool = masks > 0.0
        mask_bool = np.any(mask_all_bool, axis=0).astype(np.uint8)
        #plot it on the image
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks, # (n, h, w)
            class_id=np.array(out_obj_ids, dtype=np.int32),
        )
        annotated_frame = box_annotator.annotate(scene=img, detections=detections)
        mask_labels = [f"{ID_TO_OBJECTS[i]}: {attention_tracker_objects[i]}" for i in out_obj_ids]
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections,
                        labels=mask_labels)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        try:
            for i, obj_id in enumerate(out_obj_ids):
                obj_mask = masks[i]
                if obj_mask[gy[out_frame_idx], gx[out_frame_idx]] > 0:
                    attention_tracker_objects[obj_id] += 1
                    attention_tracker += 1
            draw_counter(annotated_frame, f"{attention_tracker}",origin=(10, 30))
            cv2.circle(annotated_frame, (gx[out_frame_idx], gy[out_frame_idx]), 8, (0,0,255), -1)
        except:
            print(f"gaze index out of bounds for {out_frame_idx}")
        out_video.write(annotated_frame)
        output_path = os.path.join(configs.SAVE_TRACKING_RESULTS_DIR, frame_names[out_frame_idx])
        cv2.imwrite(output_path, annotated_frame)

    out_video.release()
    # percentage of attention on each object
    for obj_id, count in attention_tracker_objects.items():
        print(f"{ID_TO_OBJECTS[obj_id]}: {count/frame_count*100:.2f}%")

if __name__ == "__main__":
    setup_environment()
    configs = RunConfig()
    if configs.data_dir.startswith("s3://"):
        # Handle S3 path
        configs.data_dir = configs.data_dir.rstrip("/ ")
        local_dir = download_s3_folder(configs.data_dir)
        configs.data_dir = local_dir
        configs.remote_dir = configs.data_dir
    main(configs)