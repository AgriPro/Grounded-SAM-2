import os
import cv2
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from dataclasses import dataclass

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass
class RunConfig:
    model_version: str = "sam2"
    remote_dir: str = ""
    gaze_path: str = "gaze.csv"
    output_dir: str = "./output"
    init_frame: int = 0
    frame_step: int = 1
    save_images: bool = False
    save_every_n: int = 1
    out_fps: float = 0.0
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    VIDEO_PATH = "./scene_camera.mp4"
    TEXT_PROMPT = "monitor."
    OUTPUT_VIDEO_PATH = "./tracking_demo.mp4"
    SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
    SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
    PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = "configs/sam2/sam2_hiera_t.yaml"
    sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    #model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    #sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    boxes = [[380, 936, 670, 1122], [413, 238, 1148, 709]]
    class_names = ["monitor", "monitor"]


def draw_counter(frame, text, origin=(10, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    padding = 6

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin

    # white background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + padding),
        (255, 255, 255),
        -1
    )
    # black text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


def load_video_frames(configs: RunConfig):
    video_info = sv.VideoInfo.from_video_path(configs.VIDEO_PATH)  # get video info
    print(video_info)
    frame_generator = sv.get_video_frames_generator(configs.VIDEO_PATH, stride=1, start=0, end=None)

    # saving video to frames
    source_frames = Path(configs.SOURCE_VIDEO_FRAME_DIR)
    source_frames.mkdir(parents=True, exist_ok=True)

    with sv.ImageSink(
        target_dir_path=source_frames,
        overwrite=True,
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(configs.SOURCE_VIDEO_FRAME_DIR)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names


def run_tracker(configs: RunConfig):
    video_predictor = build_sam2_video_predictor(configs.model_cfg, configs.sam2_checkpoint)
    sam2_image_model = build_sam2(configs.model_cfg, configs.sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    frame_names = load_video_frames(configs)
    inference_state = video_predictor.init_state(video_path=configs.SOURCE_VIDEO_FRAME_DIR)

    ann_frame_idx = 0  # the frame index we interact with
    img_path = os.path.join(configs.SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
    #image_source, image = load_image(img_path)
    image_source = np.array(Image.open(img_path).convert('RGB'))
    h, w, _ = image_source.shape
    #boxes = torch.tensor([380, 936, 670, 1122]).unsqueeze(0)
    input_boxes = np.array(configs.boxes)
    class_names = configs.class_names
    image_predictor.set_image(image_source)
    OBJECTS = class_names
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    if not os.path.exists(configs.SAVE_TRACKING_RESULTS_DIR):
        os.makedirs(configs.SAVE_TRACKING_RESULTS_DIR)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(configs.OUTPUT_VIDEO_PATH, fourcc, 30, (w, h))
    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
    try:
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            frame_path = os.path.join(configs.SOURCE_VIDEO_FRAME_DIR, frame_names[out_frame_idx])
            img = cv2.imread(frame_path)
            masks = (out_mask_logits > 0.0).cpu().numpy()
            if masks.ndim == 4: # Handle (N, 1, H, W) if necessary
                masks = masks.squeeze(1)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks, # (n, h, w)
                class_id=np.array(out_obj_ids, dtype=np.int32),
            )
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, 
                            labels=[ID_TO_OBJECTS[i] for i in out_obj_ids])
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            out_video.write(annotated_frame)
            if out_frame_idx%500==0:
                torch.cuda.empty_cache()
    finally:
        out_video.release()
        video_predictor.reset_state(inference_state)
        print(f"Processing complete. Video saved to {configs.OUTPUT_VIDEO_PATH}")
