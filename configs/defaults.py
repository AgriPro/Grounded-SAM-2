from dataclasses import dataclass
import torch

@dataclass
class RunConfig:
    model_version: str = "sam2"
    remote_dir: str = ""
    data_dir: str = "./data"
    gaze_path: str = "gaze.csv"
    output_dir: str = "./output"
    init_frame: int = 0
    frame_step: int = 1
    save_images: bool = False
    save_every_n: int = 1
    out_fps: float = 0.0
    max_width: int = 512
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "./gdino_checkpoints/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    video_path = "scene_camera.mp4"
    TEXT_PROMPT = "monitor."
    output_video_path = "attention_output.mp4"
    SOURCE_VIDEO_FRAME_DIR = "custom_video_frames"
    SAVE_TRACKING_RESULTS_DIR = "tracking_results"
    PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = "configs/sam2/sam2_hiera_t.yaml"
    sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    #model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    #sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    boxes = None
    class_names = None
    scale_factor: int = 0
    frame_stride: int = 3
    ann_frame_idx: int = 30
    temp_dir: str = None
