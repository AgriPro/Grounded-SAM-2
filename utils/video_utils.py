import cv2
import os
from tqdm import tqdm
import supervision as sv
from pathlib import Path

def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")


def load_video_frames(configs):
    cap = cv2.VideoCapture(configs.video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) //configs.frame_stride
    cap.release()
    video_info = sv.VideoInfo.from_video_path(configs.video_path)  # get video info
    print(video_info)
    h, w = video_info.height, video_info.width
    scale_factor = configs.max_width / max(h, w)
    frame_generator = sv.get_video_frames_generator(configs.video_path, stride=configs.frame_stride, start=0, end=None)

    # saving video to frames
    source_frames = Path(configs.SOURCE_VIDEO_FRAME_DIR)
    source_frames.mkdir(parents=True, exist_ok=True)

    with sv.ImageSink(
        target_dir_path=source_frames,
        overwrite=True,
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for idx, frame in tqdm(enumerate(frame_generator), desc="Saving Video Frames", total=frame_count):
            resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            sink.save_image(resized_frame)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(configs.SOURCE_VIDEO_FRAME_DIR)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names
