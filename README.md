# YOLOv8 Person Detection

This project utilizes the YOLOv8 object detection model to detect and count people in a given video or live stream. It employs the Ultralytics YOLO library, which is based on the YOLOv8 models.

## Installation

1. Clone the repository:

```
git clone https://github.com/Deepchavda007/People-Count-using-YOLOv8.git
```

2.Install the required Python packages:

```
pip install -r requirements.txt
```


## Model:
The YOLOv8 model used in this project is yolov8x.pt. Make sure to download and place the model file in the project directory.

### Usage
Prepare the video or live stream:

For a video file: Place the video file in the project directory.

For a live stream: Update the camera_ip variable in the code with the appropriate RTSP URL.
Update the COCO class file:

Open the coco.txt file in the project directory.
Replace the contents with the class names in the format class_name.
Run the application:

```
python main.py
```

The application will open a window showing the video stream with people bounding boxes and counts. Press Esc to exit.

The final output video will be saved as Final_output.mp4 in the project directory.

## Performance
The application uses object tracking and centroid-based counting to track people and count their entry and exit. The counting results are displayed in the video window and logged to the console.



