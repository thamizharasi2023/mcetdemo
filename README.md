Real-Time Object Detection with YOLO and Streamlit

This project is a real-time object detection web application built with Streamlit and YOLO (You Only Look Once) for detecting various objects using a webcam feed. It utilizes OpenCV for video processing, and the Ultralytics YOLO model for object detection.
Features

    Real-Time Detection: Streams live from a webcam and performs real-time object detection.
    YOLO Integration: Uses YOLO for detecting multiple objects in each frame.
    Customizable Labels: Displays labels for detected objects along with confidence scores.
    Streamlit Interface: Provides an interactive UI with object classes information.

Requirements

To run this application, you need the following Python libraries:

    streamlit
    ultralytics
    opencv-python
    numpy
    Pillow

You can install the required dependencies using:

bash

pip install -r requirements.txt

Usage
Step 1: Clone the Repository

Clone this repository to your local machine:

bash

git clone https://github.com/thamizharasi2023/mcetdemo.git
cd your-repository

Step 2: Set Up YOLO Weights

Make sure the YOLO model weights file (yolo11n.pt) is in the same directory as the script, or specify the correct path in the code.
Step 3: Run the Streamlit App

Run the application using the following command:

bash

streamlit run sample.py

Step 4: Access the App

After running the command, Streamlit will launch a local web server. Open the provided local URL in your web browser to access the app.
How It Works

    Webcam Feed: The app captures video input from your webcam.
    Object Detection: Each frame is processed by YOLO to detect objects.
    Annotated Output: Bounding boxes and labels are added to detected objects.
    Display in Streamlit: The processed video stream is displayed in the Streamlit interface.

Functions

    process_frame: Processes each video frame with YOLO and overlays bounding boxes and labels.
    plot_bboxes_1: Annotates frames by drawing bounding boxes and labels for detected objects.
    box_label: Draws the bounding box and label for each detected object on the frame.

Detectable Objects

The application supports a variety of object classes, such as people, vehicles, animals, and everyday items. A full list of detectable classes is displayed within the app.
Demo

Upon running, the app will display the live webcam feed with YOLO-detected objects highlighted in real time.
