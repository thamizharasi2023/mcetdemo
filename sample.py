import ultralytics
from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image

import time

def process_frame(frame,output_image):
    global model
    results = model(frame)  # Run YOLO on the frame
      # Annotate frame with bounding boxes
    output_image =plot_bboxes_1(output_image, results[0].boxes, model, score=True, conf=None, show_box=True, side="left" )
    
    return output_image


def plot_bboxes_1(image, boxes, model, score=True, conf=None, show_box=True, side="left"):
    # Define COCO Labels and colors if not provided
    labels = model.names
    
    # Plot each box and update the result dictionary
    for box in boxes:
            
        box_2d = box.data
        #print(box_2d)
        label_index = int(box.cls.item())
        label_name = labels[label_index]
        conf_value = str(round(100 * float(box.conf.item()), 1)) + "%" if score else ""
        label_with_conf = f"{label_name} {conf_value}" if score else label_name
            

        # Filter every box under conf threshold if conf threshold set
        if conf is not None and box.conf.item() < conf:
            continue

        # Draw box with label and color using selected box_label function
        color = (128, 128, 128)  # You can customize the color here
        
        box_1d = box_2d.view(-1)
        #print(box_1d)
        image=box_label(image, box_1d, label_with_conf, color, show_box=show_box, side=side)

    # Convert image to PIL format and return along with updated result dictionary and list
    im_pil = Image.fromarray(image)
    
    return im_pil
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), show_box=True, side='left'):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    #print(f" image flags =  {image.flags}")
 
    p1 = (int(box[0].item()), int(box[1].item()))
    p2 = (int(box[2].item()), int(box[3].item()))
    #print(f" p1  =  {p1} p2 =  {p2}")
    #print(type(p1))
    image_copy = image.copy()
    
    if show_box:
        cv2.rectangle(image_copy, p1, p2, color, thickness=2)
        
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        
        if side == 'left':
            label_rect_p1 = (p1[0], p1[1] - h - 3)
            label_rect_p2 = (p1[0] + w, p1[1] - 3)
        else:
            label_rect_p1 = (p2[0] - w, p1[1] - h - 3)
            label_rect_p2 = (p2[0], p1[1] - 3)
            
        cv2.rectangle(image_copy, label_rect_p1, label_rect_p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image_copy,
                    label, (label_rect_p1[0] + 2, label_rect_p2[1] - 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return image_copy
def main():
    st.title('Web Application')
    
    video_stream = cv2.VideoCapture("https://videos.pexels.com/video-files/855564/855564-hd_1920_1080_24fps.mp4")
    #video_stream = cv2.VideoCapture("./sampletest1.mp4") 
    global model
    model = YOLO("yolo11n.pt")
    class_names= model.names
    st.sidebar.info("### Detectable Objects:\n" + ", ".join(class_names.values()))
    
    pred = st.empty()
    st.success('Press Stop button on the top right ')
    # Check if video stream is opened successfully
    if not video_stream.isOpened():
            st.error("Error accessing webcam.")
    else:
        # Run the detection in real-time
        while True:
            ret, frame = video_stream.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            output_image=frame.copy()
            # Process the frame with YOLO
            output_image = process_frame(frame,output_image)
            output_image=np.asarray(output_image)
            pred.image(output_image,caption="Output",use_column_width=True)   
           

    video_stream.release()
      
if __name__ == '__main__':
    main()
