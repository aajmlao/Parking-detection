import cv2, json, os
import numpy as np
from ultralytics import YOLO
from multiprocessing import Process
from dotenv import load_dotenv
from stall import Stall

def read_json(file_name):
    with open(file_name, "r") as file:
        return json.load(file)

def opeartion(results, stalls, frame):
    for stall in stalls:
        spot = Stall(stall['coord'], results)
        print(f'{stall["id"]} is {spot.get_stall_state()["current_state"]}')
        cv2.imshow('RTSP Stream', spot.mark_on_frame(frame))

def main():
    load_dotenv()
    camera_address = os.getenv("camera_address")

    file_name = "spots_xyxy.json"

    stalls = read_json(file_name=file_name)

    model = YOLO('runs/detect/train5/weights/best.pt').to(0)
    
    cap = cv2.VideoCapture(camera_address)
    
    if not cap.isOpened():
        print("Error: Could not Open RTSP stream.")
        exit()
    else:
        print("It is opened!!")

    while True:
        ret, frame = cap.read() # Read a frame

        if not ret:
            print("Error: Failed to read frame or stream ended.")
            break
        
        results = model.predict(source=frame, 
                                conf=0.50, 
                                iou=0.45, # During NMS, if two boxes overlap more than 0.5, the lower-score one is suppressed.
                                agnostic_nms=True, # Prevents one object being kept twice as, say, “car” and “truck.”
                                verbose=False, 
                                vid_stride=2)
                  
        # modified_frame = opeartion(results, stalls=stalls)
        opeartion(results, stalls=stalls, frame=frame)
        # cv2.imshow('RTSP Stream', modified_frame) # Display the frameq

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()