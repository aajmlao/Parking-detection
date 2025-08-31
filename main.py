import cv2, json, os
import numpy as np
from ultralytics import YOLO
import multiprocessing as mp
from dotenv import load_dotenv
from stall import Stall

def read_json(file_name):
    with open(file_name, "r") as file:
        return json.load(file)

def operation(result, stalls, frame, stall_list):
    for stall in stalls:
        stall_id, stall_coord = stall['id'], stall['coord']

        if stall_id not in stall_list:
            # create a new stall object if we don't have one
            spot = Stall(stall_id, stall_coord)
            stall_list[stall_id] = spot
        else:
            # if the stall exit, we just call the object without creating a new one
            spot = stall_list[stall_id]

        spot.stall_occupation_status(objects_coordination = result)
        # print(f'{stall["id"]} is {spot.get_stall_state()}')
        cv2.imshow('RTSP Stream', spot.mark_on_frame(frame))

def decode_predicting_result(result) -> tuple:
    coordination = result[0].boxes.xyxy.cpu().numpy()
    conf = result[0].boxes.conf.cpu().numpy()
    return coordination, conf

def main():
    load_dotenv()
    camera_address = os.getenv("camera_address")
    file_name = "spots_xyxy.json"
    stalls = read_json(file_name=file_name)
    model = YOLO('runs/detect/train5/weights/best.pt').to(0)
    cap = cv2.VideoCapture(camera_address)

    # initilize a list to store the stall object
    stall_object_list = {}

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
        
        predicted_result = model.predict(source=frame, 
                                conf=0.45, 
                                iou=0.45, # During NMS, if two boxes overlap more than 0.5, the lower-score one is suppressed.
                                agnostic_nms=True, # Prevents one object being kept twice as, say, “car” and “truck.”
                                verbose=False, 
                                vid_stride=2)
                  
        coord, conf = decode_predicting_result(result=predicted_result)
        operation(result = coord, stalls=stalls, frame=frame, stall_list=stall_object_list)
        # cv2.imshow('RTSP Stream', predicted_result[0].plot()) # Display the predicted

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()