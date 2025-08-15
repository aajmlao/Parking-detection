import cv2, json, os
import numpy as np
from ultralytics import YOLO
from multiprocessing import Process
from dotenv import load_dotenv

def read_json(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)
    stalls = []
    for stall in data:
        stalls.append(stall)
    return stalls

def center_in_xyxy(vehicle_spot_xyxy, parking_spot):
    x1, y1, x2, y2 = vehicle_spot_xyxy
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    marked_x1, marked_y1, marked_x2, marked_y2 = parking_spot
    
    return (marked_x1 < center_x < marked_x2) and (marked_y1 < center_y < marked_y2)

def occupation(results, stalls):
    coordinates = results.boxes.xyxy.cpu().numpy()
    
    frame = results.plot()
    stall_occupation_arr = []
    for stall in stalls:
        is_occupied_arr = []
        parking_stall = stall['coord']
        for coordinate in coordinates.astype(int):
            is_occupied = center_in_xyxy(coordinate, parking_stall)
            is_occupied_arr.append((is_occupied, coordinate))
        
        x1, y1, x2, y2 = None, None, None, None
        color = (0, 255, 0)
        stall_occupation = False
        for is_occupied, coordinate in is_occupied_arr:
            if is_occupied:
                x1, y1, x2, y2 = coordinate
                color = (0, 0, 255)
                stall_occupation = is_occupied
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
                break
    
        if len(stall_occupation_arr) <= 0:
            stall_occupation_arr.append((stall['id'], stall_occupation))
        elif stall['id'] == stall_occupation_arr[-1][0]:
            stall_occupation_arr.pop()
            stall_occupation_arr.insert(0, (stall['id'], stall_occupation))
        else:
            stall_occupation_arr.insert(0, (stall['id'], stall_occupation))
        # This logic need to implement with time or time like algorithm
        if not stall_occupation_arr[0][1]:
            print(f'Stall: {stall_occupation_arr[0][0]} is empty!!!') 
        # else:
        #     print(f'Stall: {stall_occupation_arr[0][0]} is occupied!!!')
    return frame

# need to implement this function
def valification_of_stalls():
    pass


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
                  
        modified_frame = occupation(results[0], stalls=stalls)
        cv2.imshow('RTSP Stream', modified_frame) # Display the frameq

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()