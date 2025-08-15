import cv2, time

class Stall:
    def __init__(self, single_stall_coordination, predicted_objects):
        self.stall_coord = single_stall_coordination
        self.objects_coord = predicted_objects[0].boxes.xyxy.cpu().numpy()
        self.objects_confidents = predicted_objects[0].boxes.conf.cpu().numpy()
        self.occupied_coord = None
        self.is_stall_occupied = False
        self.stall_state = {"current_state": None, "current_state_start_time": None,
                            "predicted_state": None, "predicted_state_start_time": None}
        self.__stall_occupation_status()


    def __center_in_xyxy(self, object_coordination) -> bool:
        x1, y1, x2, y2 = object_coordination
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        marked_x1, marked_y1, marked_x2, marked_y2 = self.stall_coord

        return (marked_x1 < center_x < marked_x2) and (marked_y1 < center_y < marked_y2)

    def __stall_occupation_status(self): 
        for coord in self.objects_coord.astype(int):
            if self.__center_in_xyxy(coord):
                self.occupied_coord = coord
                self.is_stall_occupied = True
                break

        # initializing the stall state
        now = time.monotonic()
        
        if all(value is None for value in self.stall_state.values()):
            self.stall_state["current_state"] = self.is_stall_occupied
            self.stall_state["current_state_start_time"] = now
            self.stall_state["predicted_state"] = self.is_stall_occupied
            self.stall_state["predicted_state_start_time"] = now
        else:
            self.__update_on_stall(now)


    def get_stall_coordination(self) -> list:
        return self.stall_coord
    
    def get_predicted_objects(self) -> tuple[list, list]:
        return (self.objects_coord, self.objects_confidents)

    def mark_on_frame(self, frame):
        if self.is_stall_occupied:
            color = (0, 255, 0)  
            cv2.rectangle(frame, 
                        (self.occupied_coord[0], self.occupied_coord[1]), 
                        (self.occupied_coord[2], self.occupied_coord[3]),
                        color=color,
                        thickness=2)
        return frame

    def __update_on_stall(self, now_time):
        pass

    def get_stall_state(self) -> dict:
        return self.stall_state