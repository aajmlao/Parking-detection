import cv2, time

class Stall:
    def __init__(self, id, single_stall_coordination):
        # Init values for the stall
        self.id = id 
        self.stall_coord = single_stall_coordination
        # store the occupied coordination from 
        self.occupied_coord = None
        
        self.current_state = None
        self.current_state_start_time = None
        self.predicted_state = None
        self.predicted_state_start_time = None
        

    def __center_in_xyxy(self, object_coordination) -> bool:
        x1, y1, x2, y2 = object_coordination
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        marked_x1, marked_y1, marked_x2, marked_y2 = self.stall_coord

        return (marked_x1 < center_x < marked_x2) and (marked_y1 < center_y < marked_y2)

    def stall_occupation_status(self, objects_coordination): 
        
        for coord in objects_coordination.astype(int):
            if self.__center_in_xyxy(coord):
                self.occupied_coord = coord
                if self.current_state is None:
                    self.current_state = True
                else:
                    self.predicted_state = True
                break
        # After the checking all predicted objects
        if self.current_state is None:
            self.current_state = False
        else:
            self.predicted_state = False
 
        now = time.monotonic()
        


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

    # def get_stall_state(self) -> dict:
    #     return self.stall_state