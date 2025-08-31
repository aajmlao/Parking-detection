import cv2, time

class Stall:
    def __init__(self, id, single_stall_coordination):
        # Init values for the stall
        self.id = id 
        self.stall_coord = single_stall_coordination
        # store the occupied coordination from 
        self.occupied_coord = None
        # store the state
        self.current_state = None
        self.current_state_start_time = None
        self.hold_state = None
        self.previous_time = None
        self.accumulate_time = 0
        self.last_flip_time = None
        
    # helper function
    def __center_in_xyxy(self, object_coordination) -> bool:
        x1, y1, x2, y2 = object_coordination
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        marked_x1, marked_y1, marked_x2, marked_y2 = self.stall_coord

        return (marked_x1 < center_x < marked_x2) and (marked_y1 < center_y < marked_y2)

    def stall_occupation_status(self, objects_coordination):
        NOTIFICATION_THRESHOLD = 5 # second 
        # True is occupied. False is empty.
        for coord in objects_coordination.astype(int):
            if self.__center_in_xyxy(coord):
                self.occupied_coord = coord
                # init the state if the state is none else we need to check the 
                # state wethere is empty or occupied. All prediction here is 
                # True 
                now = time.monotonic()
                if self.current_state is None:
                    self.current_state = True
                    self.current_state_start_time = now
                    self.previous_time = now
                else:
                    self.__update_on_stall(now, True)
                break
        
        # After the checking all predicted objects
        now = time.monotonic()
        if self.current_state is None:
            self.current_state = False
            self.current_state_start_time = now
            self.previous_time = now
        else:
            self.__update_on_stall(now, False)     

        # check the parking status.
        delta_time = now - self.current_state_start_time
        if delta_time >= NOTIFICATION_THRESHOLD and self.current_state == False:
            print(f"{self.id} is empty for {delta_time}")
            self.current_state_start_time = now

    # helper function
    def __update_on_stall(self, now, predicted_state) -> None:
        FLIP_STATE_THRESHOLD = 3 # seconde
        STAY_STATE_THRESHOLD = 0.8 # seconde
        weight = 0.7 # weight more on the equal states 

        if self.previous_time is None:
            self.previous_time = self.current_state_start_time

        delta_t = now - self.previous_time
        self.previous_time = now

        if self.current_state != predicted_state:
            if self.hold_state != predicted_state: # when we first encounter the changing state, we hold it
                self.hold_state = predicted_state 
            self.accumulate_time = min(FLIP_STATE_THRESHOLD, self.accumulate_time + delta_t) # accumulate time when changing state happens
            self.last_flip_time = now
        else:
            self.accumulate_time = self.accumulate_time - delta_t * weight # equaling states weight more. Maybe a mistake
            if self.accumulate_time < 0:
                self.accumulate_time = 0 # have a negative accumulate time. minimal value is 0

            # check if stay in the same state for a while, and reset the value
            if self.last_flip_time is not None:
                if (now - self.last_flip_time) >= STAY_STATE_THRESHOLD:
                    # reset
                    self.hold_state = None
                    self.last_flip_time = None
                    self.accumulate_time = 0
            
        # check if changing the state
        if self.accumulate_time >= FLIP_STATE_THRESHOLD and self.hold_state is not None:
            # test line
            time_span_test = self.accumulate_time
            # commit the change
            self.accumulate_time = 0
            self.current_state = predicted_state
            self.current_state_start_time = now
            self.hold_state = None
            self.last_flip_time = None
             # debug lines blow
            state_test = "occupied" if self.current_state else "empty"
            print(f"{self.id} changes state to {state_test} by accumlating in {time_span_test}s.")

    def get_stall_state(self) ->  str:
        return "Occupied" if self.current_state else "Empty"

    def get_stall_coordination(self) -> list:
        return self.stall_coord
    
    # def get_predicted_objects(self) -> tuple[list, list]:
    #     return (self.objects_coord, self.objects_confidents)

    def mark_on_frame(self, frame):
        if self.current_state:
            color = (0, 255, 0)  
            cv2.rectangle(frame, 
                        (self.occupied_coord[0], self.occupied_coord[1]), 
                        (self.occupied_coord[2], self.occupied_coord[3]),
                        color=color,
                        thickness=2)
        
        return frame