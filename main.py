# from ultralytics import YOLO
# from byte_tracker_pytorch.byte_tracker_model import BYTETracker as ByteTracker
import tqdm
import argparse
import datetime
import torch
# import insightface
import cv2
import numpy as np
import math
import time
import statistics
import numpy as np
# from scipy.ndimage import zoom
import pandas as pd
import subprocess
from predictor import Predictor
import random
from bot_sort.tracker.mc_bot_sort import BoTSORT
from easydict import EasyDict as edict
import os
import onnxruntime as ort
# from mivolo.data.data_reader import InputType, get_all_files, get_input_type


# person-face model weights path
detector_weights=f"{os.getcwd()}/models/yolov8x_person_face.onnx"

# age-gender model weight path
checkpoint=f"{os.getcwd()}/models/modified_mivolo_age_gender.onnx"

# bot-sort tracker weights path
botsort_tracker_weights = f"{os.getcwd()}/models/mot20_sbs_S50_botsort.onnx"



video_path = "videos/mall.mp4"      # Input video path
MAX_FRAME_COUNT = 1000              # Maximum number of frames to do inference 
DEBUG = False                       # Change to True if need to see the logs


# Using only Cuda 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Checking if GPU is available in the system
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

 
# From Python3.7 you can add
# keyword argument capture_output
print(subprocess.run(["echo", "app is starting"],
                     capture_output=True))
name = "~/Desktop/sinage"+str(time.time())
is_watched = []


x_y_thresh = 35

class CSVDataFrame:
    def __init__(self):
        # self.df = pd.DataFrame(columns=["id","start_time","end_time","dwell_time","starting_position","max_position","ending_position","direction_movement","age","gender"])
        
        self.columns =["id","age","gender","is_customer","dwell_time","screen_time"
                       ,"start_time"
                       ,"end_time","starting_position","max_position",
                       "ending_position","direction_movement"]
        self.data_list = []
        
    def add_data(self, data):
        # print("data",data)
        # print("df",self.df)
        self.data_list.append(data)
        df = pd.DataFrame(self.data_list, columns=self.columns)
        global name
        df.to_csv(name+".csv",index=False)
    
    
def get_rotation(points):
    """
    Parameters
    ----------
    points : float32, Size = (5,2)
        coordinates of landmarks for the selected faces.
    Returns
    -------
    float32, float32, float32
    """
    # print("points",points)
    LMx = points[:,0]#points[0:5]# horizontal coordinates of landmarks
    LMy = points[:,1]#[5:10]# vertical coordinates of landmarks
    
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = (LMy[1] - LMy[0])
    angle = np.arctan(dPy_eyes / dPx_eyes) # angle for rotation based on slope
    
    alpha = np.cos(angle)
    beta = np.sin(angle)
    
    # rotated landmarks
    LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2) 
    LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)
    
    # average distance between eyes and mouth
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2
    
    # average distance between nose and eyes
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2
    
    # relative rotation 0 degree is frontal 90 degree i                                 s profile
    Xfrontal = (-90+90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    Yfrontal = (-90+90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0
    rotation = {"Z":round(angle * 180 / np.pi,2),"X": round(Xfrontal,2),"Y": round(Yfrontal,2)}
    is_approved = True if (abs(rotation["X"])<60 and abs(rotation["Y"])<60) else False
    is_add = True if (abs(rotation["X"])<35 and abs(rotation["Y"])<35) else False
    return (is_approved,is_add,rotation)

    
    
class People:
# current_frame_data[id] = {"track_id": id, "bbox": box, "score": conf, "class": 0, "missing_count": 0}
    
    def __init__(self):
        self.id = None
        self.bbox = None
        self.score = None
        self.missing_count = 0
        self.first_bbox = None
        self.last_bbox = None
        self.max_bbox = None
        self.first_center = None
        self.last_center = None
        self.max_center = None
        self.max_distance = -1
        self.first_time = None
        self.last_time = None
        self.score = None
        self.gender = []
        self.age = []
        self.is_visitor = False
        self.is_visitor_frame_count = 0
        self.face_bbox = None
        self.depth_min = 10000
        self.depth_max = 0
        self.visitor_thresh_depth = 3000
        self.is_visitor_frame_count_thresh = 20
        self.visitor_left_depth = 300
        self.visitor_left_count = 0
        self.current_depth = 0
        self.ignore_customer =0.05 #from bottom 
        
        self.distance = 0
        self.x_distance = 0
        self.y_distance = 0
        global x_y_thresh
        self.distace_threshold_for_screentime_from_center = x_y_thresh
        self.total_screen_time_milliseconds = 0
        self.starting_screen_time = None
        
        self.dwell_start_time = datetime.datetime.now()
        self.dwell_end_time = None
        self.dwell_time_thresh = 5
        
        
        
        self.dwell_out_customer = 0
        
        self.dwell_time_to_be_customer = 5
    
        self.random_color = None
        
        
    def append_data(self, data,yolo_object):
        # current_frame_data[id] = {"track_id": id, "bbox": box, "score": conf, "class": 0, "missing_count": 0}
        self.id = data["track_id"]
        self.bbox = data["bbox"]
        if DEBUG:
            self.random_color = data["random_color"]
        if self.first_bbox is None:
            self.first_bbox = data["bbox"].tolist()
            self.first_time = datetime.datetime.now()
            self.first_center = ((self.first_bbox[0]+self.first_bbox[2])/2, (self.first_bbox[1]+self.first_bbox[3])/2)
        
        if self.depth_min > data["depth"]:
            self.depth_min = data["depth"]
        
        if self.depth_max < data["depth"]:
            self.depth_max = data["depth"]
            
        self.current_depth = data["depth"]
        self.last_bbox = data["bbox"].tolist()
        self.last_center =((self.last_bbox[0]+self.last_bbox[2])/2, (self.last_bbox[1]+self.last_bbox[3])/2)
        
        if "distance_from_center_x" in data.keys() and "distance_from_center_y" in data.keys() and abs(data["distance_from_center_x"]) < self.distace_threshold_for_screentime_from_center and abs(data["distance_from_center_y"]) < self.distace_threshold_for_screentime_from_center:
                if self.starting_screen_time is not None:
                    self.total_screen_time_milliseconds += (datetime.datetime.now() - self.starting_screen_time).total_seconds()
                self.starting_screen_time = datetime.datetime.now()
        else:
            self.starting_screen_time = None
        # print(self.last_bbox[3] , yolo_object.height*self.ignore_customer)
        
        # if bottom point is less than 5% of height of window then that is not customer that is staff
        if self.last_bbox[3] < yolo_object.height - (yolo_object.height*self.ignore_customer):
            
            if data['depth'] > self.visitor_thresh_depth:
                if self.is_visitor_frame_count ==0:
                    self.dwell_start_time = datetime.datetime.now()
                self.is_visitor_frame_count += 1
            else:
                self.is_visitor_frame_count = 0
                
        if self.is_visitor and self.current_depth < self.visitor_left_depth:
            self.visitor_left_count += 1
            if self.visitor_left_count > self.is_visitor_frame_count:
                self.dwell_end_time = datetime.datetime.now()
                
            
        if self.is_visitor_frame_count > self.is_visitor_frame_count_thresh:
            self.is_visitor = True
            if DEBUG:
                print("is_visitor",self.id)
            global is_watched
            if self.id not in is_watched:
                is_watched.append(self.id)
                cv2.waitKey(0)
            
        
        temp_distance = np.linalg.norm(np.array(self.first_center) - np.array(self.last_center))
        if temp_distance > self.max_distance:
            self.max_bbox = data["bbox"].tolist()
            self.max_center = ((self.max_bbox[0]+self.max_bbox[2])/2, (self.max_bbox[1]+self.max_bbox[3])/2)
        
        
        self.last_time = datetime.datetime.now()
        self.score = data["score"]
        # print("data",data)
        if 'face_bbox' in data.keys() and data['face_bbox'] is not None:
            self.face_bbox = data['face_bbox']
            # print(data,data['face_bbox'])
            # if data['face_bbox'][1]:
            
            width_face = abs(data['face_bbox'][0]-data['face_bbox'][2])
            height_face = abs(data['face_bbox'][1]-data['face_bbox'][3])
            # Here when the image is rotated horizontally the then dont show the gender
            if height_face <  2*width_face:         

                self.gender.append(data['gender'])
                self.age.append(data['age'])
        else:
            self.face_bbox = None

    def direction(self):
        angle = math.atan2(self.max_center[1] - self.first_center[1], self.max_center[0] - self.first_center[0])
        #angle to degress
        angle = math.degrees(angle)
        return angle
        
    # def get_gender(self):
    #     #check list for Male Cound and Female count and dominaing
    #     male_count = 
        
    def final_call_data(self,yolo_object):
        if DEBUG:
            print("Hitting final Call for data",yolo_object)
        # self.df = pd.DataFrame(columns=["id","start_time","end_time","dwell_time","starting_position","max_position","ending_position","direction_movement","age","gender"])
        # temp_df = pd.DataFrame
        #dwell time in seconds
        # if self.dwell_start_time is not None and self.is_visitor:
        #     start_time = self.dwell_start_time
        #     if self.dwell_end_time is not None:
        #         end_time = self.dwell_end_time
        #     else:
        #         end_time = self.last_time
        #     dwell_time = (end_time - start_time).total_seconds()
        # else:
        #     dwell_time = None
        dwell_time = (self.last_time - self.first_time).total_seconds()
        if dwell_time > self.dwell_time_to_be_customer:
            self.is_visitor = True

        starting_position_normalized = [self.first_bbox[0]/yolo_object.width,self.first_bbox[1]/yolo_object.height,self.first_bbox[2]/yolo_object.width,self.first_bbox[3]/yolo_object.height]
        max_position_normalized = [self.max_bbox[0]/yolo_object.width,self.max_bbox[1]/yolo_object.height,self.max_bbox[2]/yolo_object.width,self.max_bbox[3]/yolo_object.height]
        ending_position_normalized = [self.last_bbox[0]/yolo_object.width,self.last_bbox[1]/yolo_object.height,self.last_bbox[2]/yolo_object.width,self.last_bbox[3]/yolo_object.height]
        if len(self.age) >0:
            age = statistics.mode(self.age)
            gender = statistics.mode(self.gender)
        else:
            age = None
            gender = None
        direction = self.direction()
        # self.columns =["id","age","gender","is_customer","dwell_time","start_time","end_time","starting_position","max_position","ending_position","direction_movement"]
        
        temp_df = [self.id,age,gender,self.is_visitor,dwell_time,self.total_screen_time_milliseconds
                ,self.first_time,self.last_time,starting_position_normalized
                ,max_position_normalized,ending_position_normalized,direction
                ]
        

        return temp_df
        

class YoloDetector:
    def __init__(self,botsort_weights,fps,device = "cpu"):
        #create empty dataframe with columns named as id,start_time,end_time,dwell_time,starting_position,ending_position,diretion_movement,age,gender
        
        # parser = argparse.ArgumentParser()
        self.df_object = CSVDataFrame()
        
        self.width = 0
        self.height = 0

        # parameters for bot-sort tacker with reid
        opt = {'track_high_thresh':0.3,'track_low_thresh':0.05,'track_buffer':30,'with_reid':True,
                         "proximity_thresh":0.5,"new_track_thresh":0.4,"appearance_thresh":0.25,"fast_reid_config":'bot_sort/fast_reid/configs/MOT20/sbs_R50.yml',
                         'fast_reid_weights':botsort_weights,'device': device,'cmc_method':'sparseOptFlow','name':'exp'
                         ,'ablation':False,'mot20':True,'match_thresh':0.7,'DEBUG':DEBUG
                         }
        # print(opt)
        opt = edict(opt)
        self.tracker = BoTSORT(opt,frame_rate=fps)
        # self.tracker = ByteTracker(1, 0.1, 0.5, 0.8, 50, 640,True)
        
        self.tracked_people = {}
        self.old_people = {}
        self.delete_threshold = 60

    def compose2(self, f1, f2):
        return lambda x: f2(f1(x))
    
    def detection_faces(self,image_frame):
        faces = self.app.get(image_frame)
        return faces
    
    def get_rotation(self,points):
        """
        Parameters
        ----------
        points : float32, Size = (5,2)
            coordinates of landmarks for the selected faces.
        Returns
        -------
        float32, float32, float32
        """
        # print("points",points)
        LMx = points[:,0]#points[0:5]# horizontal coordinates of landmarks
        LMy = points[:,1]#[5:10]# vertical coordinates of landmarks
        
        dPx_eyes = max((LMx[1] - LMx[0]), 1)
        dPy_eyes = (LMy[1] - LMy[0])
        angle = np.arctan(dPy_eyes / dPx_eyes) # angle for rotation based on slope
        
        alpha = np.cos(angle)
        beta = np.sin(angle)
        
        # rotated landmarks
        LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2) 
        LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)
        
        # average distance between eyes and mouth
        dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
        dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2
        
        # average distance between nose and eyes
        dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
        dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2
        
        # relative rotation 0 degree is frontal 90 degree i                                 s profile
        Xfrontal = (-90+90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
        Yfrontal = (-90+90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0
        rotation = {"Z":round(angle * 180 / np.pi,2),"X": round(Xfrontal,2),"Y": round(Yfrontal,2)}
        is_approved = True if (abs(rotation["X"])<60 and abs(rotation["Y"])<60) else False
        is_add = True if (abs(rotation["X"])<35 and abs(rotation["Y"])<45) else False
        return (is_approved,is_add,rotation)
    
    def add_frames_data(self,current_frame_data):
        temp_tracked_people = self.tracked_people.copy()
        temp_old_people = self.old_people.copy()
        self.tracked_people = {}
        self.old_people = {}
        
        for tracked_person in temp_tracked_people.keys():
            if tracked_person in current_frame_data.keys():
                self.tracked_people[tracked_person] = temp_tracked_people[tracked_person]
                self.tracked_people[tracked_person].append_data(current_frame_data[tracked_person],self)
                del current_frame_data[tracked_person]
            else:
                temp_tracked_people[tracked_person].missing_count += 1
                self.old_people[tracked_person] = temp_tracked_people[tracked_person]
                # self.temp_old_people[tracked_person].append_data(temp_tracked_people[tracked_person])
                
        for old_person in temp_old_people.keys():
            if old_person in current_frame_data.keys():
                self.tracked_people[old_person] = temp_old_people[old_person]
                self.tracked_people[old_person].append_data(current_frame_data[old_person],self)
                self.tracked_people[old_person].missing_count = 0
                del current_frame_data[old_person]
            else:
                if temp_old_people[old_person].missing_count < self.delete_threshold:
                    self.old_people[old_person] = temp_old_people[old_person]
                    self.old_people[old_person].missing_count += 1
                    
                else:
                    temp_df = temp_old_people[old_person].final_call_data(self)
                    self.df_object.add_data(temp_df)
                    # self.tracked_people[old_person].add
                # else:
                #     del temp_old_people[old_person]
                    
        for new_person in current_frame_data.keys():
            temp_person = People()
            # print("Hitting here")
            temp_person.append_data(current_frame_data[new_person],self)
            self.tracked_people[new_person] = temp_person
    

def load_model(detector_weights,checkpoint,with_persons = True,disable_faces = False,draw = False,device = "cpu"):
    # setup_default_logging()
    """
    Loading models
    detector_weights: Person and face detection model path
    checkpoints : Age and gender detection model path
    
    """

    class Args:
        def __init__(self, detector_weights, checkpoint, with_persons, disable_faces, draw, device,DEBUG):
            self.detector_weights = detector_weights
            self.checkpoint = checkpoint
            self.with_persons = with_persons
            self.disable_faces = disable_faces
            self.draw = draw
            self.device = device
            self.DEBUG = DEBUG

    args = Args(detector_weights, checkpoint, with_persons, disable_faces, draw, device,DEBUG)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    # os.makedirs(args.output, exist_ok=True)

    # Initializing the age-gender and peson-face model
    predictor = Predictor(args, verbose=True)
    return predictor


model = load_model(
    detector_weights=detector_weights,
    checkpoint=checkpoint,
    device=device,
)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm.tqdm(total=total_frames, desc="Processing Frames", unit="frame",colour='blue')

save_folder = "output_videos"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

output_name = video_path.split("/")[-1]
video_writer = cv2.VideoWriter(os.path.join(save_folder,output_name), 
                cv2.VideoWriter_fourcc(*'MJPG'),
                fps, size)

yolo_object = YoloDetector(botsort_weights = botsort_tracker_weights, fps = fps,device=device)
count = 0

all_complete_time = time.time()
detection_avg_time = []
recog_avg_time = []
tracking_avg_time = []

while True:
    ret, frame = cap.read()
    
    if not ret or count == MAX_FRAME_COUNT:
        print("Video file finished. or Error in reading video file or Camera not opened")
        video_writer.release()
        
        break
    count += 1
    progress_bar.update(1)
    #print(count)
    yolo_object.width,yolo_object.height = frame.shape[1],frame.shape[0]
    det_time = time.time()
    detected_object,customized_detected_objects = model.custom_detect(frame)
    detection_avg_time.append(time.time() - det_time)

    person_detections = []
    detections = []
    ages = []
    genders = []
    bboxes = detected_object.boxes
    for i,bbox in enumerate(bboxes):
        current_bbox = bbox.xyxy.detach().cpu().tolist()[0]
        class_name = bbox.cls
        score = bbox.conf
        if class_name == 0:     # person
            current_bbox.append(score.item())
            current_bbox.append(int(class_name))
            person_detections.append(current_bbox)

        else:       # This is face bbox
            current_bbox = [int(box) for box in current_bbox]
            detections.append(current_bbox)
    if len(detections) > 0:
        recog_time = time.time()
        ages,gender_codes = model.custom_recognize(customized_detected_objects,frame)
        recog_avg_time.append(time.time() - recog_time)
        ages = [int(age) for age in ages]
        
        
        
        if DEBUG:
            print('Gender code =',gender_codes)
            print('Ages = ',ages)
        for gender_val in (gender_codes):
            if gender_val[0] == 1:
                gender = "F"
            else:
                gender = "M"
            genders.append(gender)

    person_detections = np.array(person_detections)
    update_time = time.time()
    track_list  = yolo_object.tracker.update(person_detections,frame)
    tracking_avg_time.append(time.time()-update_time)
    
    id_list = [t.track_id for t in track_list]

    # Get box list
    box_list = [t.tlbr for t in track_list]

    # Get conf scores
    conf_list = [t.score for t in track_list]

    # Number of objects
    num_objects = len(box_list)
    
    current_frame_data = {}

    for i in range(num_objects):
        box = box_list[i]
        conf = conf_list[i]
        id = id_list[i]
        depth = 0
        if DEBUG:
            random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            current_frame_data[id] = {"track_id": id, "bbox": box, "score": conf, "class": 0, "missing_count": 0,"depth":depth,"random_color":random_color}
        else:
            current_frame_data[id] = {"track_id": id, "bbox": box, "score": conf, "class": 0, "missing_count": 0,"depth":depth}            

    for i,bbox,gender,age in zip(range(len(detections)),detections,genders,ages):
        best_person = None
        distance_in_y_axis = 100000
        
        idx = None
        score = None
        add_visitor = None
        for person in current_frame_data:
            person = current_frame_data[person]

            # First confirm whether the face is inside this person bbox
            if bbox[0] >= person["bbox"][0] and bbox[1] >= person["bbox"][1] and bbox[2] <= person["bbox"][2] and bbox[3] <= person["bbox"][3]:

                # y center of person bbox
                person_bbox_y_center = (person["bbox"][1]+person["bbox"][3]) / 2    
                # y center of face bbox
                current_face_bbox_y_center = (bbox[1]+bbox[3])/2                    
                
                # If the face y center is above the y center of person bbox
                if current_face_bbox_y_center < person_bbox_y_center:
                    if "face_bbox" in person.keys():                # Person already has a face
                        person_bbox_center = (person["bbox"][0]+person["bbox"][2]) / 2

                        current_face_bbox_x_center = (bbox[0]+bbox[2])/2
                        current_distance_from_center = abs(person_bbox_center - current_face_bbox_x_center)                            
                        
                        old_face_bbox_x_center = (person["face_bbox"][0]+person["face_bbox"][2]) / 2
                        old_distance_from_center = abs(person_bbox_center - old_face_bbox_x_center)
                    
                        # If the current face is near to the center of person bbox than the old face, then this current face is the actual face of this person bbox
                        if current_distance_from_center < old_distance_from_center:
                            area_current_face_bbox = abs(bbox[0]-bbox[2]) * abs(bbox[1]-bbox[3])
                            area_old_face_bbox = abs(person["face_bbox"][0]-person["face_bbox"][2]) * abs(person["face_bbox"][1]-person["face_bbox"][3])

                            # if the current face is greater than the 1.5 times of the old face of this person,
                            # this handles when the actual face is not near to the center of the person bbox and other face is near to the person bbox center but it is small, i.e., it means the other face is behind the person bbox 
                            if area_current_face_bbox > 1.5 * area_old_face_bbox:
                                best_person = person
                    else:   # Person has no face
                        best_person = person
                        
        if best_person is not None:
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            x = height*width / 10000
            distance = -5.822855 + (142529000 - -5.822855)/(1 + (x/8.791362e-15)**0.4373197)
            current_frame_data[best_person["track_id"]].update ({"face_bbox":bbox,"age":age,
                                                                "gender":gender,"distance":distance})


    yolo_object.add_frames_data(current_frame_data)

    # This will draw all the faces detected in black color, USE FOR DEBUGGING
    if DEBUG : 
        for face_detection in detections:
            frame = cv2.rectangle(frame,(int(face_detection[0]),int(face_detection[1])),(int(face_detection[2]),int(face_detection[3])),(0,0,0),2)

    for person in yolo_object.tracked_people.keys():

        person = yolo_object.tracked_people[person]
        if DEBUG:
            random_color = person.random_color
            frame = cv2.rectangle(frame,(int(person.last_bbox[0]),int(person.last_bbox[1])),(int(person.last_bbox[2]),int(person.last_bbox[3])),random_color,2)
        else:
            frame = cv2.rectangle(frame,(int(person.last_bbox[0]),int(person.last_bbox[1])),(int(person.last_bbox[2]),int(person.last_bbox[3])),(255,0,255),2)

        
        #face bbox
        if person.face_bbox is not None:


            if DEBUG:
                frame = cv2.rectangle(frame,(int(person.face_bbox[0]),int(person.face_bbox[1])),(int(person.face_bbox[2]),int(person.face_bbox[3])),random_color,2)
            else:
                frame = cv2.rectangle(frame,(int(person.face_bbox[0]),int(person.face_bbox[1])),(int(person.face_bbox[2]),int(person.face_bbox[3])),(0,0,255),2)
                
                


                # self.gender[-1] = (statistics.mode(self.gender[-20:]))
                # self.age[-1] = round(statistics.mean(self.age[-20:]))
            width_face = abs(person.face_bbox[0]-person.face_bbox[2])
            height_face = abs(person.face_bbox[1]-person.face_bbox[3])
            if len(person.age) >0 and height_face < 1.5 * width_face:

                # The below will use mode of age and gender

                frame = cv2.putText(frame,"A:"+str(round(statistics.mode(person.age[-20:]))),(int(person.face_bbox[0]),int(person.face_bbox[1])-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                frame = cv2.putText(frame,"G:"+str((statistics.mode(person.gender[-20:]))),(int(person.face_bbox[0]),int(person.face_bbox[1])+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) 

                # frame = cv2.putText(frame,"A:"+str(person.age[-1]),(int(person.face_bbox[0]),int(person.face_bbox[1])-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                # frame = cv2.putText(frame,"G:"+str(person.gender[-1]),(int(person.face_bbox[0]),int(person.face_bbox[1])+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) 
        # cropped_output = output[int(person.last_bbox[1]):int(person.last_bbox[3]),int(person.last_bbox[0]):int(person.last_bbox[2])]
         
        
        frame =cv2.putText(frame,"D:"+str(person.current_depth),(int(person.last_bbox[0]),int(person.last_bbox[1])+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        #id 
        frame = cv2.putText(frame,str(person.id),(max(80,int(person.last_bbox[0])),max(80,int(person.last_bbox[1])-10)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        
    # cv2.imwrite(f"{count}.jpeg",frame)
    video_writer.write(frame)
    # print(yolo_object.tracked_people)
    # frame = cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
    frame = cv2.resize(frame,(frame.shape[1]*2,frame.shape[0]*2))
    # cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
video_writer.release()
progress_bar.close()

print(f"total time taken with {device} is {time.time() - all_complete_time}")

detection_avg_time.pop(0)
recog_avg_time.pop(0)
tracking_avg_time.pop(0)
print("Average detection time",np.mean(detection_avg_time))
print("Average recognition time",np.mean(recog_avg_time))
print("Average tracking time",np.mean(tracking_avg_time))
        