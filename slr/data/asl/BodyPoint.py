import copy
from dataclasses import dataclass, field
from typing import List, Optional
import traceback
import numpy as np
import mediapipe as mp
import cv2
import sys
sys.path.append("..")
import utils.vid as utv
import time
from static.const import *

@dataclass
class BodyPoint():
    vArr: np.ndarray
    verbose: bool = True

    def gen_bodypoint(self) -> np.ndarray:
        mp_sol_list: List[mp.solutions.holistic] = field(default_factory=list)
        mp_sol_list = self._mp_hoistic_process()

        # bodypoint from holistic solution
        res = self._bodypoint_from_holistic(mp_sol_list)

        return res 
        
    
    def _bodypoint_from_holistic(self, mp_sol_list) -> np.ndarray:

        # init bodypoint array
        face = np.zeros((1,FACE_FEATURES,3))
        pose = np.zeros((1,POSE_FEATURES,3))
        lh = rh = np.zeros((1,HAND_FEATURES,3))
        avail_frame = 0

        for i in mp_sol_list:

            isvisible_holistic_body = ((i.left_hand_landmarks or i.right_hand_landmarks) 
                                        and i.pose_landmarks and i.face_landmarks) is not None 
            

            # if isvisible flag gives false, it can be interpolated.
            # so check avail_frame != 0
            if isvisible_holistic_body or avail_frame != 0:
                avail_frame += 1
                
                pose = self._concat_bodypoint(i.pose_landmarks, pose)

                # data interpolation
                # for hands keypoint interpolation
                lh_indices =[15,17,19,21]
                rh_indices = [16,18,20,22]
                
                # in case: hands not detected or 
                # when too far two of each data which hands data and pose's hands data 
                lh_avr = pose[-1,lh_indices,:].mean(axis=0) 
                rh_avr = pose[-1,rh_indices,:].mean(axis=0)

                face = self._concat_bodypoint(i.face_landmarks, face)
                lh = self._concat_bodypoint(i.left_hand_landmarks, lh, lh_avr)
                rh = self._concat_bodypoint(i.right_hand_landmarks, rh, rh_avr)
                #print(lh[-1].mean(axis=0) == lh_avr, lh_avr, lh[-1].mean(axis=0), lh[-1].shape)
        
        
        face = np.array(face[1:])
        pose = np.array(pose[1:])
        lh = np.array(lh[1:])
        rh = np.array(rh[1:])

        try:
            assert len(face) == len(pose) == len(lh) == len(rh) == avail_frame
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("array len: ",len(face) , len(pose) ,len(lh) ,len(rh) ,avail_frame)
         


        return np.concatenate((face,pose,lh,rh), axis=1)
    
    def _concat_bodypoint(self, landmarks: object, base:np.ndarray, hands_mean:int = 0) -> np.ndarray:
        
        if landmarks:
            tmp = np.array([[i.x,i.y,i.z] for i in landmarks.landmark])            
            # # init bodypoint array
            # if base.size == 0: return np.expand_dims(tmp,0)

            return np.vstack([base,np.expand_dims(tmp,0)])
        
        # 중간에 landmark 사라짐(disapear)
        # -> 보간(interpolation) 수행
        else:

            """
            if landmark.landmark is None
            when keypoint disappear for a while
            interpolate(보간) data with previous keypoint
            """
            # a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
            # if there's no data interploate
            # fill the data with pose's hand's data

            # interpolation case: 
            # when hands are not detected, fill with pose's hand data 
            # when hands are too far from pose's hand's coordinate
            if (np.any(hands_mean) and ~np.any(base[-1])) or \
            (np.any(hands_mean) and (np.sqrt(sum((base[-1].mean(axis=0) - hands_mean) **2)) > 0.5)):
                tmp = np.zeros(base[-1].shape)
                tmp[:,:] = hands_mean
                return np.vstack([base,np.expand_dims(tmp,0)])
            
            # if VERBOSE and np.any(hands_mean): print(f"distance: {np.sqrt(sum((base[-1].mean(axis=0) - hands_mean) **2))}")
            
            return np.vstack([base,np.expand_dims(base[-1],0)])

    
    def _mp_hoistic_process(self):
        mp_holistic = mp.solutions.holistic

        with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
            
            #image to keypoint(mediapipe solution)
            res = [holistic.process(img) for img in self.vArr]
            
            assert len(res) == len(self.vArr)
        
        if self.verbose: self._show_vid(res)

        return res
        
    
    def _draw_landmarks(self,image: np.ndarray,mp_drawing, results: mp.solutions.holistic):
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))
        mp_drawing.draw_landmarks(
            image, results.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(circle_radius=1))

    #draw video with mediapipe bodypoint
    def _show_vid(self, mp_sol):
        
        drawing_vArr = copy.deepcopy(self.vArr)
        mp_drawing = mp.solutions.drawing_utils

        try:
            #draw bodypoint process with mediapipe solution
            for n,(i,r) in enumerate(zip(drawing_vArr, mp_sol)):
                i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
                self._draw_landmarks(i, mp_drawing, r)
                drawing_vArr[n] = cv2.cvtColor(i,cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(e)
            print(traceback.format_exc())
        
        utv.show_vid(drawing_vArr)