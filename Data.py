from dataclasses import dataclass, field
import os,sys
import pickle
from typing import List, NoReturn, Iterable
from cv2 import VariationalRefinement
import numpy as np
import cv2
import utils.vid as utv

EXCEPT_DIR = ['.DS_Store','a', '_all']
DEFAULT_VID_ROOT = '/Users/0hyun/Desktop/vid'
@dataclass
class VidData():
    data_root: str
    verbose: bool
    class_list: List[str] = field(default_factory=list)
    def __init__(self, 
                data_root = DEFAULT_VID_ROOT, verbose = False):
        self.data_root = data_root
        self.verbose = verbose
        self.class_list = [l for l in os.listdir(self.data_root) if l not in EXCEPT_DIR]
        
        # self._count_all_vid()
        

    def _count_all_vid(self):
        self.cnt_vid: int = 0
        dir_class = [os.path.join(self.data_root, l) for l in self.class_list]

        for d in dir_class:
            self.cnt_vid += len([f for f in os.listdir(d) if '.mp4' in f])

        print(self.cnt_vid)        

    def load_video(self, save = True) -> Iterable[np.array]:
        for l in self.class_list:

            class_dir = os.path.join(self.data_root, l)
            for d in [c for c in os.listdir(class_dir) if '.mp4' in c]:
                d = os.path.join(class_dir, d) #abs_path
                
                if self.exists(d.replace('.mp4','.pkl')):
                    yield pickle.load(open(d.replace('.mp4','.pkl'), 'rb'))
                else:    
                    a = self._vid2arr(os.path.join(class_dir, d))
                    if save: self._save_vid2pkl(a,class_dir,d)
                    yield a      
    
    def _vid2arr(self, vid_path) -> np.ndarray:
        cap = cv2.VideoCapture(vid_path)
        vArr: List[np.array] = []
        while cap.isOpened():
            ret,frame = cap.read()
            

            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vArr.append(frame)
        
        vArr = np.array(vArr)
        if self.verbose: ut.show_vid(vArr)
        
        return vArr


    def exists(self, file) -> bool:
        return os.path.exists(file)

    
    def _save_vid2pkl(self,vArr,class_dir, name):
        name = name.replace('.mp4','.pkl')
        pickle.dump(vArr, open(os.path.join(self.data_root, class_dir, name), 'wb'))
        


