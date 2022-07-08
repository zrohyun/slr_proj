import os, sys
import time
from pathlib import Path

from dataclasses import dataclass, field
from typing import List
from collections import defaultdict
import pickle


import numpy as np

from slr.data.ksl.keypoint import *
from slr.data.ksl.datapath import DataPath

# from typing import 
DATA_ROOT_PATH_AIHUB = r"D:\유영현\ksl\수어 영상\1.Training\[라벨]02_real_word_keypoint\02"
SAMPLE_DATA_PATH = r'd:\유영현\ksl\수어 영상\1.Training\[라벨]02_real_word_keypoint\02\NIA_SL_WORD0001_REAL02_D\NIA_SL_WORD0001_REAL02_D_000000000000_keypoints.json'

@dataclass
class KeypointSeq():
    data_path: Path
    word_cls: str  = ""# word class 
    

    @property
    def key_arr(self) -> np.ndarray:
        npy_path = self.data_path / (self.data_path.name + ".npy")
        return self._load_data(npy_path)

    def _load_data(self, file_name:Path) -> np.ndarray:
        if file_name.is_file():
            # print('loading arr')
            return np.load(file_name)
        else:
            # print('saving arr')
            data = np.array([Parser(i).keypoints for i in self.data_path.iterdir()])
            np.save(file_name, data)
            return data


@dataclass
class Parser():
    path: Path
    angle: str
    word: str
    method: str
    kpoint: Keypoint

    def __init__(self, path:Path, dimension = 2):
        self.path = path
        self.d = dimension

        parts = self.path.parts[-2].split("_")
        self.word = "_".join(parts[:3])
        self.method = parts[3]
        self.angle = parts[4]


    def parse_json(self):
        with open(self.path) as f:
            people = eval(f.read())['people'] # data is read by str, casting str to dict
            
            data = []
            if self.d == 2:
                for k in ['face_keypoints_2d', 'pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
                    data.append(people[k])
            else:
                for k in ['face_keypoints_3d', 'pose_keypoints_3d', 'hand_left_keypoints_3d', 'hand_right_keypoints_3d']:
                    data.append(people[k])

            self._set_kpoint(data)


    def _set_kpoint(self, data):
            if self.d == 2:
                self.kpoint = Kpoint2D(data)
            else:
                self.kpoint = Kpoint3D(data)

    @property    
    def keypoints(self) -> np.array:
        self.parse_json()
        return self.kpoint.key_data


def main():
    print(Path(DATA_ROOT_PATH_AIHUB)/ os.listdir(Path(DATA_ROOT_PATH_AIHUB))[0])
    print(Path(SAMPLE_DATA_PATH).parts)



if __name__ == '__main__':
    # main()
    a = Parser(Path(SAMPLE_DATA_PATH))
    # a.parse_json()
    # print(a.keypoints.shape)
    cls_dir= DataPath(class_limit=10).class_dict
    st = time.time()
    key_dir = defaultdict(list)
    for vid_dir in cls_dir[list(cls_dir.keys())[0]]:
        print(KeypointSeq(vid_dir,list(cls_dir.keys())[0]).key_arr.shape)
        break
        # arr = KeypointSeq(list(cls_dir.keys())[0],vid_dir).key_arr
        # key_dir[0].append(arr)

    # print(sys.getsizeof(key_dir))

    print(time.time()-st)
