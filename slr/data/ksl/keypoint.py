import numpy as np
from dataclasses import dataclass

class Keypoint():
    def __init__(self, key_data,dimension):
        self.d = dimension
        self.keypointlen = list(map(len, key_data)) # ordered face, pose, left_hand, right_hand
        self.key_data = np.concatenate(tuple(np.array(i).reshape((-1,self.d+1)) for i in key_data))
        
@dataclass
class Kpoint2D(Keypoint):
 
    def __init__(self, key_data):
        # dimension : x,y,confidence
        super().__init__(key_data, dimension=2)

@dataclass
class Kpoint3D(Keypoint):
    def __init__(self, key_data):
        # dimension: x,y,z,confidence
        super().__init__(key_data, dimension=3)

