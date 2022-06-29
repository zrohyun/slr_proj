import enum

import traceback
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from .const import *
from abc import *

class AbsAnimatedScatter(metaclass=ABCMeta):
    kstream: np.ndarray

    # each element is a head index of the feature
    lable_dict: dict = {
        'face': FACE_FEATURES,
        'pos_lh': FACE_FEATURES + 21,
        'pos_rh':FACE_FEATURES + 22,
        'lh':FACE_FEATURES + POSE_FEATURES,
        'rh':FACE_FEATURES + POSE_FEATURES + HAND_FEATURES,
    }

    def __init__(self,bodypoints):
        self.bodypoints = bodypoints
    
    @abstractclassmethod
    def setup_plot(self):
        pass

    @abstractclassmethod
    def update(self,i):
        pass

    def show(self):
        try:
            plt.show()
        except Exception as e:   
            print(e)
            print(traceback.format_exc())

class AnimatedScatter2D(AbsAnimatedScatter):
    def __init__(self,bodypoints,title=''):
        super().__init__(bodypoints)
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0,2), ylim=(-2,2))
        self.ax.set_title(title)

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=10, frames=bodypoints, 
                                                init_func=self.setup_plot, blit=False, repeat=False)      
        self.kstream = self.ani.new_frame_seq()

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        k = next(self.kstream)
        self.ax.invert_yaxis()
        self.scat = self.ax.scatter(k[:,0],k[:,1])

        self.txt = [self.ax.text(k[idx,0],k[idx,1],lable) for lable,idx in self.lable_dict.items()]
        # self.ax.set_xlim([0,1])
        # self.ax.set_ylim([0,1])


        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def update(self,i):
        """Update the scatter plot."""

        # Set x and y data...
        self.scat.set_offsets(i[:,:2])
        for (l,idx),t in zip(self.lable_dict.items(), self.txt):
            t.set_position(i[idx,:2])

        
        # Set sizes...
        # self.scat.set_sizes(300 * abs(k[:, 2])**1.5 + 100)
        # Set colors..
        # self.scat.set_array(k[:, 3])

        if np.array_equal(i,self.bodypoints[-1,:,:]):
            # self.ani.event_source.stop()
            # del self.ani
            plt.close() # force shutdown spit an error(AttributeError)

        
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

class AnimatedScatter3D(AbsAnimatedScatter):
    
    def __init__(self, bodypoints, title = ""):
        super().__init__(self,bodypoints)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,projection='3d')
        self.ax.set_title(title)
        self.ani = animation.FuncAnimation(self.fig, self.update3d, interval=10, frames=bodypoints, 
                                            init_func=self.setup, blit=False, repeat=False)   
        self.kstream = self.ani.new_frame_seq()
            

    def setup_plot(self):
        k = next(self.kstream)
        self.ax.view_init(45,45)
        self.scat = self.ax.scatter(k[:,0],k[:,1],k[:,2])
        self.txt = [self.ax.text(k[idx,0],k[idx,1],k[idx,2],lable) for lable,idx in self.lable_dict.items()]        
        return self.scat,

    def update(self,i):
        self.scat._offsets3d=(i[:,0],i[:,1],i[:,2])
        for (l,idx),t in zip(self.lable_dict.items(), self.txt):
            t.set_position(i[idx,:])

        return self.scat,