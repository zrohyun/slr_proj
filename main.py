from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# from slr.data.asl.BodyPoint import BodyPoint
import os,sys
from slr.data.datagenerator import KeyDataGenerator
from slr.data.ksl.datapath import DataPath
# from slr.model.convnet import mobileNetV2, resnet50
# from slr.utils.plt import *
# from slr.utils.utils import get_tensorboard_callback
# import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard
# from slr.utils import AbsAnimatedScatter



sys.path.append(os.pardir)

def main():
    x,y = DataPath(class_limit=10).data

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.3)

    train_generator = KeyDataGenerator(x_train,y_train,64)
    print(train_generator[0][0].shape)
    # print(np.array(train_generator[0][0] / train_generator[0][0].max(axis=(1,2))[:,np.newaxis,np.newaxis,:]).shape)

    pass

if __name__ == '__main__':
    main()
    # model = mobileNetV2()
    
    # a = VidData()
    # print(a)
     
    # for n,(i,lable) in enumerate(a.load_video(False)):
    #     print(lable)
    #     a = BodyPoint(i,False).gen_bodypoint()
    #     print(a.shape)
    #     print(type(a))

    #     k = AnimatedScatter2D(a,title=lable)
    #     k.save()
        # utv.plot_2D_keypoint_every_move(a)




# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np

# class AnimatedScatter(object):
#     """An animated scatter plot using matplotlib.animations.FuncAnimation."""
#     def __init__(self, numpoints=50):
#         self.numpoints = numpoints
#         self.stream = self.data_stream()

#         # Setup the figure and axes...
#         self.fig, self.ax = plt.subplots()
#         # Then setup FuncAnimation.
#         self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
#                                           init_func=self.setup_plot, blit=True)

#     def setup_plot(self):
#         """Initial drawing of the scatter plot."""
#         x, y, s, c = next(self.stream).T
#         self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
#                                     cmap="jet", edgecolor="k")
#         self.ax.axis([-10, 10, -10, 10])
#         # For FuncAnimation's sake, we need to return the artist we'll be using
#         # Note that it expects a sequence of artists, thus the trailing comma.
#         return self.scat,

#     def data_stream(self):
#         """Generate a random walk (brownian motion). Data is scaled to produce
#         a soft "flickering" effect."""
#         xy = (np.random.random((self.numpoints, 2))-0.5)*10
#         s, c = np.random.random((self.numpoints, 2)).T
#         while True:
#             xy += 0.03 * (np.random.random((self.numpoints, 2)) - 0.5)
#             s += 0.05 * (np.random.random(self.numpoints) - 0.5)
#             c += 0.02 * (np.random.random(self.numpoints) - 0.5)
#             yield np.c_[xy[:,0], xy[:,1], s, c]

#     def update(self, i):
#         """Update the scatter plot."""
#         data = next(self.stream)

#         # Set x and y data...
#         self.scat.set_offsets(data[:, :2])
#         # Set sizes...
#         self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
#         # Set colors..
#         self.scat.set_array(data[:, 3])

#         # We need to return the updated artist for FuncAnimation to draw..
#         # Note that it expects a sequence of artists, thus the trailing comma.
#         return self.scat,


# if __name__ == '__main__':
#     a = AnimatedScatter()
#     plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# TWOPI = 2*np.pi

# fig, ax = plt.subplots()

# t = np.arange(0.0, TWOPI, 0.001)
# s = np.sin(t)
# l = plt.plot(t, s)

# ax = plt.axis([0,TWOPI,-1,1])

# redDot, = plt.plot([0], [np.sin(0)], 'ro')

# def animate(i):
#     redDot.set_data(i, np.sin(i))
#     return redDot,

# # create animation using the animate() function
# myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(0.0, TWOPI, 0.1), \
#                                       interval=10, blit=True, repeat=True)

# plt.show()