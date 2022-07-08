from matplotlib import pyplot as plt

from slr.data.asl.BodyPoint import BodyPoint
import os,sys
from slr.utils.plt import *
import slr.utils.vid as utv
import time
sys.path.append(os.pardir)

def main():
    pass

if __name__ == '__main__':
    main()
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