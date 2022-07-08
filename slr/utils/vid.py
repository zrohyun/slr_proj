import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_vid(vArr: np.ndarray, title = 'Images'):
    for a in vArr:
        cv2.imshow(title, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        cv2.waitKey(10)

def plot_2D_keypoint_every_move(keypoints, with_face = False, save=False,):

    # if not with_face: keypoints = keypoints[:,FACE_FEATUERS:,:]
    figure ,ax = plt.subplots(figsize=(8,6))
    axsca = ax.scatter(keypoints[0,:,0],keypoints[0,:,1])
    t1 = ax.text(keypoints[0,468,0],keypoints[0,468,1],'face')
    
    ax.set_xlim([keypoints[:,:,0].min(),keypoints[:,:,0].max()])
    ax.set_ylim([keypoints[:,:,1].min(),keypoints[:,:,1].max()])
    ax.invert_yaxis()

    for n,k in enumerate(keypoints):
        #print(i[:,:1].shape)
        axsca.set_offsets(k[:,:2])
        t1.set_position((k[468,0],k[468,1]))
        figure.canvas.draw_idle() 
        plt.pause(0.1)
        if save and with_face: plt.savefig(f"./scatter_gif/{save} ({n}).png")
        elif save and not with_face: plt.savefig(f"./scatter_gif/{save}_without_face ({n}).png")
    plt.close()