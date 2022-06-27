import cv2

def show_vid(vArr, title = 'Images'):
    for a in vArr:
        cv2.imshow(title, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        cv2.waitKey(10)