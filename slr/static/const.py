from pathlib import Path

FACE_FEATURES = 468
POSE_FEATURES = 33
HAND_FEATURES = 21
DEFAULT_VID_ROOT = "/Users/0hyun/Desktop/vid"
EXCEPT_DIR = [".DS_Store", "a", "_all"]

ROOT = Path(r"D:\ksl\ksl\수어 영상\1.Training")

TGCN_INOUT_CHANNELS_ver1 = [
    (256, 256, 1),
    (256, 256, 2),
    (256, 512, 1),
    (512, 512, 2),
    (512, 512, 1),
    (512, 256, 1),
]
TGCN_INOUT_CHANNELS_ver2 = [
    (128, 128, 1),
    (128, 128, 2),
    (128, 256, 1),
    (256, 256, 2),
    (256, 256, 1),
    (256, 128, 1),
]
