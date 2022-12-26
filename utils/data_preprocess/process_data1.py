import argparse
import os
import cdflib
import cv2
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, required=True, choices=['3dpw', '3dhp', 'h36m', 'internet'],
# help='process which dataset?')
parser.add_argument('--dataset', type=str, default='h36m')

CAMERA_DICT = {
    '55011271': 'cam1',  # 1,8
    '58860488': 'cam2',  # 1,5,6,7--8
    '60457274': 'cam3',  # all
    '54138969': 'cam0',  # 1,8--5,6,7
}

DEBUG = False


# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_train_extract(dataset_path, training_split=True):
    # users in validation set
    if training_split:
        user_list = [8]
    else:
        user_list = [9, 11]

    # go over each user
    for user_i in tqdm(user_list, desc='user'):
        user_name = 'S%d' % user_i
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in tqdm(seq_list, desc='seq'):
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')  # action:Direction 1; camera: 54138969
            action = action.replace(' ', '_')  # action:Direction_1
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            # poses_3d = pycdf.CDF(seq_i)['Pose'][0]
            poses_3d = cdflib.CDF(seq_i)['Pose'][0]  # pose_3d:(1383,96)

            # video file
            vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
            imgs_path = os.path.join(dataset_path, 'images2')
            vidcap = cv2.VideoCapture(vid_file)  # 读取视频帧

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                success, image = vidcap.read()  # (1002,1000,3)
                if not success:
                    break

                # check if you can keep this frame
                if frame_i % 5 == 0 and camera == '58860488':  # and (protocol == 1 or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i + 1)

                    # save image
                    img_out = os.path.join(imgs_path, imgname)
                    cv2.imwrite(img_out, image)


if __name__ == '__main__':
    args = parser.parse_args()

    h36m_train_extract('/media/milan/TOSHIBA EXT/human36m_full_raw/training', training_split=True)
