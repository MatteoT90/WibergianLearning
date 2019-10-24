# -*- coding: utf-8 -*-
"""
Created on Sep 20 11:43 2018

@author: Matteo Toso
"""

import numpy as np
import h5py
import os
# from utils.draw import *
from utils.config import *
SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
SUBJECT_TRAIN = [1, 5, 6, 7, 8]
SUBJECT_TEST = [9, 11]

DIR_PATH = DATA_DIR + '../h36m/'

# Creating the subdirectories for the training and testing set.

if not os.path.exists(DATA_DIR + 'training'):
        os.makedirs(DATA_DIR + 'training')
if not os.path.exists(DATA_DIR + 'testing'):
        os.makedirs(DATA_DIR + 'testing')

ACTIONS = ["Directions", "Discussion", "Eating", "Greeting",
           "Phoning", "Photo", "Posing", "Purchases",
           "Sitting", "SittingDown", "Smoking", "Waiting",
           "WalkDog", "Walking", "WalkTogether",
           "Directions_1", "Discussion_1", "Eating_1", "Greeting_1",
           "Phoning_1", "Photo_1", "Posing_1", "Purchases_1",
           "Sitting_1", "SittingDown_1", "Smoking_1", "Waiting_1",
           "WalkDog_1", "Walking_1", "WalkTogether_1"]
CAMERAS = [54138969, 55011271, 58860488, 60457274]
JOINTS = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]


def create_set(actor_set, mode_dir, mode_name):

    poses = []
    detections = []
    all_cameras = []

    for action in ACTIONS:

        action_poses = []
        action_detections = []
        action_all_cameras = []

        for subject in actor_set:
            if action == 'Directions' and subject == 11:
                continue
	    # print 'Loading subject', subject, ', action ', action

            path_3d = DIR_PATH + 'S' + str(subject) + '/MyPoses/3D_positions/' + action + '.h5'
            f = h5py.File(path_3d, 'r')
            poses_3d = f['3D_positions'].value
            f.close()

            JOINT_SELECT = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
            # We keep only 17 keypoints
            p3d = np.transpose(np.reshape(poses_3d, (32, 3, -1))[JOINT_SELECT], (2, 0, 1))
            ll = len(p3d)
            det = []
            cams = []
            for cam in range(4):

                cameras = h5py.File(DIR_PATH + 'cameras.h5', 'r')
                rot = cameras['subject' + str(subject) + '/camera' + str(cam+1) + '/R'].value
                cameras.close()
                path_2d = DIR_PATH + 'S' + str(subject) + '/StackedHourglass/' + action + '.' + str(CAMERAS[cam]) + '.h5'
                f = h5py.File(path_2d, 'r')
                p2d = f['poses'].value
                f.close()
                nn = len(p2d)
                # The original stacked hourglass detections have only 16 keypoints. We reorder them and set the
                # missing joint to zero. It will be considered occluded.
                pos = np.zeros((ll, 17, 2))
                pos[:, JOINTS] = p2d
		# if nn != ll:
		#     print 'Something is amiss', action, subject, ' ', cam, ' ', nn, ' ', ll
                cameras = np.zeros((nn, 3, 3))
                cameras[:] = rot.T
                det.append(pos)
                cams.append(cameras)
            if subject == actor_set[0]:

                action_poses = np.reshape(p3d, [-1, 17, 3])
                action_detections = np.reshape(np.transpose(det, (1, 0, 2, 3)), [-1, 4, 17, 2])
                action_all_cameras = np.reshape(np.transpose(cams, (1, 0, 2, 3)), [-1, 4, 3, 3])

            else:
                t1 = np.reshape(np.transpose(det, (1, 0, 2, 3)), [-1, 4, 17, 2])
                t2 = np.reshape(np.transpose(cams, (1, 0, 2, 3)), [-1, 4, 3, 3])
                t3 = np.reshape(np.array(p3d), [-1, 17, 3])

                action_detections = np.concatenate((action_detections, t1), axis=0)
                action_all_cameras = np.concatenate((action_all_cameras, t2), axis=0)
                action_poses = np.concatenate((action_poses, t3), axis=0)

        # The lists are stored in the new file.
        h5name = DATA_DIR + mode_dir + action + '_sh.h5'
        f = h5py.File(h5name, 'w')
        f['joint_2d'] = action_detections
        f['joint_3d'] = action_poses
        f['camera'] = action_all_cameras
        f.close()

        if action == ACTIONS[0]:
            poses = action_poses
            detections = action_detections
            all_cameras = action_all_cameras
        else:
            detections = np.concatenate((detections, action_detections), axis=0)
            all_cameras = np.concatenate((all_cameras, action_all_cameras), axis=0)
            poses = np.concatenate((poses, action_poses), axis=0)

    # The lists are stored in the new file.
    h5name = DATA_DIR + mode_name + '.h5'
    f = h5py.File(h5name, 'w')
    f['joint_2d'] = detections
    f['joint_3d'] = poses
    f['camera'] = all_cameras
    f.close()


if __name__ == "__main__":
    create_set(SUBJECT_TRAIN, 'training/', 'train_sh')
    create_set(SUBJECT_TEST, 'testing/', 'test_sh')
