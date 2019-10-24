# -*- coding: utf-8 -*-
"""
Created on Sep 20 11:43 2018

@author: Matteo Toso
"""

# from utils.wieberg import Prob3dPose
from utils.math_operations import compute_error
from utils.parameters_io import *
from utils.settings import parse_parameters
from utils.draw import *
import time


def test():
    """
    """
    FLAGS = parse_parameters()
    if FLAGS.l2_solver:
        from utils.wieberg_l2 import Prob3dPose
    else:
        from utils.wieberg import Prob3dPose

    visibility_mask = np.ones((4, 2, H36M_NUM_JOINTS), dtype=bool)
    visibility_mask[:, :, 9] = 0

    print 'Data set loaded, building graph.'

    # Building the graph
    poses2d = tf.placeholder(tf.float64, shape=[4, 17, 2])
    cameras = tf.placeholder(tf.float64, shape=[4, 3, 3])
    gt_pose = tf.placeholder(tf.float64, shape=[1, 17, 3])
    lifter = Prob3dPose(models=FLAGS.models, init_path=FLAGS.check_path)

    true_pose, scale = lifter.normalise_gt(gt_pose)
    rec, n_pose, cct, sc = lifter.compute_3d(poses2d, cam=cameras, visibility_joints=visibility_mask)
    reconstruction = tf.transpose(scale * tf.transpose(rec, [1, 2, 0]), [2, 0, 1])
    errors, aligned_p, t1, t2 = compute_error(true_pose, reconstruction)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        action_results = []
        mean_result = 0
        tot_frames = 0
        all_errors = []
        all_poses = []
        all_gt = []
        act_err = []
        for action in ACTIONS:
            detection, gt, cams, n_frames = action_data_load(action)
            current_error = []
            start = time.time()
            for pose_idx in range(n_frames):
                feed_dict = {
                    poses2d: detection[pose_idx],
                    gt_pose: np.expand_dims(gt[pose_idx], axis=0),
                }

                reconstruction_error, truth, new_pose = \
                    sess.run([errors, true_pose, reconstruction], feed_dict=feed_dict)
                # print reconstruction_error
                current_error.append(reconstruction_error[FLAGS.metric])
                all_errors.append(reconstruction_error[FLAGS.metric])
                all_poses.append(new_pose)
                all_gt.append(truth)
            end = time.time()
            action_results.append(np.mean(current_error))
            tot_frames += n_frames
            action_e = np.mean(current_error)
            mean_result += n_frames * action_e
            print 'Action ', action, ' :: -> ', action_e
            print '# frames: ', n_frames, 'Elapsed: ', end - start, 'ET 1 frame: ', (end - start)/n_frames
            act_err.append(action_e)
        print 'Mean error :: -> ', np.sum(mean_result)/tot_frames
        act_err.append(np.sum(mean_result)/tot_frames)
        print ' over ', tot_frames
        np.save(CHK_DIR+FLAGS.name+'/errors', all_errors)
        np.save(CHK_DIR+FLAGS.name+'/poses', all_poses)
        np.save(CHK_DIR+FLAGS.name+'/truth', all_gt)
        np.save(CHK_DIR+FLAGS.name+'/tabled_res', act_err)
        print " {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} &" \
              " {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\".format(*act_err)
