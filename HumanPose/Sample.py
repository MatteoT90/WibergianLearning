# -*- coding: utf-8 -*-
"""
Created on Sep 25 11:43 2019

@author: Matteo Toso
"""

from utils.wieberg import Prob3dPose
from utils.math_operations import compute_error
from utils.parameters_io import *
from utils.draw import *
import numpy as np

if __name__ == "__main__":

    p1 = [[483., 450.], [503., 450.], [503., 539.], [496., 622.], [469., 450.], [462., 546.],
          [469., 622.], [483., 347.], [483., 326.], [0., 0.], [489., 264.], [448., 347.],
          [448., 408.], [441., 463.], [517., 347.], [524., 408.], [538., 463.]]

    p2 = [[570., 404.], [550., 404.], [550., 479.], [550., 555.], [591., 404.], [584., 486.],
          [584., 562.], [570., 314.], [564., 294.], [0., 0.], [557., 232.], [605., 308.],
          [605., 363.], [619., 411.], [536., 314.], [529., 363.], [522., 411.]]

    p3 = [[609., 374.], [623., 374.], [630., 470.], [630., 546.], [588., 374.], [588., 470.],
          [602., 546.], [609., 278.], [609., 257.], [0., 0.], [609., 195.], [568., 278.],
          [568., 339.], [554., 388.], [643., 278.], [650., 339.], [657., 394.]]

    p4 = [[507., 355.], [480., 355.], [473., 458.], [466., 561.], [528., 362.], [514., 458.],
          [500., 554.], [507., 238.], [507., 211.], [0., 0.], [514., 162.], [549., 238.],
          [542., 300.], [507., 348.], [466., 238.], [452., 307.], [432., 369.]]

    p3d = [[-64.02449799,  234.28799438,  936.80401611], [-194.34162903,  209.85072327,  946.58123779],
           [-175.16213989,  241.63711548,  505.24526978], [-183.36831665,  308.24963379,   56.02492523],
           [66.29286194,  258.72531128,  927.02679443], [33.84835815,  272.51428223,  485.53762817],
           [7.85348368,  328.86349487,   35.59024048], [-79.81334686,  266.91677856, 1167.44372559],
           [-79.80996704,  268.46591187, 1424.51672363], [-92.38868713,  203.63616943, 1526.06738281],
           [-97.50853729,  285.3243103, 1606.85327148], [50.91806793,  289.73007202, 1351.9263916],
           [111.53924561,  341.23425293, 1084.62878418], [164.3368988,  393.09893799,  844.02081299],
           [-209.54525757,  262.54519653, 1347.41809082], [-277.17016602,  299.09204102, 1079.32763672],
           [-335.82885742,  321.83474731,  835.58752441]]

    poses_2d = np.array([p1, p2, p3, p4])
    poses_3d = np.array(p3d)

    visibility_mask = np.ones((4, 2, H36M_NUM_JOINTS), dtype=bool)
    visibility_mask[:, :, 9] = 0

    for i in range(3):
        plot_2d_pose_16(poses_2d[i])

    # Building the graph
    poses2d = tf.placeholder(tf.float64, shape=[4, 17, 2])
    gt_pose = tf.placeholder(tf.float64, shape=[1, 17, 3])
    # lifter = Prob3dPose(models=3, init_path=None)
    lifter = Prob3dPose(models=3, init_path='wiberg_results/psh_mc_2.h5')

    true_pose, scale = lifter.normalise_gt(gt_pose)
    rec, n_pose, cct, sc = lifter.compute_3d(poses2d, visibility_joints=visibility_mask)
    reconstruction = tf.transpose(scale * tf.transpose(rec, [1, 2, 0]), [2, 0, 1])
    errors, aligned_p, t1, t2 = compute_error(true_pose, reconstruction)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        feed_dict = {
            poses2d: poses_2d,
            gt_pose: np.expand_dims(poses_3d, axis=0),
        }

        rec_pose, truth, err = \
            sess.run([reconstruction, true_pose, errors ], feed_dict=feed_dict)
    plot_poses_overlapping(np.concatenate([truth, rec_pose], axis=0),
                           'Reconstructed (Trained) vs GT - Error [err=%0.3fmm]'%err[2])
