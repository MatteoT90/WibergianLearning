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


def train():
    """
    """
    FLAGS = parse_parameters()
    visibility_mask = np.ones((FLAGS.batch_size*4, 2, H36M_NUM_JOINTS), dtype=bool)
    visibility_mask[:, :, 9] = 0
    if FLAGS.l2_solver:
        from utils.wieberg_l2 import Prob3dPose
    else:
        from utils.wieberg import Prob3dPose

    print 'Starting network training...'

    print 'Loading data set...'
    # Loading the training set
    detection, gt, cams, n_frames = data_load(FLAGS.input_path)

    print 'Data set loaded, building graph.'

    # Building the graph
    poses2d = tf.placeholder(tf.float64, shape=[FLAGS.batch_size * 4, 17, 2])
    cameras = tf.placeholder(tf.float64, shape=[FLAGS.batch_size * 4, 3, 3])
    gt_pose = tf.placeholder(tf.float64, shape=[FLAGS.batch_size * 1, 17, 3])
    total_error = tf.placeholder(tf.float64, shape=[])
    block_error = tf.placeholder(tf.float64, shape=[])

    lifter = Prob3dPose(models=FLAGS.models, init_path=FLAGS.check_path)

    true_pose, scale = lifter.normalise_gt(gt_pose)
    rec, _, _, sc = lifter.compute_3d(poses2d, cam=cameras, visibility_joints=visibility_mask)
    reconstruction = tf.transpose(scale * tf.transpose(rec, [1, 2, 0]), [2, 0, 1])
    rec_error, aligned, t1, t2 = compute_error(true_pose, reconstruction)

    print 'Evaluating on metric ', FLAGS.metric
    mean_error = rec_error[FLAGS.metric]

    tf.summary.scalar("errors/current_error", mean_error)
    tf.summary.scalar("errors/epoch_error", total_error)
    tf.summary.scalar("errors/error_from_last", block_error)

    all_params = lifter.return_model()

    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.wlr, beta1=0.9, beta2=0.999, epsilon=0.1)
    # opt = tf.train.GradientDescentOptimizer(learning_rate=0.005)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    grads_and_vars = opt.compute_gradients(mean_error)
    train_op = opt.minimize(mean_error, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    clip_op = tf.group(
        lifter.weights_coefficients.assign(tf.clip_by_value(lifter.weights_coefficients, 0.001, 100000)),
        lifter.exp_weights.assign(tf.clip_by_value(lifter.exp_weights, 0.00001, 1000)),
        lifter.exp_weights_model.assign(tf.clip_by_value(lifter.exp_weights_model, 0.00001, 1000)),
        lifter.warping_weights.assign(tf.clip_by_value(lifter.warping_weights, 0.00001, 1)),
        lifter.sigma.assign(tf.clip_by_value(lifter.sigma, -5., 5.)),
        lifter.ir_reg.assign(tf.clip_by_value(lifter.ir_reg, 0.001, 5.)),
        lifter.hl_params.assign(tf.clip_by_value(lifter.hl_params, 0.001, 5.))
    )
    sum_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init_op)
        idx_0 = 0
        train_writer = tf.summary.FileWriter(CHK_DIR+FLAGS.name+'/logger/', sess.graph)
        n_batches = n_frames/FLAGS.batch_size
        print 'Starting training on ', n_batches, ' batches.'

        for j in range(FLAGS.epochs):
            print 'current epoch ', j
            detection, gt, cams = data_shuffle(detection, gt, cams, FLAGS.batch_size)
            current_error = []
            temp_error = []

            for pose_idx in range(n_batches):
                feed_dict = {
                    poses2d: detection[pose_idx],
                    cameras: cams[pose_idx],
                    gt_pose: gt[pose_idx]
                }

                _, reconstruction_error = sess.run([train_op, mean_error], feed_dict=feed_dict)

                current_error.append(reconstruction_error)
                temp_error.append(reconstruction_error)
                sess.run(clip_op)
                # print reconstruction_error
                if pose_idx % FLAGS.save_f == 0:
                    print '     Batch ', pose_idx, ' out of ', n_batches, '. Partial error :: ', np.mean(current_error)

                    feed_dict_all = {
                        poses2d: detection[pose_idx],
                        cameras: cams[pose_idx],
                        gt_pose: gt[pose_idx],
                        total_error: np.mean(current_error),
                        block_error: np.mean(temp_error)
                    }

                    params = sess.run(all_params)
                    check_point_save(params, FLAGS.name + '/check/partial_checkpoint', j)
                    summary_e = sess.run(sum_op, feed_dict=feed_dict_all)
                    train_writer.add_summary(summary_e, global_step=global_step.eval())
                    temp_error = []
            print 'Epoch ', j, ' completed with error ', np.mean(current_error)
            params = sess.run(all_params)
            check_point_save(params, FLAGS.name+'/check/epoch_' + str(j+idx_0), j)


def build_dir_tree(path_name):
    import os
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    log_path = path_name+'/logger/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    check_path = path_name+'/check/'
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    image_path = path_name+'/images/'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    results_path = path_name+'/evaluations/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

