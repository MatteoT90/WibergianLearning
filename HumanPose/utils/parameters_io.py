# -*- coding: utf-8 -*-
"""
Created on Sep 25 2019

@author: Matteo Toso'
"""

import tensorflow as tf
import numpy as np
import h5py
from config import *
from random import shuffle


def check_point_save(param, name, epoch):
    h5name = CHK_DIR + name + '.h5'
    f = h5py.File(h5name, 'w')
    f['mean_pose'] = param[0]
    f['base_vectors'] = param[1]
    f['mean_pose_rec'] = param[2]
    f['base_vectors_rec'] = param[3]
    f['joint_weights'] = param[4]
    f['exp_weights'] = param[5]
    f['exp_weights_model'] = param[6]
    f['exp_additive_model'] = param[7]
    f['scaling_coefficient'] = param[8]
    f['lambda_0'] = param[9]
    f['lambda_scale'] = param[10]
    f['scale_regularizer'] = param[11]
    f['warping_weights'] = param[12]
    f['sigma_variance'] = param[13]
    f['cameras'] = param[14]
    f['hl_params'] = param[15]
    f['ir_reg'] = param[16]
    f['epoch'] = epoch
    f.close()


def initialization_params(n_models, mu, mod_mu, e, mod_e, np_lambda, mod_lambda, number_joints=17):
    mu_t = tf.cast(mu/mod_mu, dtype=tf.float64)
    e_t = tf.cast(e/mod_e, dtype=tf.float64)
    mu_rec = tf.cast(mu/mod_mu, dtype=tf.float64)
    e_rec = tf.cast(e / mod_e, dtype=tf.float64)
    weights_coefficients = tf.ones([n_models, number_joints * 2], dtype=tf.float64)
    exp_weights = tf.ones([n_models, 1], dtype=tf.float64)
    exp_weights_model = tf.ones([n_models, 1], dtype=tf.float64)
    exp_gamma_model = 0.0000001 + tf.ones([n_models, 1], dtype=tf.float64)
    magic_number = tf.cast(0.97, dtype=tf.float64)
    lambda_0 = 10.0*tf.ones([n_models, 1], dtype=tf.float64)
    lambda_scale = tf.ones([n_models, 1], dtype=tf.float64)
    scale_reg_obj = tf.cast(1, dtype=tf.float64)
    warping_weights = tf.ones([2, number_joints], dtype=tf.float64)
    sigma = tf.cast(np_lambda/mod_lambda, dtype=tf.float64)
    camera_mat = tf.ones([4, 3, 3], dtype=tf.float64)
    return [mu_t, e_t, mu_rec, e_rec, weights_coefficients, exp_weights, exp_weights_model, exp_gamma_model,
            magic_number, lambda_scale, lambda_0, scale_reg_obj, warping_weights, sigma, camera_mat]


def check_point_load(name):
    h5name = CHK_DIR + name
    f = h5py.File(h5name, 'r')

    o1 = tf.cast(f['mean_pose'].value, dtype=tf.float64)
    o2 = tf.cast(f['base_vectors'].value, dtype=tf.float64)
    o3 = tf.cast(f['mean_pose_rec'].value, dtype=tf.float64)
    o4 = tf.cast(f['base_vectors_rec'].value, dtype=tf.float64)
    o5 = tf.cast(f['joint_weights'].value, dtype=tf.float64)
    o6 = tf.cast(f['exp_weights'].value, dtype=tf.float64)
    o7 = tf.cast(f['exp_weights_model'].value, dtype=tf.float64)
    o8 = tf.cast(f['exp_additive_model'].value, dtype=tf.float64)
    o9 = tf.cast(f['scaling_coefficient'].value, dtype=tf.float64)
    o10 = tf.cast(f['lambda_0'].value, dtype=tf.float64)
    o11 = tf.cast(f['lambda_scale'].value, dtype=tf.float64)
    o12 = tf.cast(f['scale_regularizer'].value, dtype=tf.float64)
    o13 = tf.cast(f['warping_weights'].value, dtype=tf.float64)
    o14 = tf.cast(f['sigma_variance'].value, dtype=tf.float64)
    o15 = tf.cast(f['cameras'].value, dtype=tf.float64)
    o16 = tf.cast(f['hl_params'].value, dtype=tf.float64)
    o17 = tf.cast(f['ir_reg'].value, dtype=tf.float64)
    f.close()

    return o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17


def check_point_load_old(name):
    h5name = CHK_DIR + name
    f = h5py.File(h5name, 'r')
    o1 = tf.cast(f['base_vectors'].value, dtype=tf.float64)
    o2 = tf.cast(f['mean_pose'].value, dtype=tf.float64)
    o3 = tf.cast(f['joint_weights'].value, dtype=tf.float64)
    o4 = tf.cast(f['exp_weights'].value, dtype=tf.float64)
    o5 = tf.cast(f['exp_weights_model'].value, dtype=tf.float64)
    o6 = tf.cast(f['exp_additive_model'].value, dtype=tf.float64)
    o7 = tf.cast(f['scaling_coefficient'].value, dtype=tf.float64)
    o8 = tf.cast(f['s2'].value, dtype=tf.float64)
    o9 = tf.cast(f['s1'].value, dtype=tf.float64)
    o10 = tf.cast(f['scale_regularizer'].value, dtype=tf.float64)
    o11 = tf.cast(f['warping_weights'].value, dtype=tf.float64)
    o12 = tf.cast(f['sigma'].value, dtype=tf.float64)
    o13 = tf.cast(f['mean_pose_rec'].value, dtype=tf.float64)
    o14 = tf.cast(f['base_rec'].value, dtype=tf.float64)
    o15 = tf.cast(f['epoch'].value, dtype=tf.float64)
    o16 = tf.cast(f['cameras'].value, dtype=tf.float64)
    f.close()

    return o1, o2, o13, o14, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o16


def data_shuffle(p2d, p3d, cam, batch_size):
    n_joints = np.shape(p2d)[-2]
    p2d = np.reshape(p2d, (-1, 4, n_joints, 2))
    p3d = np.reshape(p3d, (-1, 1, n_joints, 3))
    cam = np.reshape(cam, (-1, 4, 3, 3))
    index = range(len(p2d))
    shuffle(index)
    p2d = np.reshape(p2d[index], (-1, batch_size * 4, n_joints, 2))
    p3d = np.reshape(p3d[index], (-1, batch_size * 1, n_joints, 3))
    cam = np.reshape(cam[index], (-1, batch_size * 4, 3, 3))

    return p2d, p3d, cam


def data_load(file_path):
    input_file = h5py.File(file_path, 'r')
    p2d = input_file['joint_2d'].value
    p3d = input_file['joint_3d'].value
    cam = input_file['camera'].value
    input_file.close()
    p2d = np.reshape(p2d, (-1, 4, 17, 2))
    p3d = np.reshape(p3d, (-1, 1, 17, 3))
    cam = np.reshape(cam, (-1, 4, 3, 3))

    return p2d[::5], p3d[::5], cam[::5], len(p2d[::5])


def data_load_16(file_path, all_joints):
    input_file = h5py.File(file_path, 'r')
    p2d = input_file['joint_2d'].value
    p3d = input_file['joint_3d'].value
    cam = input_file['camera'].value
    input_file.close()
    p2d = np.reshape(p2d, (-1, 17, 2))
    p3d = np.reshape(p3d, (-1, 17, 3))
    cam = np.reshape(cam, (-1, 3, 3))

    return p2d[::5, all_joints], p3d[::5, all_joints], cam[::5], len(p2d[::5])


def action_data_load(action, train=False):

    if train:
        subdir = 'training/'
    else:
        subdir = 'testing/'

    input_file_0 = h5py.File(SH_DIR + subdir + action + '_sh.h5', 'r')
    input_file_1 = h5py.File(SH_DIR + subdir + action + '_1_sh.h5', 'r')
    idx = 0
    p2d_0 = input_file_0['joint_2d'].value[idx::5]
    p3d_0 = input_file_0['joint_3d'].value[idx::5]
    cam_0 = input_file_0['camera'].value[idx::5]
    p2d_1 = input_file_1['joint_2d'].value[idx::5]
    p3d_1 = input_file_1['joint_3d'].value[idx::5]
    cam_1 = input_file_1['camera'].value[idx::5]
    input_file_0.close()
    input_file_1.close()
    p2d = np.concatenate([p2d_0, p2d_1], axis=0)
    p3d = np.concatenate([p3d_0, p3d_1], axis=0)
    cam = np.concatenate([cam_0, cam_1], axis=0)
    return p2d, p3d, cam, len(p2d)


def action_mask():
    mask = []
    subdir = 'test_actionwise/'
    for idx, action in enumerate(ACTIONS):
        input_file_0 = h5py.File(SH_DIR + subdir + action + '_sh.h5', 'r')
        input_file_1 = h5py.File(SH_DIR + subdir + action + '_1_sh.h5', 'r')
        if idx in [1, 9]:
            shape_0 = idx * 2 * np.ones(len(input_file_0))
            shape_1 = (idx * 2 + 1) * np.ones(len(input_file_1))
        else:
            shape_0 = idx * 2 * np.ones(len(input_file_1))
            shape_1 = (idx * 2 + 1) * np.ones(len(input_file_0))
        mask = np.concatenate([mask, shape_0, shape_1], axis=0)
        input_file_0.close()
        input_file_1.close()
    return mask


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
