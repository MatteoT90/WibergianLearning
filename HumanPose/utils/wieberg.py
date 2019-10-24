# -*- coding: utf-8 -*-
"""
Created on November 03 2018

@author: Matteo Toso
"""
import logging
from utils.math_operations import *
from utils.config import DATA_DIR, H36M_NUM_JOINTS, NUM_CAMERAS
from math import pi
from parameters_io import initialization_params, check_point_load
import tensorflow as tf


class Prob3dPose:
    def __init__(self, models=3, init_path=None):
        """
        Constructor
        :param models: number of models, as index to load the models parameters
        """
        import h5py
        path_params = DATA_DIR + '/model_parameters.h5'
        params = h5py.File(path_params, 'r')
        e = np.array(params['e_'+str(models)].value)
        mu = np.array(params['mu_'+str(models)].value)
        sigma = np.array(params['sigma_'+str(models)].value)
        params.close()
        f_cams = DATA_DIR + '/avg_cameras.h5'
        cams_file = h5py.File(f_cams, 'r')
        cam = cams_file['cams'].value
        cams_file.close()

        self.number_joints = 17

        self.n_models = e.shape[0]

        translations = np.zeros((self.n_models, 3, 3, self.number_joints))
        translations[:, 0, 0] = 1.0
        translations[:, 1, 1] = 1.0
        translations[:, 2, 2] = 1.0
        self.translations = tf.constant(translations)

        self._logger = logging.getLogger("Prob3dPose")

        cam_value = tf.constant(cam, dtype=tf.float64)

        low_lambda = np.zeros((sigma.shape[0], 4))
        low_lambda[:, :4] = 10 ** -5
        low_lambda[:, 0] = 1.2 * 0.0

        # Trainable parameters #

        self.mod_mu = tf.constant(np.mean(np.abs(mu)))
        self.mod_e = tf.constant(np.mean(np.abs(e)))
        self.mod_lambda_scale = tf.reshape(sigma[:, -1], [self.n_models, 1])

        l_diagonal = np.concatenate([low_lambda, sigma[:, :-1]], axis=1)
        np_lambda = []

        for i in l_diagonal:
            temp = np.diag(i)
            np_lambda.append(temp)

        np_lambda = np.array(np_lambda)
        self.mod_lambda = tf.constant(np.mean(np.abs(np_lambda)))

        if init_path is None:
            start_params = initialization_params(self.n_models, mu, self.mod_mu, e, self.mod_e, np_lambda,
                                                 self.mod_lambda)
        else:
            start_params = check_point_load(init_path)

        """ Network's Parameters. Set trainable to False to prevent optimization of that parameter"""

        """PPCA Parameters (mean pose, basis vector and variance)"""
        self.mu = tf.get_variable(name='mean_pose', initializer=start_params[0])
        self.e = tf.get_variable(name='base_vectors', initializer=start_params[1])
        self.sigma = tf.get_variable(name='sigma_variance', initializer=start_params[13], trainable=True)
        """ Not needed for this code, used in the single camera implementation"""
        self.mu_rec = tf.get_variable(name='mean_pose_rec', initializer=start_params[2], trainable=False)
        self.e_rec = tf.get_variable(name='base_vectors_rec', initializer=start_params[3], trainable=False)
        self.warping_weights = tf.get_variable(name='warping_weights', initializer=start_params[12], trainable=False)
        """Weights on the camera matrices and on the joint reprojection error"""
        self.weights_coefficients = tf.get_variable(name='joint_weights', initializer=start_params[4], trainable=True)
        self.camera_mat = tf.get_variable(name='cameras', initializer=start_params[14], trainable=True)
        """Scale factors for lifting problem, mean pose scale and final reconstruction"""
        self.magic_number = tf.get_variable(name='scaling_coefficient', initializer=start_params[8], trainable=True)
        self.lambda_0 = tf.get_variable(name='lambda_0', initializer=start_params[9], trainable=True)
        self.lambda_scale = tf.get_variable(name='lambda_scale', initializer=start_params[10], trainable=True)
        self.scale_reg_obj = tf.get_variable(name='scale_regularizer', initializer=start_params[11], trainable=True)
        """Parameters for the weighted averaging or poses from different rotations or models"""
        self.exp_weights = tf.get_variable(name='exp_weights', initializer=start_params[5], trainable=True)
        self.exp_weights_model = tf.get_variable(name='exp_weights_model', initializer=start_params[6], trainable=True)
        self.exp_gamma_model = tf.get_variable(name='exp_additive_model', initializer=start_params[7], trainable=True)
        """Parameters for the huber loss"""
        self.hl_params = tf.get_variable(name='hl_param', initializer=tf.ones([3], dtype=tf.float64), trainable=True)
        self.ir_reg = tf.get_variable(name='ir_reg', initializer=tf.ones([3], dtype=tf.float64), trainable=True)

        """
            self.ir_iterations = 3
            self.hl_params = 0.1
            self.ir_regulariser = 0.64
        """

        self.cam = tf.multiply(self.camera_mat, cam_value)
        """ The next lines add parameters to the logger, to monitor how they change during training """
        tf.summary.scalar("weights/rotation/0", self.exp_weights[0, 0])
        tf.summary.scalar("weights/rotation/1", self.exp_weights[1, 0])
        tf.summary.scalar("weights/rotation/2", self.exp_weights[2, 0])
        tf.summary.scalar("weights/model/0", self.exp_weights_model[0, 0])
        tf.summary.scalar("weights/model/1", self.exp_weights_model[1, 0])
        tf.summary.scalar("weights/model/2", self.exp_weights_model[2, 0])
        tf.summary.scalar("weights/additional/0", self.exp_gamma_model[0, 0])
        tf.summary.scalar("weights/additional/1", self.exp_gamma_model[1, 0])
        tf.summary.scalar("weights/additional/2", self.exp_gamma_model[2, 0])
        tf.summary.scalar("ir/hl_param/0", self.hl_params[0])
        tf.summary.scalar("ir/hl_param/1", self.hl_params[1])
        tf.summary.scalar("ir/hl_param/2", self.hl_params[2])
        tf.summary.scalar("ir/ir_reg/0", self.ir_reg[0])
        tf.summary.scalar("ir/ir_reg/1", self.ir_reg[1])
        tf.summary.scalar("ir/ir_reg/2", self.ir_reg[2])
        tf.summary.histogram("weights/joints/0", self.weights_coefficients[0])
        tf.summary.histogram("weights/joints/1", self.weights_coefficients[1])
        tf.summary.histogram("weights/joints/2", self.weights_coefficients[2])
        # tf.summary.histogram("weights/warping/0", self.warping_weights[0])
        # tf.summary.histogram("weights/warping/1", self.warping_weights[1])
        tf.summary.histogram("variance", self.sigma)
        tf.summary.scalar("coefficient/magic_scale", self.magic_number)
        tf.summary.scalar("coefficient/scale_regularizer", self.scale_reg_obj)
        tf.summary.scalar("coefficient/lambda_0/0", self.lambda_0[0, 0])
        tf.summary.scalar("coefficient/lambda_0/1", self.lambda_0[1, 0])
        tf.summary.scalar("coefficient/lambda_0/2", self.lambda_0[2, 0])
        tf.summary.scalar("coefficient/lambda_scale/0", self.lambda_scale[0, 0])
        tf.summary.scalar("coefficient/lambda_scale/1", self.lambda_scale[1, 0])
        tf.summary.scalar("coefficient/lambda_scale/2", self.lambda_scale[2, 0])

        self.lambda_values = tf.multiply(tf.expand_dims(self.lambda_0, 1), tf.multiply(self.mod_lambda, self.sigma))
        self.loss_t_c = tf.multiply(self.mod_lambda_scale, self.lambda_scale)

        self.bases = tf.concat(
            [tf.expand_dims(tf.multiply(self.mod_mu, self.mu), 1), self.translations,
             tf.multiply(self.mod_e, self.e)], axis=1)
        self.bases_rec = tf.concat(
            [tf.expand_dims(tf.multiply(self.mod_mu, self.mu_rec), 1), self.translations,
             tf.multiply(self.mod_e, self.e_rec)], axis=1)
        buff = tf.cast(np.ones((self.n_models, 29)), dtype=tf.float64)
        self.weights_c = tf.concat([self.weights_coefficients, buff], axis=1)

        # skeleton tree
        self._POSE_TREE = tf.constant(np.array([[0, 1], [1, 2], [2, 3], [0, 4],
                                                [4, 5], [5, 6], [0, 7], [7, 8],
                                                [8, 9], [8, 10], [10, 11],
                                                [11, 12], [8, 13], [13, 14], [14, 15]]).T, dtype=tf.int32)

        self.weights = None

        def _create_mask():
            _n_xz = 2
            _n_coordinates = 3
            mask = np.zeros((H36M_NUM_JOINTS * _n_coordinates, H36M_NUM_JOINTS * _n_xz))
            for index in range(H36M_NUM_JOINTS):
                for j in range(_n_xz):
                    for k in range(_n_coordinates):
                        mask[index * 3 + k, index * 2 + j] = 1
            return mask

        self.multi_camera_mask = tf.constant(_create_mask(), dtype=tf.float64)

    def return_model(self):
        return [self.mu, self.e, self.mu_rec, self.e_rec, self.weights_coefficients, self.exp_weights,
                self.exp_weights_model, self.exp_gamma_model, self.magic_number, self.lambda_0, self.lambda_scale,
                self.scale_reg_obj, self.warping_weights, self.sigma, self.camera_mat, self.hl_params, self.ir_reg]

    @staticmethod
    def _loss_function(x, ir_reg, hl_params, w_gradient=False):
        """
        This function contains the chosen loss function for the ir problem and its re weighting coefficients
        :Param x: vector of the residuals. [:51] res of 3D pose, [51:186] res of 2D reprojection, [186:] res of variance
        :Param w_gradient: if false, evaluate the actual loss, if true evaluates its gradient to compute the IR weights.
        The loss function used is l2 when the residuals are bigger than a certain threshold and the l1 norm
        (or the euclidean distance) otherwise.
        """

        ir_reg *= 0.75
        hl_params *= 0.1
        res = tf.pow(x+0.0000001, 2)

        if w_gradient:
            w1 = tf.ones(51, tf.float64)
            w2 = tf.pow((res[51:186:2]+res[52:187:2]) + 0.000001, 0.5)
            w_h = tf.reciprocal(tf.pow(1+tf.pow(w2/hl_params, 2), 0.5))
            w3 = ir_reg * tf.ones(29, tf.float64)
            err = tf.concat([w1, tf.reshape(tf.concat([w_h, w_h], 1), [136]), w3], 0)
            err = tf.diag(tf.sqrt(err))

        else:
            mod_error = res[:51]
            pov_error = res[51:187]
            pov_e = tf.pow((pov_error[::2]+pov_error[1::2]) + 0.000001, 0.5)
            reg_error = res[187:]
            err = tf.multiply(tf.square(hl_params), tf.pow(1+tf.pow(pov_e/hl_params, 2), 0.5)-1)
            mod = tf.reduce_sum(mod_error)/2
            reg = ir_reg * tf.reduce_sum(reg_error)/2
            err = mod + tf.reduce_sum(err) + reg
        return err

    def _ir_solver(self, a, b, hu1, hu2):
        x = tf.matrix_solve_ls(a, b, l2_regularizer=0.001, fast=True, name=None)
        for i in range(3):
            rec = tf.matmul(a, x)
            r_e = tf.subtract(rec, b)
            w = self._loss_function(r_e, hu1, hu2, w_gradient=True)
            wa = tf.matmul(w, a)
            wb = tf.matmul(w, b)
            x = tf.matrix_solve_ls(wa, wb, l2_regularizer=0.001, fast=True, name=None)

        rec_f = tf.matmul(a, x)
        err_f = tf.abs(tf.subtract(rec_f, b))
        error = self._loss_function(err_f, hu1, hu2)
        return x, error

    def _estimate_a_and_rotation(self, w, e, lambda_values, check, weights, training_p, exp_weight, scale_l, mu, p1, p2):
        """
        Solve optimisation problem by fixing the rotation rotation and solve the convex problem in a.
        :param w: 2D poses in the format (frames x 2 x n_joints)
        :param e: set of 3D pose bases in the format (n_bases x 3 x n_joints)
        :param lambda_values: variance coefficients used in the P.P.C.A.
        :param check: sampling around the circle to test the rotation at that point
        :param weights: weights used to define the importance of the variance terms (frames x (2 x n_joints))
        :return: a (bases coefficients)
                 rotation (representation of rotations as a complex number)
                 residual (reconstruction error when using the best rotation and base coefficients)
        """
        # NOTE: data here are processed by reshaping the data in the format -> xxx...xxx,yyy...zzz

        n_frames = w.get_shape().as_list()[0]
        n_points = w.get_shape().as_list()[-1]
        n_basis = e.get_shape().as_list()[0]
        n_rotations = check.get_shape().as_list()[0]
        d = lambda_values
        q = tf.diag(-tf.ones(shape=(H36M_NUM_JOINTS * 3), dtype=tf.float64))
        w_reshape = tf.reshape(tf.transpose(w, [0, 1, 3, 2]), [n_frames, NUM_CAMERAS, n_points * 2])
        weights_reshape = tf.reshape(tf.transpose(weights, [0, 2, 1]), [n_frames, 4, -1])

        pose_init = tf.TensorArray(dtype=tf.float64, size=n_rotations)
        score_init = tf.TensorArray(dtype=tf.float64, size=n_rotations)

        idx_rot_init = tf.constant(0, tf.int32)
        idx_cam_init = tf.constant(0, tf.int32)
        idx_batch_init = tf.constant(0, tf.int32)

        main_min_3d = tf.concat([
            tf.reshape(tf.transpose(e, [0, 2, 1]), shape=(n_basis, -1)),
            q], axis=0)

        main_min_obj = tf.reshape(tf.multiply(tf.multiply(w_reshape, scale_l), weights_reshape), [n_frames, -1])
        scale_reg_obj = tf.expand_dims(tf.expand_dims(self.scale_reg_obj * -0.003, -1), -1)
        res = tf.concat([tf.zeros(shape=(n_frames, H36M_NUM_JOINTS * 3, 1), dtype=tf.float64),
                         tf.expand_dims(main_min_obj, axis=-1),
                         scale_reg_obj * tf.ones(shape=(n_frames, 1, 1), dtype=tf.float64),
                         tf.zeros(shape=(n_frames, n_basis - 1, 1), dtype=tf.float64)], axis=1)

        jw = tf.concat([tf.ones([51], dtype=tf.float64), training_p[:34], training_p[:34], training_p[:34],
                        training_p[:34], training_p[34:]], axis=0)
        jw_diagonal = tf.diag(jw)

        def condition_cameras(i, *_):
            return i < NUM_CAMERAS

        def condition_rotations(i, *_):
            return i < n_rotations

        def condition_batch(i, *_):
            return i < n_frames

        def exec_rotations(idx_rotation, pose_base, score_base):
            """Run on a given P.P.C.A. model"""
            sin = tf.sin(check[idx_rotation])
            cos = tf.cos(check[idx_rotation])
            rot_mat = tf.transpose(upgrade_r(sin, cos))

            projection_e_init = tf.TensorArray(dtype=tf.float64, size=NUM_CAMERAS)

            def exec_cameras(idx_camera, idx_rot, projection_cams):
                """Run on individual frames"""

                g_rot = tf.transpose(tf.matmul(self.cam[idx_camera], rot_mat)[:2], [1, 0])
                cam_rot = tf.tile(g_rot, [17, 17])
                cam_weights = tf.tile(tf.expand_dims(
                    flatten(tf.tile(tf.expand_dims(
                        weights[idx_camera, 0], axis=1),
                        [1, 3])), axis=1), [1, H36M_NUM_JOINTS * 2])
                projection_cam = tf.multiply(tf.multiply(cam_rot, cam_weights), self.multi_camera_mask) * scale_l
                projection_cams = projection_cams.write(idx_camera, projection_cam)

                return [tf.add(idx_camera, 1), idx_rot, projection_cams]

            # run execution on all frames
            _, _, projection_e_cams = tf.while_loop(condition_cameras, exec_cameras,
                                                    [idx_cam_init, idx_rotation, projection_e_init])

            main_min_project = tf.concat([tf.zeros((n_basis, H36M_NUM_JOINTS * 2 * NUM_CAMERAS), dtype=tf.float64),
                                          tf.reshape(tf.transpose(projection_e_cams.stack(), [1, 0, 2]),
                                                     [H36M_NUM_JOINTS * 3, H36M_NUM_JOINTS * 2 * NUM_CAMERAS])], axis=0)
            reg_params = tf.concat([tf.transpose(d),
                                    tf.zeros((H36M_NUM_JOINTS * 3, n_basis), dtype=tf.float64)], axis=0)

            projected_e = tf.concat([main_min_3d, main_min_project, reg_params], axis=1)

            wa = tf.matmul(jw_diagonal, tf.transpose(projected_e))
            wb = tf.matmul(tf.tile(tf.expand_dims(jw_diagonal, axis=0), [n_frames, 1, 1]), res)

            a_init = tf.TensorArray(dtype=tf.float64, size=n_frames)
            s_init = tf.TensorArray(dtype=tf.float64, size=n_frames)

            def exec_batch(idx_batch, a_base, s_base):
                # model_a = tf.matrix_solve_ls(wa, wb[idx_batch], l2_regularizer=0.0, fast=True, name=None)
                model_a, model_s = self._ir_solver(wa, wb[idx_batch], p1, p2)
                a_base = a_base.write(idx_batch, model_a)
                s_base = s_base.write(idx_batch, model_s)
                # s_base = s_base.write(idx_batch, model_a)
                return [tf.add(idx_batch, 1), a_base, s_base]

            _, t_a, t_s = tf.while_loop(condition_batch, exec_batch, [idx_batch_init, a_init, s_init])

            all_a = tf.reshape(t_a.stack(), [n_frames, 80, 1])
            score = tf.reshape(t_s.stack(), [n_frames])
            # reconstructed = tf.matmul(tf.tile(tf.expand_dims(wa, axis=0), [n_frames, 1, 1]), all_a)
            # score = tf.reduce_sum(tf.pow(tf.subtract(reconstructed, wb), 2), axis=1)

            # run execution on all frames

            best_rot = tf.gather(check, idx_rotation)
            rotation = tf.stack([tf.sin(best_rot), tf.cos(best_rot)], 0)
            scale = all_a[:, 0]
            bases_e = tf.reshape(tf.divide(all_a[:, 1:n_basis], tf.expand_dims(scale, axis=-1)), [n_frames, -1])
            rec_poses = tf.reshape(all_a[:, n_basis:], (n_frames, H36M_NUM_JOINTS, 3))
            selected_r = tf.tile(tf.expand_dims(rotation, 1), [1, n_frames])
            rec = self._build_and_rot_model(bases_e, e[1:], mu, selected_r)
            rec = rec * -tf.abs(tf.expand_dims(scale, -1))
            rec *= -1

            rot = upgrade_multiple_r(selected_r)
            estimated_q_pose = tf.transpose(rec_poses, [0, 2, 1])
            rotated_q_pose = tf.matmul(rot, estimated_q_pose) * -tf.abs(tf.expand_dims(scale, -1))

            poses_q, _ = self.re_normalise_poses(rotated_q_pose)
            poses_p, _ = self.re_normalise_poses(rec)

            pose_base = pose_base.write(idx_rotation, poses_q)
            score_base = score_base.write(idx_rotation, score)

            return [tf.add(idx_rotation, 1), pose_base, score_base]

        # run on a single P.P.C.A. model
        _, t_pose, t_score = tf.while_loop(condition_rotations, exec_rotations, [idx_rot_init, pose_init, score_init])

        poses = tf.reshape(t_pose.stack(), [n_rotations, n_frames, -1])
        scores = tf.reshape(t_score.stack(), [n_rotations, n_frames])

        best_score = tf.reduce_min(scores, axis=0)

        def pose_avg(rotation_poses, rotation_scores):
            exp = (rotation_scores-tf.reduce_min(rotation_scores, 0))
            marginalisation_weights = tf.exp(-1*exp * tf.abs(exp_weight) * 5000)
            denominator = tf.reduce_sum(marginalisation_weights, 0)
            pose_average = tf.multiply(rotation_poses, tf.expand_dims(marginalisation_weights, -1))
            pose_average = tf.reduce_sum(pose_average, 0) / tf.expand_dims(denominator, -1)
            return pose_average

        pose = pose_avg(poses, scores)

        return pose, best_score

    def _pick_e(self, w_i, e, lambda_values, scale, interval=0.01):
        """
        Find among all the P.P.C.A. model that one that better represents this given pose
        :param w_i: 2D poses in the format (frames x 2 x n_joints)
        :param e: set of 3D pose bases in the format (n_bases x 3 x n_joints)
        :param lambda_values: variance coefficients used in the P.P.C.A.
        :param interval: sampling factor for identifying the rotation
        :return: best P.P.C.A. model index with relative base coefficients a and rotation coefficients r
        """

        w = tf.reshape(w_i, [-1, NUM_CAMERAS, 2, 17])
        n_frames = w.get_shape().as_list()[0]

        # define variables
        check_rotations = tf.range(0, 1, delta=interval, dtype=tf.float64) * 2 * pi
        # linearised_weights = tf.reshape(self.weights, [n_frames, -1])

        # init returned variables
        a_init = tf.TensorArray(dtype=tf.float64, size=self.n_models)
        score_init = tf.TensorArray(dtype=tf.float64, size=self.n_models)
        model_mu = tf.multiply(self.mod_mu, self.mu)

        def condition_models(i, *_):
            return i < self.n_models

        def exec_models(model, a_models, score_models):
            curr_pose, curr_score = self._estimate_a_and_rotation(w, e[model],
                                                                  lambda_values[model], check_rotations,
                                                                  self.weights, self.weights_c[model],
                                                                  self.exp_weights[model], scale[model],
                                                                  model_mu[model], self.hl_params[model],
                                                                  self.ir_reg[model])
            a_models = a_models.write(model, flatten(curr_pose))
            score_models = score_models.write(model, flatten(curr_score))
            return [tf.add(model, 1), a_models, score_models]

        _, t_a, t_s = tf.while_loop(condition_models, exec_models, [0, a_init, score_init], parallel_iterations=1)

        a = tf.reshape(t_a.stack(), [self.n_models, n_frames, -1])
        # check a is n_models frames 51
        score = tf.reshape(t_s.stack(), [self.n_models, n_frames]) / 2

        return score, a

    def _create_weights(self, h36m_poses_2d, visibility_joints=None):
        """
        :param h36m_poses_2d: tensor containing a set of 2D poses in the h36m format (num_poses x h36m_num_joints x 2)
        :param visibility_joints: bool tensor with the visibility of the joints in each pose (num_poses, 2, num_joints)
        :return:
        """
        _n_poses = h36m_poses_2d.get_shape().as_list()[0]

        if visibility_joints is None:
            visibility_joints = tf.ones((_n_poses, 2, self.number_joints), dtype=tf.bool)

        self.weights = tf.cast(visibility_joints, dtype=tf.float64)

    def centre_all(self, data, indices=None):
        """
        Center all data
        :param data: 2D or 3D poses in format (n_poses, n_D, n_joints)
        :param indices: indices of the joints to consider
        :return: centered poses
        """
        if indices is None:
            indices = tf.ones(self.number_joints, dtype=tf.bool)

        if tf.equal(tf.rank(data), 2) is True:
            mean_val = tf.reduce_mean(tf.boolean_mask(
                tf.transpose(data, [1, 0]), indices), axis=0)
            translated_data = tf.transpose(
                tf.subtract(tf.transpose(data, [1, 0]), mean_val), [1, 0])
            return translated_data, mean_val

        mean_val = tf.reduce_mean(
            tf.boolean_mask(tf.transpose(data, [2, 0, 1]), indices), axis=0)
        translated_data = tf.transpose(
            tf.subtract(tf.transpose(data, [2, 0, 1]), mean_val), [1, 2, 0])
        return translated_data, mean_val

    def _normalise_data(self, pose_2d):
        """
        Normalise data according to height
        :param pose_2d: matrix with poses in 2D
        :return: normalised 2D poses
                 scale factor used for normalisation
                 mean value used for normalisation
        """

        _n_poses = pose_2d.get_shape().as_list()[0]
        idx_consider = tf.cast(self.weights[0, 0], dtype=tf.bool)

        d2 = tf.transpose(pose_2d, [0, 2, 1])
        d2, mean_values = self.centre_all(d2, idx_consider)

        # Height normalisation (2 meters)
        visible_ys = tf.boolean_mask(tf.transpose(d2, [2, 1, 0]),
                                     tf.concat([tf.zeros((self.number_joints, 1), dtype=tf.bool),
                                                tf.expand_dims(idx_consider, axis=1)], axis=1))
        m2 = tf.reduce_min(visible_ys, axis=0) / 2.0
        m2 -= tf.reduce_max(visible_ys, axis=0) / 2.0
        m2.set_shape(_n_poses)

        m2_dims = tf.expand_dims(tf.expand_dims(m2, axis=1), axis=2)
        d2 = tf.divide(d2, m2_dims)
        return d2, m2, mean_values

    def normalise_gt(self, poses):
        """
        Normalise ground truth data to find reconstruction error with reconstructed pose.
        It consists into centering the joints according to the mean value.
        :param poses: 3D poses in the format (n_poses x H36M_NUM_JOINTS x 3)
        :return: normalised pose
        """
        d3 = tf.transpose(poses, [0, 2, 1])
        d3, _ = self.centre_all(d3)

        _, m3 = self.re_normalise_poses(d3)
        return d3, m3

    @staticmethod
    def _build_model(bases_coefficients, bases, mean_pose):
        """
        Build model and rotate according to the identified rotation matrix
        :param bases_coefficients: bases coefficients (n_poses x n_bases)
        :param bases: (n_poses x n_bases x 3 x H36M_NUM_JOINTS)
        :param mean_pose: (n_bases x 3 x H36M_NUM_JOINTS)
        :return: built 3D pose
        """
        weighed_sum = tf.multiply(tf.expand_dims(
            tf.expand_dims(bases_coefficients, axis=2), axis=3), bases)
        final_pose = tf.reduce_sum(weighed_sum, axis=1) + mean_pose
        return final_pose

    def _build_and_rot_model(self, bases_coefficients, bases, mean_pose, rotation):
        """
        Build model and rotate according to the identified rotation matrix
        :param bases_coefficients: bases coefficients (n_poses x n_bases)
        :param bases: (n_poses x n_bases x 3 x H36M_NUM_JOINTS)
        :param mean_pose: (n_poses x 3 x H36M_NUM_JOINTS)
        :param rotation: (2 x n_poses)
        :return: built and rotated 3D pose
        """
        rot_matrix = upgrade_multiple_r(rotation)
        ground_pose = self._build_model(bases_coefficients, bases, mean_pose)
        mod = tf.matmul(rot_matrix, ground_pose)
        return mod

    def better_rec(self, poses_2d, model, weights):
        """
        Considers both observations and model predictions to find final x,z coordinates of the 3D poses.
        :param poses_2d: 2d observations coming from the convolution layers
        :param model: reconstructed 3D poses
        :param weights: define the relevance of the observations over the model predictions
        :return: poses considering also observations
        """
        _n_poses = poses_2d.get_shape().as_list()[0]
        ext_cam_rot = self.cam
        projection = tf.matmul(ext_cam_rot, model)

        weights = tf.multiply(tf.tile(tf.expand_dims(1.55 * self.warping_weights, 0), [_n_poses, 1, 1]), weights)
        new_xz = tf.divide(projection[:, :2] + tf.multiply(poses_2d, weights),
                           weights + tf.ones_like(weights))
        new_3d_poses = tf.concat([new_xz,
                                  tf.expand_dims(projection[:, -1], axis=1)], axis=1)
        out = tf.matmul(tf.transpose(ext_cam_rot, [0, 2, 1]), new_3d_poses)
        return out

    def re_normalise_poses(self, poses_3d):
        """
        Normalise poses in order to have mean joint length one
        :param poses_3d: 3D poses in the format (n_poses x 3 x H36M_NUM_JOINTS)
        :return: normalised poses
                 scale used for the normalisation of each of the poses
        """

        t_poses = tf.transpose(poses_3d, [2, 0, 1])
        limb_subtract = tf.gather(
            t_poses, self._POSE_TREE[0]) - tf.gather(t_poses, self._POSE_TREE[1])
        scale = tf.sqrt(tf.reduce_sum(
            tf.reduce_sum(tf.pow(limb_subtract, 2), axis=2), axis=0))
        norm_poses_3d = tf.divide(poses_3d,
                                  tf.expand_dims(tf.expand_dims(scale, axis=1), axis=2))
        return norm_poses_3d, scale

    def create_rec(self, w2):
        """
        Reconstruct 3D poses given 2D poses
        :param w2: normalised 2D poses in the format (n_poses x 2 x NUM_H36M_JOINTS)
        :return: 3D poses
        """

        score, pose = self._pick_e(w2, self.bases, self.lambda_values, self.loss_t_c, interval=0.01)

        def pose_avg(poses, scores):
            exp = (scores-tf.reduce_min(scores, 0))
            weights = tf.exp(-1*tf.abs(exp * self.exp_weights_model * 5000 + self.exp_gamma_model))
            denominator = tf.reduce_sum(weights, 0)
            pose_average = tf.multiply(poses, tf.expand_dims(weights, -1))
            pose_average = tf.reduce_sum(pose_average, 0) / tf.expand_dims(denominator, -1)
            return pose_average

        pose = pose_avg(pose, score)

        rec = tf.reshape(pose, [-1, 3, self.number_joints])
        # rec = self.better_rec(w2, selected_pose, self.weights * 1.55) * -1
        rec, _ = self.re_normalise_poses(rec)
        rec *= self.magic_number
        return rec, score

    def compute_3d(self, poses_2d, cam=None, visibility_joints=None):
        """
        Reconstruct 3D poses given 2D estimations
        :param poses_2d: matrix containing 2D poses in the format (num_poses x num_joints x 2)
        :param cam: give camera matrix at test time if needed in the format (3 x 3) or (n x 3 x 3)
        :param visibility_joints: matrix with the visibility of the joints in each pose (num_poses x num_joints)
        :return: 3D poses in format (n_batch, 3, n_joints)
        """
        new_poses = poses_2d
        self._create_weights(new_poses, visibility_joints)

        norm_pose, _, _ = self._normalise_data(new_poses)

        reconstructed, score = self.create_rec(norm_pose)

        return reconstructed, norm_pose, cam, score
