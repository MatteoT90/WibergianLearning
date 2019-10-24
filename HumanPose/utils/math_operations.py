# -*- coding: utf-8 -*-
"""
Created on Jul 10 15:35 2017

@author: Denis Tome'
"""
import tensorflow as tf
import numpy as np


def pseudo_inverse(x, eps=1e-5):
    """
    Compute pseudo-inverse of matrix x
    :param x: input matrix
    :param eps: epsilon value added to the matrix in case determinant is zero
    :return: pseudo-inverse matrix
    """
    c = tf.constant(eps, dtype=tf.float64)
    xtx = tf.matmul(tf.transpose(x), x)
    size_in_matrix = xtx.get_shape().as_list()
    invertible = tf.cond(tf.matrix_determinant(xtx) > 0,
                         lambda: xtx, lambda: xtx + tf.eye(size_in_matrix[1], dtype=tf.float64) * c)
    pseudo_inv = tf.matmul(tf.matrix_inverse(invertible), tf.transpose(x))
    return pseudo_inv


def lstsq(x, y):
    """
    Solving least square error in Tensorflow in a closed form.
    :param x: matrix of size (m x n)
    :param y: matrix of size size (n x l)
    :return: sol the solution of the least square problem
             res the error resulting from the optimisation problem
    """
    pseudo_inv = pseudo_inverse(x)
    sol = tf.matmul(pseudo_inv, y)
    # compute residual
    reconstructed = tf.matmul(x, sol)
    tf.assert_equal(reconstructed.shape, y.shape)
    res = tf.reduce_sum(tf.pow(tf.subtract(reconstructed, y), 2))
    return sol, res


def upgrade_r(sin, cos):
    """
    Construct the rotation matrix
    :param sin: sin value
    :param cos: cos value
    :return: rotation matrix
    """
    rot_matrix = [sin, -cos, 0.0, cos, sin, 0.0, 0, 0, 1.0]
    rot_matrix = tf.reshape(rot_matrix, [3, 3])
    return rot_matrix


def upgrade_multiple_r(rotations):
    """
    Construct the rotation matrix
    :param rotations: list of rotations expressed in quaternions (2 x n_rotations)
    :return: rotation matrix (n_rotations x 3 x 3) transposed, ready for multiplication
    """
    _n_poses = rotations.get_shape().as_list()[1]
    neg_val = tf.expand_dims(tf.transpose(
        rotations * tf.expand_dims(
            tf.constant([1, -1], dtype=tf.float64), axis=1),
        [1, 0]), axis=2)
    # n_rotations x 2 x 2
    sub_matrix = tf.transpose(tf.concat(
        [tf.expand_dims(tf.transpose(rotations, [1, 0]), axis=2),
         neg_val[:, 1::-1]], axis=2), [2, 0, 1])
    pad_bottom = tf.expand_dims(tf.zeros((_n_poses, 2), dtype=tf.float64), axis=0)
    padded_bottom = tf.transpose(
        tf.concat([sub_matrix, pad_bottom], axis=0), [2, 1, 0])
    pad_right = tf.expand_dims(
        tf.tile(tf.expand_dims(tf.constant([0, 0, 1], dtype=tf.float64), axis=0),
                [_n_poses, 1]), axis=0)
    rot_matrix = tf.transpose(tf.concat([padded_bottom, pad_right], axis=0), [1, 2, 0])
    return rot_matrix


def flatten(x):
    """
    Expresses a matrix of unspecified shape as a vector;
    :param x: matrix
    :return: vectorised matrix
    """
    return tf.reshape(x, [-1])


def custom_py_layer(func, inp, tout, stateful=True, name=None, grad=None):
    """
    Allow to use a custom python layer with the associated gradient
    :param func: the function of which we want to customise the gradient. Must use only python!!!
    :param inp: the list with all inputs to func and to its gradient. Both must receive the same inputs.
    :param tout: type of the output of the layer.
    :param stateful:
    :param name: name of the new function
    :param grad: the gradient of func, using tensorflow only.
    :return: the output of func.
    """

    # Need to generate a unique name to avoid duplicates if there are more than one custom gradients:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 100000000))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, tout, stateful=stateful, name=name)[0]


def solve_squares(x, y, name=None):

    def forward(a, b):
        # We solve the weighted least square problem || a.x - b ||^2
        a_dag = np.linalg.pinv(a)
        out = np.matmul(a_dag, b)
        return out.astype(np.float64)

    def backward(op, grad):
        # This is the gradient of the pose wrt omega

        a = op.inputs[0]
        b = op.inputs[1]

        shape_a = tf.shape(a)

        a_dag = pseudo_inverse(a)
        out = tf.matmul(a_dag, b)

        ad_adt = tf.matmul(a_dag, tf.transpose(a_dag, [1, 0]))
        # adt_ad_b = tf.matmul(tf.matmul(tf.transpose(a_dag, [1, 0]), a_dag), b)
        a_ad_b = tf.matmul(tf.subtract(tf.eye(shape_a[0], dtype=tf.float64), tf.matmul(a, a_dag)), b)
        # ad_a = tf.subtract(tf.eye(shape_a[1], dtype=tf.float64), tf.matmul(a_dag, a))

        grad_b = tf.expand_dims(tf.reduce_sum(tf.multiply(a_dag, grad), 0), -1)

        dx_da_1 = tf.expand_dims(tf.reduce_sum(tf.multiply(a_dag, grad), 0), -1) * tf.transpose(out)
        dx_da_2 = tf.transpose(tf.expand_dims(tf.reduce_sum(tf.multiply(ad_adt, grad), 0), -1) * tf.transpose(a_ad_b))

        # In our base case, dx_da_3 is always zero and can therefore be ignored.
        # dx_da_3 = tf.transpose(tf.expand_dims(tf.reduce_sum(tf.multiply(ad_a, grad), 0), -1) * tf.transpose(adt_ad_b))

        grad_a = tf.subtract(dx_da_2, dx_da_1)

        return grad_a, grad_b

    with tf.name_scope(name, "FindPose", [x, y]) as name:
        return custom_py_layer(forward, [x, y], [tf.float64], name=name, grad=backward)


def _procrustes(x, y, compute_optimal_scale=True):
    """
    A Numpy port of MATLAB `procrustes` function.
    Args
      matX: array NxM of targets, with N number of points and M point dimensionality
      matY: array NxM of inputs
      compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
      d: squared error after transformation
      z: transformed Y
      t: computed rotation
      b: scaling
      c: translation
    """
    mu_x = tf.reduce_mean(x, axis=0)
    mu_y = tf.reduce_mean(y, axis=0)

    x0 = x - mu_x
    y0 = y - mu_y

    ss_x = tf.reduce_sum(tf.square(x0))
    ss_y = tf.reduce_sum(tf.square(y0))

    # centred Frobenius norm
    norm_x = tf.sqrt(ss_x)
    norm_y = tf.sqrt(ss_y)

    # scale to equal (unit) norm
    x0 = x0 / norm_x
    y0 = y0 / norm_y

    # optimum rotation matrix of Y
    a = tf.matmul(tf.transpose(x0), y0)

    s, u, v = tf.svd(a, full_matrices=False)

    t = tf.matmul(v, tf.transpose(u))

    # Make sure we have a rotation
    det_t = tf.matrix_determinant(t)
    v_s = tf.concat([v[:, :-1], tf.expand_dims(v[:, -1] * tf.sign(det_t), axis=1)], axis=1)
    s_s = tf.concat([s[:-1], tf.expand_dims(tf.sign(det_t) * s[-1], 0)], axis=0)

    t = tf.matmul(v_s, tf.transpose(u))
    trace_ta = tf.reduce_sum(s_s)

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = trace_ta * norm_x / norm_y
        d = 1 - tf.square(trace_ta)
        z = norm_x * trace_ta * tf.matmul(y0, t) + mu_x
    else:  # If no scaling allowed
        b = 1
        d = 1 + ss_y / ss_x - 2 * trace_ta * norm_y / norm_x
        z = norm_y * tf.matmul(y0, t) + mu_x
    c = tf.expand_dims(mu_x, 0) - b * tf.matmul(tf.expand_dims(mu_y, 0), t)

    return d, z, t, b, c


def compute_error(a, b):
    """
    Computes the reconstruction error for a batch of reconstructions
    Inputs and outputs are in numpy format.
    :param a: set of 3D poses in the format (size_batch, n_dim, n_joints)
    :param b: set of 3D ground truth in the format (size_batch, n_dim, n_joints)
    :return: list of reconstruction errors for the batch
             mean reconstruction error
    """

    protocol_1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(a, b)), 1)), axis=-1)

    n_poses = a.get_shape().as_list()[0]
    align_init = tf.TensorArray(dtype=tf.float64, size=n_poses)

    idx_frame_init = tf.constant(0, tf.int32)

    def condition_frames(i, *_):
        return i < n_poses

    def align_frames(idx_frame, aligned_p):
        """Run on individual frames"""
        _, z, t, b_p, c = _procrustes(tf.transpose(a[idx_frame]), tf.transpose(b[idx_frame]))
        aligned_poses = b_p * tf.matmul(tf.transpose(b[idx_frame]), t) + c
        aligned_p = aligned_p.write(idx_frame, aligned_poses)
        return [tf.add(idx_frame, 1), aligned_p]

        # run execution on all frames
    _, model_a = tf.while_loop(condition_frames, align_frames, [idx_frame_init, align_init])

    aligned_b = tf.reshape(tf.squeeze(model_a.stack()), [n_poses, 17, 3])

    new_pose = tf.transpose(aligned_b, [0, 2, 1])

    protocol_2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(a, new_pose)), 1))

    p2_17 = tf.reduce_mean(protocol_2)

    p14 = tf.concat([protocol_2[:, 1:7], tf.expand_dims(protocol_2[:, 8], -1), protocol_2[:, 10:]], axis=-1)

    p2_14 = tf.reduce_mean(p14)

    return [p2_14, p2_17, tf.reduce_mean(protocol_1)], new_pose, tf.reduce_mean(protocol_2, axis=-1), protocol_1
