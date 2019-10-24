# -*- coding: utf-8 -*-
"""
Created on Mar 23 15:04 2017

@author: Denis Tome'
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

def hm_plot(k):
    """
    :param k: 2D matrix to be expressed as a heat map
    :return:
    """
    plt.imshow(k, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


def plot_2d_pose_16(pose_2d, visible=None):
    """
    Draw the 2D pose without the occluded/not visible joints.
    :param pose_2d: matrix of size 17x2
    :param visible: vector of size 17
    """
    _COLORS = np.array([(0, 0, 0), (255, 0, 255), (0, 0, 255),
                        (0, 255, 255), (255, 0, 0), (0, 255, 0)])/255.0
    _COLOR_MATCH = np.array([0, 1, 1, 1, 2, 2, 2, 0, 0, 3, 3, 4, 4, 4, 5, 5, 5])
    _LIMBS = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])

    if not visible:
        visible = np.ones_like(pose_2d, dtype=np.bool)

    plt.axis('equal')
    for lid, (p0, p1) in enumerate(_LIMBS):
        if not (visible[p0].all() and visible[p1].all()):
            continue
        x0, y0 = pose_2d[p0]
        x1, y1 = pose_2d[p1]
        plt.plot([x0, x1], [y0, y1], '-o', color=_COLORS[_COLOR_MATCH[p0]])
    plt.show()


def plot_2d_pose(pose_2d, visible=None):
    """
    Draw the 2D pose without the occluded/not visible joints.
    :param pose_2d: matrix of size 17x2
    :param visible: vector of size 17
    """
    _COLORS = np.array([(0, 0, 0), (255, 0, 255), (0, 0, 255),
                        (0, 255, 255), (255, 0, 0), (0, 255, 0)])/255.0
    _COLOR_MATCH = np.array([0, 1, 1, 1, 2, 2, 2, 0, 0, 3, 3, 4, 4, 4, 5, 5, 5])
    _LIMBS = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                       [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])

    if not visible:
        visible = np.ones_like(pose_2d, dtype=np.bool)
        visible[9] = 0

    plt.axis('equal')
    for lid, (p0, p1) in enumerate(_LIMBS):
        if not (visible[p0].all() and visible[p1].all()):
            continue
        x0, y0 = pose_2d[p0]
        x1, y1 = pose_2d[p1]
        plt.plot([x0, x1], [y0, y1], '-o', color=_COLORS[_COLOR_MATCH[p0]])

    plt.show()


def _plot_pose(pose, ax, name=None):
    """
    Plot the 3D pose showing the joint connections.
    :param pose: matrix of size 3x17
    """
    _CONNECTION = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                   [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    def joint_color(jc):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if jc in range(1, 4):
            _c = 1
        if jc in range(4, 7):
            _c = 2
        if jc in range(9, 11):
            _c = 3
        if jc in range(11, 14):
            _c = 4
        if jc in range(14, 17):
            _c = 5
        return colors[_c]

    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)

    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color(c[0])
        ax.plot([pose[0, c[0]], pose[0, c[1]]],
                [pose[1, c[0]], pose[1, c[1]]],
                [pose[2, c[0]], pose[2, c[1]]], c=col)
    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % joint_color(j)
        ax.scatter(pose[0, j], pose[1, j], pose[2, j], c=col, marker='o', edgecolor=col)
    scale = 1.15 * np.max(np.absolute(pose))
    smallest = -scale
    largest = scale
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)

    ax.set_title(name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_aspect('equal')


def _plot_pose_overlapping(poses, ax, name=None):
    """
    Plot the 3D pose showing the joint connections.
    :param poses: matrix of size 3x17
    """

    _CONNECTION = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                   [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    def joint_color(jc):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if jc in range(1, 4):
            _c = 1
        if jc in range(4, 7):
            _c = 2
        if jc in range(9, 11):
            _c = 3
        if jc in range(11, 14):
            _c = 4
        if jc in range(14, 17):
            _c = 5
        return colors[_c]
    scale = 0
    for pose in poses:
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(c[0])
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col)
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % joint_color(j)
            ax.scatter(pose[0, j], pose[1, j], pose[2, j], c=col, marker='o', edgecolor=col)
        scale = np.max(1.15 * np.max(np.absolute(pose)), scale)
    smallest = -scale
    largest = scale
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)

    ax.set_title(name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_aspect('equal')


def _plot_2d_pose(pose_2d, ax, name, visible=None):
    """
    Draw the 2D pose without the occluded/not visible joints.
    :param pose_2d: matrix of size 17x2
    :param visible: vector of size 17
    """
    _COLORS = np.array([(0, 0, 0), (255, 0, 255), (0, 0, 255),
                        (0, 255, 255), (255, 0, 0), (0, 255, 0)])/255.0
    _COLOR_MATCH = np.array([0, 1, 1, 1, 2, 2, 2, 0, 0, 3, 3, 4, 4, 4, 5, 5, 5])
    _LIMBS = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])

    if not visible:
        visible = np.ones_like(pose_2d, dtype=np.bool)

    plt.axis('equal')
    for lid, (p0, p1) in enumerate(_LIMBS):
        if not (visible[p0].all() and visible[p1].all()):
            continue
        x0, y0 = pose_2d[p0]
        x1, y1 = pose_2d[p1]
        ax.plot([x0, x1], [y0, y1], '-o', color=_COLORS[_COLOR_MATCH[p0]])

    ax.set_title(name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.set_aspect('equal')


def plot_2d_pose_overlapped(pose_2d, visible=None):
    """
    Draw the 2D pose without the occluded/not visible joints.
    :param pose_2d: matrix of size 17x2
    :param visible: vector of size 17
    """
    _COLORS = np.array([(0, 0, 0), (255, 0, 255), (0, 0, 255),
                        (0, 255, 255), (255, 0, 0), (0, 255, 0)])/255.0
    _COLOR_MATCH = np.array([0, 1, 1, 1, 2, 2, 2, 0, 0, 3, 3, 4, 4, 4, 5, 5, 5])
    _LIMBS = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                       [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])

    if not visible:
        visible = np.ones_like(pose_2d, dtype=np.bool)

    plt.axis('equal')
    for lid, (p0, p1) in enumerate(_LIMBS):
        if not (visible[p0].all() and visible[p1].all()):
            continue
        x0, y0 = pose_2d[p0]
        x1, y1 = pose_2d[p1]
        plt.plot([x0, x1], [y0, y1], '-o', color=_COLORS[_COLOR_MATCH[p0]])

    plt.show()


def plot_pose(pose, name=None):
    """
    Plot the 3D pose showing the joint connections.
    :param pose: matrix of size 3x17
    :param name: path and name to save the photo
    """

    _CONNECTION = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                   [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    def joint_color(jc):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if jc in range(1, 4):
            _c = 1
        if jc in range(4, 7):
            _c = 2
        if jc in range(9, 11):
            _c = 3
        if jc in range(11, 14):
            _c = 4
        if jc in range(14, 17):
            _c = 5
        return colors[_c]

    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    fig = plt.figure()
    ax = Axes3D(fig)
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color(c[0])
        ax.plot([pose[0, c[0]], pose[0, c[1]]],
                [pose[1, c[0]], pose[1, c[1]]],
                [pose[2, c[0]], pose[2, c[1]]], c=col)
    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % joint_color(j)
        ax.scatter(pose[0, j], pose[1, j], pose[2, j], c=col, marker='o', edgecolor=col)
    scale = 1.15 * np.max(np.absolute(pose))
    smallest = -scale
    largest = scale
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)
    for angle in range(45, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)

    if name is None:
        plt.show()
    else:
        plt.savefig(name+'.png')   # save the figure to file
    plt.close()


def plot_poses_overlapping(poses, title=None):
    """
    Plot the 3D pose showing the joint connections.
    :param poses: matrix of size n_poses x 3 x 17
    :param title: path and name to save the photo
    """

    _CONNECTION = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                   [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    def joint_color(jc):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if jc in range(1, 4):
            _c = 1
        if jc in range(4, 7):
            _c = 2
        if jc in range(9, 11):
            _c = 3
        if jc in range(11, 14):
            _c = 4
        if jc in range(14, 17):
            _c = 5
        return colors[_c]

    assert (poses.ndim == 3)
    assert (poses.shape[1] == 3)
    fig = plt.figure()
    ax = Axes3D(fig)
    smallest = np.inf
    largest = - np.inf
    for pose in poses:
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(c[0])
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col)
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % joint_color(j)
            ax.scatter(pose[0, j], pose[1, j], pose[2, j], c=col, marker='o', edgecolor=col)
        smallest = np.min([pose.min(), smallest])
        largest = np.max([pose.max(), largest])
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)

    if title is not None:
        plt.title(title)

    for angle in range(50, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)


def plot_poses_overlapping_16(poses, title=None):
    """
    Plot the 3D pose showing the joint connections.
    :param poses: matrix of size n_poses x 3 x 17
    :param title: path and name to save the photo
    """

    _CONNECTION = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
                   [8, 10], [10, 11], [11, 12], [8, 13], [13, 14], [14, 15]]

    def joint_color(jc):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if jc in range(1, 4):
            _c = 1
        if jc in range(4, 7):
            _c = 2
        if jc in range(9, 11):
            _c = 3
        if jc in range(11, 14):
            _c = 4
        if jc in range(14, 17):
            _c = 5
        return colors[_c]

    assert (poses.ndim == 3)
    assert (poses.shape[1] == 3)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    smallest = np.inf
    largest = - np.inf
    for pose in poses:
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(c[0])
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col)
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % joint_color(j)
            ax.scatter(pose[0, j], pose[1, j], pose[2, j], c=col, marker='o', edgecolor=col)
        smallest = np.min([pose.min(), smallest])
        largest = np.max([pose.max(), largest])
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)

    if title is not None:
        plt.title(title)
    ax.set_aspect('equal')
    plt.show()


def _weight_plotter(w, name):

    import matplotlib

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 15}

    matplotlib.rc('font', **font)

    pose_2d = np.array([[1.472, -0.05647059], [1.6, -0.05647059], [1.664, -0.63247059], [1.664, -1.27247059],
                        [1.344, -0.05647059], [1.344, -0.69647059], [1.408, -1.33647059], [1.536, 0.26352941],
                        [1.472, 0.64752941], [1.408, 0.77552941], [1.408, 0.90352941], [1.28, 0.58352941],
                        [1.152, 0.13552941], [1.216, 0.00752941], [1.664, 0.58352941], [1.856, 0.19952941],
                        [1.836, -0.10752941], [1, -1.2], [1, 1.2]])

    pose_2d_2 = pose_2d + np.array([1, 0])

    pose_2d = np.concatenate([pose_2d, pose_2d_2], axis=0) + np.array([0, 0.2])

    x = np.transpose(pose_2d)[0]
    y = np.transpose(pose_2d)[1]

    c_x = w

    _LIMBS = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                       [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])
    _LIMBS_2 = _LIMBS + np.array([19, 19])
    _LIMBS = np.concatenate([_LIMBS, _LIMBS_2], axis=0)
    plt.figure(facecolor="white")
    for lid, (p0, p1) in enumerate(_LIMBS):
        x0, y0 = pose_2d[p0]
        x1, y1 = pose_2d[p1]
        plt.plot([x0, x1], [y0, y1], '-', color='black', zorder=1)

    plt.scatter(x, y, c=c_x, s=180, zorder=2, cmap='Blues')
    plt.colorbar()
    plt.axis('equal')

    plt.xticks([])
    plt.yticks([])
    plt.title(name)
    plt.show()


def plot_weights(weights):
    """
    Plot the 3D pose showing the joint connections.
    :param weights: weights for each joint, arranged as [n_joints, dims]
    """

    n_mod = len(weights)
    w_min = np.min(weights)
    w_max = np.max(weights)
    w_n = np.reshape(weights, [-1])
    w_n = 2*(w_n-w_min)/(w_max-w_min)
    w_n = np.reshape(w_n, [-1, 2, 17])
    buf2 = 2 * np.ones([n_mod, 2, 1])
    buf0 = np.zeros([n_mod, 2, 1])
    w_n = np.reshape(np.concatenate([w_n, buf2, buf0], axis=-1), [n_mod, -1]).tolist()

    for i, w in enumerate(w_n):
        _weight_plotter(w, 'Model ' + str(i))


def ordered_errors(paths_list):
    """
    Function to plot errors of several experiments, ordered by size of the error in the first experiment
    :param paths_list: list of names of the experiments to be compared;
    """

    sort_order = np.argsort(np.load(paths_list[0] + 'errors.npy'))
    for name in paths_list:
        errors = np.load(name + 'errors.npy')
        plt.plot(errors[sort_order])
        plt.ylim([0, 500])
    plt.show()


def _weight_plot(w, ax, name):

    from matplotlib import cm
    pose_2d = ([[1.472, -0.05647059], [1.6, -0.05647059], [1.664, -0.63247059], [1.664, -1.27247059],
                [1.344, -0.05647059], [1.344, -0.69647059], [1.408, -1.33647059], [1.536, 0.26352941],
                [1.472, 0.64752941], [1.408, 0.77552941], [1.408, 0.90352941], [1.28, 0.58352941],
                [1.152, 0.13552941], [1.216, 0.00752941], [1.664, 0.58352941], [1.856, 0.19952941],
                [1.536, 0.00752941]])

    min_w = min(w)-0.00001
    max_w = max(w)+0.00001
    diff = (max_w - min_w)
    w_r = np.floor(((w - min_w)/diff)/0.05)
    w_r_int = w_r.reshape([17]).astype(int)
    start = 0.0
    stop = 1.0
    number_of_lines = 20
    cm_subsection = np.linspace(start, stop, number_of_lines)
    colors = [cm.jet(x) for x in cm_subsection]

    _LIMBS = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
                       [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])

    for lid, (p0, p1) in enumerate(_LIMBS):
        x0, y0 = pose_2d[p0]
        x1, y1 = pose_2d[p1]
        ax.plot([x0, x1], [y0, y1], '-', color='black')
    area = 15**2
    for i in range(len(pose_2d)):
        joint = pose_2d[i]
        ax.scatter(joint[0], joint[1], s=area, color=colors[w_r_int[i]], alpha=1)

    ax.set_title(name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.set_aspect('equal')


def plot_e(e, k):
    import matplotlib.gridspec as grids

    e_frame = e.shape[0]

    gs1 = grids.GridSpec(5, 5)
    gs1.update(wspace=-0.00, hspace=0.15)
    plt.axis('off')
    pli = 0
    for i in range(e_frame):
        ax1 = plt.subplot(gs1[pli], projection='3d')
        _plot_pose(e[i], ax1, name='Basis vectors (model ' + str(k) + ', index ' + str(i) + ')')
        pli = pli+1
    plt.show()


def plot_mu(mu):
    import matplotlib.gridspec as grids

    mu_frame = mu.shape[0]

    gs1 = grids.GridSpec(2, 3)
    gs1.update(wspace=-0.00, hspace=0.15)
    plt.axis('off')
    pli = 0
    for i in range(mu_frame):
        ax1 = plt.subplot(gs1[pli], projection='3d')
        _plot_pose(mu[i], ax1, name='Mean Pose (model ' + str(i) + ')')
        pli = pli+1
    plt.show()


def sample_plotter(prediction, rec, aligned, gt, e1, e2):
    import matplotlib.gridspec as grids

    n_frames = 2

    predictions_2d = prediction[:n_frames]
    predictions_3d = rec[:n_frames]
    predictions_3d_aligned = aligned[:n_frames]
    gt_3d = gt[:n_frames]

    gs1 = grids.GridSpec(2, 3)

    gs1.update(wspace=-0.00, hspace=0.15)
    plt.axis('off')
    pli = 0

    for i in range(n_frames):

        set1 = np.concatenate([np.expand_dims(predictions_3d[i], 0), np.expand_dims(gt_3d[i], 0)], axis=0)
        set2 = np.concatenate([np.expand_dims(predictions_3d_aligned[i], 0), np.expand_dims(gt_3d[i], 0)], axis=0)

        ax1 = plt.subplot(gs1[pli])
        _plot_2d_pose(predictions_2d[i], ax1, name='Initial Detections - Example ' + str(i))
        ax2 = plt.subplot(gs1[pli+1], projection='3d')
        _plot_pose_overlapping(
            set1, ax2, name='Reconstructed Pose vs GT - Examples ' + str(i) + 'Error [err=%0.3fmm]' % e1[i])
        ax3 = plt.subplot(gs1[pli+2], projection='3d')
        _plot_pose_overlapping(
            set2, ax3, name='Reconstructed Aligned (Pose) vs GT- Examples ' + str(i) + 'Error [err=%0.3fmm]' % e2[i])

        pli = pli+3
    plt.show()


def explore_checkpoint():
    import h5py
    file_name = 'results/3mods_mc_t_2/check/partial_checkpoint.h5'
    f = h5py.File(file_name, 'r')
    mean_pose = f['base_vectors'].value
    # basis = f['mean_pose'].value
    joint_weight = f['joint_weights'].value
    # e_rot = f['exp_weights'].value
    # e_mod = f['exp_weights_model'].value
    # e_pm = f['exp_additive_model'].value
    # magic_number = f['scaling_coefficient'].value
    # lambda_var = f['s1'].value
    # lambda_0 = f['s2'].value
    # scale_prior = f['scale_regularizer'].value
    variance = f['sigma'].value
    # ref_mean_pose = f['mean_pose_rec'].value
    # ref_basis = f['base_rec'].value
    # cameras = f['cameras'].value
    f.close()

    plot_pose(mean_pose[0])
    plt.imshow(variance[0], cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(joint_weight, cmap='hot', interpolation='nearest')
    plt.show()


def appendix_plot():
    def detailed_plots(pose_idx):
        plot_poses_overlapping(np.concatenate([pose_0[pose_idx], gt_0[pose_idx]], axis=0), title='Error [err=%0.3fmm]'
                                                                                                 % error_0[pose_idx])
        plot_poses_overlapping(np.concatenate([pose_3[pose_idx], gt_3[pose_idx]], axis=0), title='Error [err=%0.3fmm]'
                                                                                                 % error_3[pose_idx])
        plot_poses_overlapping(np.concatenate([pose_6[pose_idx], gt_6[pose_idx]], axis=0), title='Error [err=%0.3fmm]'
                                                                                                 % error_6[pose_idx])
        plot_poses_overlapping(np.concatenate([pose_9[pose_idx], gt_9[pose_idx]], axis=0), title='Error [err=%0.3fmm]'
                                                                                                 % error_9[pose_idx])

    error_0 = np.load('/user/HS229/mt00853/Desktop/e_model0.npy', 'r')
    error_3 = np.load('/user/HS229/mt00853/Desktop/e_model3.npy', 'r')
    error_6 = np.load('/user/HS229/mt00853/Desktop/e_model6.npy', 'r')
    error_9 = np.load('/user/HS229/mt00853/Desktop/e_model10.npy', 'r')
    pose_0 = np.load('/user/HS229/mt00853/Desktop/p_model0.npy', 'r')
    pose_3 = np.load('/user/HS229/mt00853/Desktop/p_model3.npy', 'r')
    pose_6 = np.load('/user/HS229/mt00853/Desktop/p_model6.npy', 'r')
    pose_9 = np.load('/user/HS229/mt00853/Desktop/p_model10.npy', 'r')
    gt_0 = np.load('/user/HS229/mt00853/Desktop/gt_model0.npy', 'r')
    gt_3 = np.load('/user/HS229/mt00853/Desktop/gt_model3.npy', 'r')
    gt_6 = np.load('/user/HS229/mt00853/Desktop/gt_model6.npy', 'r')
    gt_9 = np.load('/user/HS229/mt00853/Desktop/gt_model10.npy', 'r')

    print np.mean(error_3), np.median(error_3)
    print np.mean(error_6), np.median(error_6)
    print np.mean(error_9), np.median(error_9)

    q1 = np.argsort(error_3)
    q2 = np.argsort(error_6)
    q3 = np.argsort(error_9)

    plt.plot(error_3[q1], zorder=15, alpha=1, color='blue', label='3 Models')
    plt.plot(error_6[q1], zorder=5, alpha=0.75, color='red', label='6 Models')
    plt.plot(error_9[q1], zorder=10, alpha=0.75, color='green', label='10 Models')
    plt.show()

    plt.plot(error_3[q2], zorder=5, alpha=0.75, color='blue', label='3 Models')
    plt.plot(error_6[q2], zorder=15, alpha=1, color='red', label='6 Models')
    plt.plot(error_9[q2], zorder=10, alpha=0.75, color='green', label='10 Models')
    plt.show()

    plt.plot(error_3[q3], zorder=10, alpha=0.75, color='blue', label='3 Models')
    plt.plot(error_6[q3], zorder=12, alpha=0.75, color='red', label='6 Models')
    plt.plot(error_9[q3], zorder=15, alpha=1, color='green', label='10 Models')
    plt.show()

    pid = q1[0]
    detailed_plots(pid)
    pid = q1[-1]
    detailed_plots(pid)
    pid = q1[15000]
    detailed_plots(pid)
    pid = np.argmax(error_3 - error_6)
    print pid
    detailed_plots(pid)
