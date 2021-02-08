from mpl_toolkits.mplot3d import Axes3D
from abr_control.utils import transformations as tform
import abr_jaco2
import matplotlib.pyplot as plt
import numpy as np

def get_target_quat(target_heading, start_heading):
    """
    Gets the quaternion orientation that rotates the start heading to target heading
    """
    # get the vector perpendicular to the z axis and target heading
    # this is the vector we rotate about
    start_heading = start_heading / np.linalg.norm(start_heading)
    target_heading = target_heading / np.linalg.norm(target_heading)
    axis = np.cross(start_heading, target_heading)
    # axis = [0, 1, 0]
    # the angle I want to rotate around it by
    # theta1 = np.pi/2
    theta1 = np.arccos(
            (np.dot(target_heading, start_heading)
            /(np.linalg.norm(target_heading) * np.linalg.norm(start_heading)))
        )

    # get quaternion from angle and axis
    quat = tform.quaternion_about_axis(theta1, axis)

    return quat


def rot_vec_by_quat(vec, quat):
    """
    Applies a quaternion rotation to a vector
    """
    #NOTE look at implementing the simplified method here
    # https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    print('starting: ', vec)
    print('quat: ', quat)
    quat /= np.linalg.norm(quat)
    # vec /= np.linalg.norm(vec)
    vec_quat = np.array([0, vec[0], vec[1], vec[2]])
    new_vec = tform.quaternion_multiply(tform.quaternion_multiply(quat, vec_quat), tform.quaternion_conjugate(quat))
    print('rotated: ', new_vec)

    return new_vec[1:]


def plot_heading(start_heading, rotated_heading):
    """
    Plots the world axis along with the start and rotated headings
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(0, 0, 0, 1.3, 0, 0, label='X', linestyle='-', color='r')
    ax.quiver(0, 0, 0, 0, 1.3, 0, label='Y', linestyle='-', color='g')
    ax.quiver(0, 0, 0, 0, 0, 1.3, label='Z', linestyle='-', color='b')

    # starting vector
    ax.quiver(0, 0, 0, start_heading[0], start_heading[1], start_heading[2], label='original', color='k')
    # rotated vector
    ax.quiver(0, 0, 0, rotated_heading[0], rotated_heading[1], rotated_heading[2], label='rotated', linestyle='--', color='m')
    ax.scatter(0, 0, 0, label='origin', color='r')
    ax.legend()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()

def local_to_global_heading(local_start_heading, transform=None):
    """
    Transforms a heading in local coordinates to one in world frame
    """
    # if no transform then we applying identity
    if transform is None:
        transform = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1]]
        )

    if len(local_start_heading) == 3:
        local_start_heading = np.array([
            [local_start_heading[0],
             local_start_heading[1],
             local_start_heading[2],
             1]
        ]).T

    # we want teh vector from EE origin to the start heading
    # the transform gives us the location of the vector head in world coordinates
    # so we need to subtract the EE origin (in world coordinates) which acts as
    # the vector tail
    local_heading_tail = np.array([0, 0, 0, 1]).T
    # get our local heading vector head and tail in global coordinates
    start_heading_head = np.squeeze(np.dot(transform, local_start_heading)[:3])
    start_heading_tail = np.squeeze(np.dot(transform, local_heading_tail)[:3])
    # get our start heading in world coordinates
    global_start_heading = start_heading_head - start_heading_tail
    # get the quaternion orientation that moves our start heading (now in global
    # coordinates) to the global target heading
    global_start_heading = global_start_heading / np.linalg.norm(global_start_heading)
    return global_start_heading


def get_target_orientation(local_start_heading, global_target_heading, transform=None, plot=False):
    """
    Gets the target orientation to move a start heading in local coordinates to a target
    heading in world coordinates

    Parameters
    ----------
    local_heading: 3d np.array
        unit vector in local frame defining what direction is to be
        aligned with out target
    global_target_heading: 3d np.array
        unit vector in global frame defining the target direction to
        align our local heading to
    """

    global_start_heading = local_to_global_heading(local_start_heading, transform)
    quat = get_target_quat(global_target_heading, global_start_heading)

    if plot:
        # start_heading = start_heading / np.linalg.norm(start_heading)
        # target_heading = target_heading / np.linalg.norm(target_heading)
        # my rotated vector
        rotated_heading = rot_vec_by_quat(vec=global_start_heading, quat=quat)
        # plot to check if we rotated as expected
        plot_heading(global_start_heading, rotated_heading)

    return quat


def plot_path_with_local_frame(
        q_track, local_start_heading, robot_config, sampling=20):
    """
    Plots the path of the EE over time, but also adds the local reference frame and heading we are
    trying to align with a global heading
    """
    global_headings = []
    xyz_path = []
    local_axes = {'x': [], 'y': [], 'z': []}
    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]

    for state in q_track:
        xyz_path.append(robot_config.Tx('EE', state))
        transform = robot_config.T('EE', state)
        global_headings.append(
                local_to_global_heading(local_start_heading, transform))

        # get local coordinate system for plotting
        local_axes['x'].append(
                local_to_global_heading(x, transform))
        local_axes['y'].append(
                local_to_global_heading(y, transform))
        local_axes['z'].append(
                local_to_global_heading(z, transform))

    xyz_full_path = np.asarray(xyz_path).T
    # TODO add the sampling before the extra math done above
    # NOTE we still need / want the full path for plotting
    xyz_path = xyz_full_path[:, ::sampling]
    local_axes['x'] = np.asarray(local_axes['x']).T[:, ::sampling]
    local_axes['y'] = np.asarray(local_axes['y']).T[:, ::sampling]
    local_axes['z'] = np.asarray(local_axes['z']).T[:, ::sampling]
    global_headings = np.asarray(global_headings).T[:, ::sampling]
    print('local x: ', np.asarray(local_axes['x']).shape)
    print('global: ', np.asarray(global_headings).shape)

    # dx = np.zeros(len(state))
    # dy = np.zeros(len(state))
    # dz = np.ones(len(state))
    # print(dx.shape)
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(xyz_full_path[0], xyz_full_path[1], xyz_full_path[2], color='k', label='path')
    ax.quiver(xyz_path[0], xyz_path[1], xyz_path[2], global_headings[0], global_headings[1], global_headings[2], color='tab:purple', label='local heading')
    ax.quiver(xyz_path[0], xyz_path[1], xyz_path[2], local_axes['x'][0], local_axes['x'][1], local_axes['x'][2], color='r', linestyle='-', label='local x')
    ax.quiver(xyz_path[0], xyz_path[1], xyz_path[2], local_axes['y'][0], local_axes['y'][1], local_axes['y'][2], color='g', linestyle='-', label='local y')
    ax.quiver(xyz_path[0], xyz_path[1], xyz_path[2], local_axes['z'][0], local_axes['z'][1], local_axes['z'][2], color='b', linestyle='-', label='local z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    plt.legend()
    plt.show()
