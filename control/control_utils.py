from mpl_toolkits.mplot3d import Axes3D
from abr_control.utils import transformations as transform
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
    quat = transform.quaternion_about_axis(theta1, axis)

    return quat


def rot_vec_by_quat(vec, quat):
    """
    Applies a quaternion rotation to a vector
    """
    #NOTE look at implementing the simplified method here
    # https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    quat /= np.linalg.norm(quat)
    # vec /= np.linalg.norm(vec)
    vec_quat = np.array([0, vec[0], vec[1], vec[2]])
    new_vec = transform.quaternion_multiply(transform.quaternion_multiply(quat, vec_quat), transform.quaternion_conjugate(quat))

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


def get_target_orientation_from_heading(local_start_heading, global_target_heading, transform=None, plot=False):
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


def plot_6dof_path(
       pos_path, ori_path, global_start_heading, axes, sampling=20, show_axes=False, scale=1):
    """
    Plots the path of the EE over time, but also adds the local reference frame and heading we are
    trying to align with a global heading

    Parameters
    ----------
    scale: float
        value to scale quiver plot arrow lengths
    """
    quat_path = []
    print('Converting euler path to quaternion path')
    for step in ori_path:
        quat_path.append(transform.quaternion_from_euler(
            step[0], step[1], step[2], axes)
        )
    quat_path = np.asarray(quat_path)

    global_headings = []
    pos_path = np.asarray(pos_path).T

    for ii, quat in enumerate(quat_path):
        current_heading = rot_vec_by_quat(global_start_heading, quat)
        global_headings.append(current_heading)

    global_headings = np.asarray(global_headings).T[:, ::sampling]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(pos_path[0], pos_path[1], pos_path[2], color='k', label='path')
    ax.quiver(
            pos_path[0][::sampling],
            pos_path[1][::sampling],
            pos_path[2][::sampling],
            global_headings[0]/scale,
            global_headings[1]/scale,
            global_headings[2]/scale,
            color='tab:purple',
            label='local heading')

    if show_axes:
        axes = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
        ax.quiver(
                pos_path[0][::sampling]/scale,
                pos_path[1][::sampling]/scale,
                pos_path[2][::sampling]/scale,
                axes['x'][0],
                axes['x'][1],
                axes['x'][2],
                color='r',
                linestyle='-',
                label='x')

        ax.quiver(
                pos_path[0][::sampling]/scale,
                pos_path[1][::sampling]/scale,
                pos_path[2][::sampling]/scale,
                axes['y'][0],
                axes['y'][1],
                axes['y'][2],
                color='g',
                linestyle='-',
                label='y')

        ax.quiver(
                pos_path[0][::sampling]/scale,
                pos_path[1][::sampling]/scale,
                pos_path[2][::sampling]/scale,
                axes['z'][0],
                axes['z'][1],
                axes['z'][2],
                color='b',
                linestyle='-',
                label='z')

    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.set_zlim(0, 2)
    plt.legend()
    plt.show()


def plot_6dof_path_from_q(
        q_track, local_start_heading, robot_config, sampling=20):
    """
    Plots the path of the EE over time, but also adds the local reference frame and heading we are
    trying to align with a global heading
    """
    global_headings = []
    pos_path = []
    local_axes = {'x': [], 'y': [], 'z': []}
    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]

    for state in q_track:
        pos_path.append(robot_config.Tx('EE', state))

    for state in q_track[::sampling]:
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

    pos_path = np.asarray(pos_path).T
    local_axes['x'] = np.asarray(local_axes['x']).T
    local_axes['y'] = np.asarray(local_axes['y']).T
    local_axes['z'] = np.asarray(local_axes['z']).T
    global_headings = np.asarray(global_headings).T

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(
            pos_path[0],
            pos_path[1],
            pos_path[2],
            color='k',
            label='path')

    ax.quiver(
            pos_path[0][::sampling],
            pos_path[1][::sampling],
            pos_path[2][::sampling],
            global_headings[0],
            global_headings[1],
            global_headings[2],
            color='tab:purple',
            label='local heading')

    ax.quiver(
            pos_path[0][::sampling],
            pos_path[1][::sampling],
            pos_path[2][::sampling],
            local_axes['x'][0],
            local_axes['x'][1],
            local_axes['x'][2],
            color='r',
            linestyle='-',
            label='local x')

    ax.quiver(
            pos_path[0][::sampling],
            pos_path[1][::sampling],
            pos_path[2][::sampling],
            local_axes['y'][0],
            local_axes['y'][1],
            local_axes['y'][2],
            color='g',
            linestyle='-',
            label='local y')

    ax.quiver(
            pos_path[0][::sampling],
            pos_path[1][::sampling],
            pos_path[2][::sampling],
            local_axes['z'][0],
            local_axes['z'][1],
            local_axes['z'][2],
            color='b',
            linestyle='-',
            label='local z')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    plt.legend()
    plt.show()

# def generate_path_to_surface(start_state, target_pos, approach_heading, local_heading_to_align, approach_offset=0):
#     """
#     Generates a gaussian path to a surface, given a location on the surface to approach
#     and the approach heading. The approach offset backs the EE off from the surface target position
#     specified
#
#     Parameters
#     ----------
#     target_pos: 3D np.array
#         the cartesian position on the surface to approach
#     approach_heading: 3D np.array
#         the cartesian vector that points towards the direction we want to be facing
#         when we get to the surface
#     local_heading_to_align: 3D np.array
#         the cartesian vector in our EE local coordinates that we want to align with the
#         approach heading
#     approach_offset: float, Optional (Default: 0.0)
#         the distance [meters] to be away from the position on the surface. This can be used
#         for accounting for the distance of a pen or other pointing tool.
#
#     """

