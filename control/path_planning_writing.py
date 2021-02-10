"""
Running operational space control with a PyGame display, and using the pydmps
library to specify a trajectory for the end-effector to follow, in
this case, a circle.

To install the pydmps library, clone https://github.com/studywolf/pydmps
and run 'python setup.py develop'
"""
import numpy as np
import sys
import pydmps

import abr_jaco2
from abr_control.controllers import OSC, Damping
from abr_control.utils import transformations as transform
from gauss_path_planner import GaussianPathPlanner
import control_utils

import matplotlib.pyplot as plt
#TODO
"""
1) scale dmp path to desired writing size
2) arc path up between characters to avoid writing letter
3) add checks to make sure we're not writing further than we can reach
4) generalize writing to any plane
5) add offset for pen distance
"""

# load our alphanumerical path
if len(sys.argv) > 1:
    text = sys.argv[1]
else:
    text = '1'

axes = 'rxyz'                                   # for conversion between quat and euler
local_start_heading = np.array([0, 0, 1])       # the direction in EE local coordinates that the pen tip is facing
global_target_heading = np.array([0, 0, -1])    # the target direction in world coordinates to be writing
target_pos = np.array([0.7, 0.5, 0.6])          # where to start writing on the plane defined by global_target_heading
letter_spacing = 1                            # in meters
dmp_steps = 100

def load_paths(text, save_loc=None, plot=False):
    print('Loading alphanumerical path')
    if save_loc is None:
        save_loc = 'handwriting_trajectories'

    text_paths = {}
    for char in text:
        if char not in text_paths.keys():
            char_save_loc = '%s/%s.npz' % (save_loc, char)
            char_path = np.load(char_save_loc)['arr_0'].T
            text_paths[char] = char_path

            if plot:
                plt.figure()
                plt.plot(char_path[0], char_path[1], label='char')
                plt.legend()
                plt.show()

    return text_paths

text_paths = load_paths(text, plot=False)

# instantiate robot config and comm interface
print('Connecting to arm and interface')
robot_config = abr_jaco2.Config()
interface = abr_jaco2.Interface(robot_config)
interface.connect()
interface.init_position_mode()
interface.send_target_angles(robot_config.START_ANGLES)

# get our joint angles at start position
print('Getting start state information')
q_start = interface.get_feedback()['q']
T_EE = robot_config.T('EE', q_start)
#NOTE disconnecting during testing, but eventuall will move to the path
interface.disconnect()

# where we want to approach for writing
print('Getting target quaternion to align headings')
target_quat = control_utils.get_target_orientation_from_heading(
        local_start_heading,
        global_target_heading,
        T_EE,
        plot=False)

target_euler = transform.euler_from_quaternion(target_quat, axes)

# get our start EE position and quat orientation
start_pos = robot_config.Tx('EE', q_start)
start_quat = robot_config.quaternion('EE', q_start)
start_euler = transform.euler_from_quaternion(start_quat, axes)

# get our path to the writing start position
print('Generating path to board')
gauss_path_planner = GaussianPathPlanner(
            max_a=5,
            max_v=5,
            dt=0.001,
            axes=axes
    )

# generate the path to the board
gauss_path_planner.generate_path(
        state=np.array([
            start_pos[0], start_pos[1], start_pos[2],
            0, 0, 0,
            start_euler[0], start_euler[1], start_euler[2],
            0, 0, 0
        ]),
        target=np.array([
            target_pos[0], target_pos[1], target_pos[2],
            0, 0, 0,
            target_euler[0], target_euler[1], target_euler[2],
            0, 0, 0
        ]),
        start_v=0,
        target_v=0,
        plot=False
    )

pos_path = gauss_path_planner.position_path
ori_path = gauss_path_planner.orientation_path
writing_origin = np.copy(pos_path[-1])

dmp2 = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10)
dmp3 = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500, ay=np.ones(3) * 10)

# use dmp to imitate our gauss planner
dmp3.imitate_path(pos_path.T, plot=False)
pos_path = dmp3.rollout(dmp_steps)[0]
dmp3.imitate_path(ori_path.T, plot=False)
ori_path = dmp3.rollout(dmp_steps)[0]

for ii, char in enumerate(text):
    # generate the writing position path
    dmp2.imitate_path(text_paths[char], plot=False)
    dmp_pos = dmp2.rollout(dmp_steps)[0]

    # add our last point on the way to the board since the dmp begins at the origin
    dmp_pos = np.asarray(dmp_pos)
    dmp_pos = np.hstack((dmp_pos, np.ones((dmp_steps, 1))*writing_origin[2]))
    dmp_pos[:, 0] += writing_origin[0]
    dmp_pos[:, 1] += writing_origin[1]
    dmp_ori = np.ones((dmp_steps, 3))*ori_path[-1]

    # generate the path to the start of our letter
    gauss_path_planner.generate_path(
            state=np.array([
                pos_path[-1][0], pos_path[-1][1], pos_path[-1][2],
                0, 0, 0,
                ori_path[-1][0], ori_path[-1][1], ori_path[-1][2],
                0, 0, 0
            ]),
            target=np.array([
                dmp_pos[0][0], dmp_pos[0][1], dmp_pos[0][2],
                0, 0, 0,
                dmp_ori[0][0], dmp_ori[0][1], dmp_ori[0][2],
                0, 0, 0
            ]),
            start_v=0,
            target_v=0,
            plot=False
        )

    # use dmp to imitate our gauss planner
    dmp3.imitate_path(gauss_path_planner.position_path.T, plot=False)
    pos_path_to_next_letter = dmp3.rollout(dmp_steps)[0]
    dmp3.imitate_path(gauss_path_planner.orientation_path.T, plot=False)
    ori_path_to_next_letter = dmp3.rollout(dmp_steps)[0]

    pos_path = np.vstack((pos_path, pos_path_to_next_letter))
    ori_path = np.vstack((ori_path, ori_path_to_next_letter))

    pos_path = np.vstack((pos_path, dmp_pos))
    ori_path = np.vstack((ori_path, np.ones((dmp_steps, 3))*ori_path[-1]))

    # shift the writing origin over in one dimension as we write
    writing_origin[0] += letter_spacing


# get the next point in the target trajectory from the dmp
print('Plotting 6dof path')
control_utils.plot_6dof_path(
        pos_path=pos_path,
        ori_path=ori_path,
        global_start_heading=control_utils.local_to_global_heading(
            local_start_heading, T_EE),
        sampling=5,
        show_axes=False,
        axes=axes,
        scale=10
        )
