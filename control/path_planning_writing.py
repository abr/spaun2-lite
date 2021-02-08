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

# load our alphanumerical path
if len(sys.argv) > 1:
    number = sys.argv[1]
else:
    number = '1'

print('Loading alphanumerical path')
alphanum_path = 'handwriting_trajectories/%s.npz' % number
alphanum_path = np.load(alphanum_path)['arr_0'].T

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
interface.disconnect()
# get our transform to get our local heading in global coordinates
T_EE = robot_config.T('EE', q_start)

# the direction in EE local coordinates that the pen tip is facing
local_start_heading = np.array([0, 0, 1])
# local_start_heading = np.array([1, 1, 1])
# the target direction in world coordinates to be writing
# global_target_heading = np.array([1, 0, 0])
global_target_heading = np.array([0, 0, -1])

# where we want to approach for writing
print('Getting target quaternion to align headings')
axes = 'rxyz'
target_pos = np.array([0.7, 0.5, 0.6])
target_quat = control_utils.get_target_orientation(
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
path_to_board = GaussianPathPlanner(
            max_a=5,
            max_v=5,
            dt=0.001,
            axes=axes
    )

path_to_board.generate_path(
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

quat_path = []
print('Converting euler path to quaternion path')
for step in path_to_board.orientation_path:
    quat_path.append(transform.quaternion_from_euler(
        step[0], step[1], step[2], axes)
    )
quat_path = np.asarray(quat_path)

# generate the writing position path
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10)
dmp.imitate_path(alphanum_path, plot=False)
dmp_path = dmp.rollout(path_to_board.n_timesteps)[0]

# add our last point on the way to the board since the dmp begins at the origin
dmp_quat_path = []
for ii in range(0, len(dmp_path)):
    dmp_path[ii] += path_to_board.position_path[-1][0]
    dmp_quat_path.append(target_quat)

dmp_path = np.asarray(dmp_path)
dmp_quat_path = np.asarray(dmp_quat_path)
print('dmp pos path: ', dmp_path.shape)
print('dmp quat: ', dmp_quat_path.shape)
print('pos path: ', path_to_board.position_path.shape)
print('quat path shape: ', quat_path.shape)
print((np.ones((path_to_board.n_timesteps, 1))*path_to_board.position_path[-1][2]).shape)
dmp_path = np.hstack((dmp_path, np.ones((path_to_board.n_timesteps, 1))*path_to_board.position_path[-1][2]))
print('dmp pos path: ', dmp_path.shape)

pos_path = np.vstack((path_to_board.position_path, dmp_path))
quat_path = np.vstack((quat_path, np.ones((path_to_board.n_timesteps, 4))*target_quat))
print('final pos path: ', pos_path.shape)
print('final quat path: ', quat_path.shape)

# get the next point in the target trajectory from the dmp
# target_xyz[0], target_xyz[1] = dmp.step(error * 1e2)[0]
# target_xyz += np.array([0, 2, 0])
print('Plotting 6dof path')
control_utils.plot_path_from_quat(
        # pos_path=path_to_board.position_path,
        pos_path=pos_path,
        quat_path=quat_path,
        global_start_heading=control_utils.local_to_global_heading(
            local_start_heading, T_EE),
        sampling=20,
        show_axes=False
)

