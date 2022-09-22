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
from abr_analyze import DataHandler

import matplotlib.pyplot as plt
#TODO
"""
DONE 1) scale dmp path to desired writing size
2) arc path up between characters to avoid writing between letters
3) add checks to make sure we're not writing further than we can reach
DONE 4) add offset for pen distance
5) generalize writing to any plane, including buffer offset direction
"""
plot = False

dt = 0.001
# np.set_printoptions(threshold=sys.maxsize)

kp = 100
kv = 14
ko = 150
ctrlr_dof = [True, True, True, True, True, False]
save_loc = 'kp=%i|kv=%i|ko=%i|dof=%i' % (kp, kv, ko, int(np.sum(ctrlr_dof)))

# load our alphanumerical path
if len(sys.argv) > 1:
    text = sys.argv[1]
else:
    text = '1'

# for conversion between quat and euler
axes = 'rxyz'
# the direction in EE local coordinates that the pen tip is facing
local_start_heading = np.array([0, 0, 1])
# the target direction in world coordinates to be writing
global_target_heading = np.array([-1, 0, 0])
# writing instrument offset from EE in EE coodrinates
pen_buffer = [0, 0, 0]
# pen_buffer = [0, 0, 0.06]
# where to start writing on the plane defined by global_target_heading
# target_pos = np.array([-0.6, 0.0, 0.44])
target_pos = np.array([-0.62, -0.15, 0.44])
# character size [x, y] in meters
char_size = [0.05, 0.05]
# char_size = [0.025, 0.025]
# spacing between letters in meters
letter_spacing = char_size[0]/2
# how many steps for each dmp path (currently all the same)
dmp_steps = 2000
# for plotting to improve arrow visibility
sampling = 25
# distance to back away from plane between letters
backup_buffer = np.array([-0.1, 0, 0])

def load_paths(text, save_loc=None, plot=False, char_size=None):
    if char_size is None:
        char_size = [1, 1]

    print('Loading alphanumerical path')
    if save_loc is None:
        save_loc = 'handwriting_trajectories'

    text_paths = {}
    for char in text:
        if char not in text_paths.keys():
            char_save_loc = '%s/%s.npz' % (save_loc, char)
            char_path = np.load(char_save_loc)['arr_0'].T
            # scale our characters to be size 1m x 1m
            # then apply the user defined char size scaling
            char_path[0, :] *= (0.5 * char_size[0])
            char_path[1, :] *= (0.5 * char_size[1])
            text_paths[char] = char_path

            if plot:
                plt.figure()
                plt.plot(char_path[0], char_path[1], label='char')
                plt.legend()
                # plt.xlim(-0.25, 0.2)
                # plt.ylim(-0.25, 0.2)
                plt.show()

    return text_paths

text_paths = load_paths(text, plot=False, char_size=char_size)

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
start_pos = robot_config.Tx('EE', q_start, x=pen_buffer)
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
            target_pos[0]-backup_buffer[0], target_pos[1]-backup_buffer[1], target_pos[2]-backup_buffer[2],
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
writing_origin = np.copy(pos_path[-1]) + backup_buffer

dmp2 = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10, dt=dt)
dmp3 = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500, ay=np.ones(3) * 10, dt=dt)

# use dmp to imitate our gauss planner
# dmp3.imitate_path(pos_path.T, plot=False)
# pos_path = dmp3.rollout(dmp_steps)[0]
# dmp3.imitate_path(ori_path.T, plot=False)
# ori_path = dmp3.rollout(dmp_steps)[0]

for ii, char in enumerate(text):
    # generate the writing position path
    dmp2.imitate_path(text_paths[char], plot=False)
    dmp_pos = dmp2.rollout()[0]
    dmp_steps = len(dmp_pos)

    # add our last point on the way to the board since the dmp begins at the origin
    # print('dmp pos: ', dmp_pos)
    dmp_pos = np.asarray(dmp_pos)
    # plt.figure()
    # plt.plot(dmp_pos[:, 0], dmp_pos[:, 1])
    # plt.show()

    # write on xy plane
    # dmp_pos = np.hstack((dmp_pos, np.ones((dmp_steps, 1))*writing_origin[2]))
    # # spacing along line
    # max_horz_point = max(dmp_pos[:, 0])
    # dmp_pos[:, 0] += writing_origin[0] # + max_horz_point
    # # vertical alignment
    # dmp_pos[:, 1] += writing_origin[1]
    # dmp_ori = np.ones((dmp_steps, 3))*ori_path[-1]
    # # shift the writing origin over in one dimension as we write
    # writing_origin[0] += letter_spacing + max_horz_point


    # write on yz plane
    dmp_pos = np.hstack((np.ones((dmp_steps, 1))*writing_origin[0], dmp_pos))
    # spacing along line
    max_horz_point = max(dmp_pos[:, 1])
    dmp_pos[:, 1] += writing_origin[1] # + max_horz_point
    # vertical alignment
    dmp_pos[:, 2] += writing_origin[2]
    dmp_ori = np.ones((dmp_steps, 3))*ori_path[-1]
    # shift the writing origin over in one dimension as we write
    writing_origin[1] += letter_spacing + max_horz_point


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
        )

    # use dmp to imitate our gauss planner
    # dmp3.imitate_path(gauss_path_planner.position_path.T, plot=False)
    # pos_path_to_next_letter = dmp3.rollout(dmp_steps)[0]
    pos_path_to_next_letter = gauss_path_planner.position_path
    # Add offset to back up from board
    # pos_path_to_next_letter[:, 0] += 0.02
    # dmp3.imitate_path(gauss_path_planner.orientation_path.T, plot=False)
    # ori_path_to_next_letter = dmp3.rollout(dmp_steps)[0]
    ori_path_to_next_letter = gauss_path_planner.orientation_path

    pos_path = np.vstack((pos_path, pos_path_to_next_letter))
    ori_path = np.vstack((ori_path, ori_path_to_next_letter))

    pos_path = np.vstack((pos_path, dmp_pos))
    ori_path = np.vstack((ori_path, np.ones((dmp_steps, 3))*ori_path[-1]))

    # backup to move over to next letter
    gauss_path_planner.generate_path(
            state=np.array([
                pos_path[-1][0], pos_path[-1][1], pos_path[-1][2],
                0, 0, 0,
                ori_path[-1][0], ori_path[-1][1], ori_path[-1][2],
                0, 0, 0
            ]),
            target=np.array([
                pos_path[-1][0]-backup_buffer[0], pos_path[-1][1]-backup_buffer[1], pos_path[-1][2]-backup_buffer[2],
                0, 0, 0,
                ori_path[-1][0], ori_path[-1][1], ori_path[-1][2],
                0, 0, 0
            ]),
            start_v=4,
            target_v=0,
            plot=False
        )

    pos_path = np.vstack((pos_path, gauss_path_planner.position_path))
    ori_path = np.vstack((ori_path, gauss_path_planner.orientation_path))


gauss_path_planner.generate_path(
        state=np.array([
            pos_path[-1][0], pos_path[-1][1], pos_path[-1][2],
            0, 0, 0,
            ori_path[-1][0], ori_path[-1][1], ori_path[-1][2],
            0, 0, 0
        ]),
        target=np.array([
            pos_path[-1][0]+0.1, pos_path[-1][1], pos_path[-1][2],
            0, 0, 0,
            ori_path[-1][0], ori_path[-1][1], ori_path[-1][2],
            0, 0, 0
        ]),
        start_v=0,
        target_v=0,
        plot=False
    )

# use dmp to imitate our gauss planner
# dmp3.imitate_path(gauss_path_planner.position_path.T, plot=False)
# pos_path_to_next_letter = dmp3.rollout(dmp_steps)[0]
pos_path_to_next_letter = gauss_path_planner.position_path
# dmp3.imitate_path(gauss_path_planner.orientation_path.T, plot=False)
# ori_path_to_next_letter = dmp3.rollout(dmp_steps)[0]
ori_path_to_next_letter = gauss_path_planner.orientation_path

pos_path = np.vstack((pos_path, pos_path_to_next_letter))
ori_path = np.vstack((ori_path, ori_path_to_next_letter))


# control_utils.plot_6dof_path(
#             pos_path=pos_path,
#             ori_path=ori_path,
#             global_start_heading=control_utils.local_to_global_heading(
#                 local_start_heading, T_EE),
#             sampling=sampling,
#             show_axes=False,
#             axes=axes,
#             scale=10,
#             ax=None,
#             show=True
#             # ax=ax,
#             # show=False
#     )

ee_track = []
q_track = []

# create opreational space controller
damping = Damping(robot_config, kv=10)
ctrlr = OSC(robot_config, kp=kp, ko=ko, kv=kv, null_controllers=[damping],
            vmax=None, #vmax=[10, 10],  # [m/s, rad/s]
            # control (x, y, beta, gamma) out of [x, y, z, alpha, beta, gamma]
            ctrlr_dof=ctrlr_dof)


interface.connect()
interface.init_position_mode()
interface.send_target_angles(robot_config.START_ANGLES)
interface.init_force_mode()

try:
    for ii in range(0, pos_path.shape[0]):
        # get arm feedback
        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx('EE', feedback['q'], x=pen_buffer)

        target = np.hstack((pos_path[ii], ori_path[ii]))

        u = ctrlr.generate(
            q=feedback['q'],
            dq=feedback['dq'],
            target=target,
            xyz_offset=pen_buffer
            # target_vel=np.hstack([vel, np.zeros(3)])
            )

        # apply the control signal, step the sim forward
        interface.send_forces(np.array(u, dtype='float32'))

        # track data
        ee_track.append(np.copy(hand_xyz))
        q_track.append(np.copy(feedback['q']))

finally:
    interface.init_position_mode()
    interface.send_target_angles(robot_config.START_ANGLES)
    interface.disconnect()
    # np.savez_compressed('arm_results.npz', q=q_track, ee=ee_track)

    if plot:
        # get the next point in the target trajectory from the dmp
        print('Plotting 6dof path')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax = control_utils.plot_6dof_path(
        #         pos_path=pos_path,
        #         ori_path=ori_path,
        #         global_start_heading=control_utils.local_to_global_heading(
        #             local_start_heading, T_EE),
        #         sampling=sampling,
        #         show_axes=False,
        #         axes=axes,
        #         scale=10,
        #         # ax=None,
        #         # show=True
        #         ax=ax,
        #         show=False
        # )
        # plt.show()

        control_utils.plot_6dof_path_from_q(
                q_track=q_track,
                local_start_heading=local_start_heading,
                robot_config=robot_config,
                sampling=sampling,
                # ax=None,
                ax=ax,
                show=True,
                show_axes=True
        )
        # plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ee_track = np.asarray(ee_track)
        ax.plot(pos_path[:, 0], pos_path[:, 1], pos_path[:, 2], c='b', label='path_planner')
        ax.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], c='g', label='ee_trajectory')
        plt.legend()
        plt.show()

    data = {}
    data['pos_err'], data['ori_err'] = control_utils.calc_error(
        q_track=q_track,
        pos_path=pos_path,
        ori_path=ori_path,
        robot_config=robot_config,
        axes=axes,
        offset=pen_buffer)

    dat = DataHandler('writing_gain_tuning')
    save_loc = 'pos_err=%.2f|ori_err=%.2f|' % (np.sum(data['pos_err']), np.sum(data['ori_err'])) + save_loc
    save_loc = '%s/%s' % (text, save_loc)
    dat.save(data, save_loc, overwrite=True)

    if plot:
        plt.figure()
        plt.title('Error')
        plt.subplot(211)
        plt.title('Position Error')
        plt.plot(data['pos_err'], label='%.2f' % np.sum(data['pos_err']))
        plt.legend()
        plt.subplot(212)
        plt.title('Orientation Error')
        plt.plot(data['ori_err'], label='%.2f' % np.sum(data['ori_err']))
        plt.legend()
        plt.show()
