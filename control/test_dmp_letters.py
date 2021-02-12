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
import control_utils

import matplotlib.pyplot as plt
dt = 0.0005
np.set_printoptions(threshold=sys.maxsize)

# load our alphanumerical path
if len(sys.argv) > 1:
    text = sys.argv[1]
else:
    text = '1'

# character size [x, y] in meters
char_size = [0.05, 0.05]
# spacing between letters in meters
letter_spacing = char_size[0] * 2
# how many steps for each dmp path (currently all the same)
# dmp_steps = 2000
# for plotting to improve arrow visibility
sampling = 5
# location to go to for writing
writing_origin = np.array([1, 0, 0.5])

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
            char_path[0, :] *= (2 * char_size[0])
            char_path[1, :] *= (0.5 * char_size[1])
            text_paths[char] = char_path

            if plot:
                plt.figure()
                plt.title('Saved alphanumeric path')
                plt.plot(char_path[0], char_path[1], label='char')
                plt.legend()
                plt.xlim(-0.25, 0.2)
                plt.ylim(-0.25, 0.2)
                plt.show()

    return text_paths

text_paths = load_paths(text, plot=False, char_size=char_size)

dmp2 = pydmps.dmp_discrete.DMPs_discrete(
    n_dmps=2,
    n_bfs=500,
    ay=np.ones(2) * 10,
    dt=dt
)

print('timesteps: ', dmp2.timesteps)

# hard code the orientation for now while debugging
ori_path = np.array([[1.57, 0, 0]])

for ii, char in enumerate(text):
    # generate the writing position path
    dmp2.imitate_path(text_paths[char], plot=False)
    dmp_pos = dmp2.rollout()[0]

    # add our last point on the way to the board since the dmp begins at the origin
    dmp_pos = np.asarray(dmp_pos)
    # plt.figure()
    # plt.title('DMP imitated alphanumeric path')
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
    dmp_pos = np.hstack((np.ones((dmp_pos.shape[0], 1))*writing_origin[0], dmp_pos))
    # spacing along line
    max_horz_point = max(dmp_pos[:, 1])
    dmp_pos[:, 1] += writing_origin[1] # + max_horz_point
    # vertical alignment
    dmp_pos[:, 2] += writing_origin[2]
    dmp_ori = np.ones((dmp_pos.shape[0], 3))*ori_path[-1]
    # shift the writing origin over in one dimension as we write
    writing_origin[1] += letter_spacing + max_horz_point

    # if ii == 0:
    pos_path = np.copy(dmp_pos)
    #     ori_path = np.copy(dmp_ori)
    #     print('first pass')
    # else:
    #     pos_path = np.vstack((np.copy(pos_path), np.copy(dmp_pos)))
    #     ori_path = np.vstack((np.copy(ori_path), np.copy(np.ones((dmp_steps, 3))*ori_path[-1])))
    #     print('subsequent pass')
    # print('pos_path shape: ', pos_path.shape)
plt.figure()
plt.title('Full DMP Path')
plt.plot(pos_path[:, 1], pos_path[:, 2])
# plt.plot(dmp2.goal[0], dmp2.goal[1], 'rx', mew=3)
plt.show()

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
