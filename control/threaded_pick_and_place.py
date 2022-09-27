"""
Running operational space control with a PyGame display, and using the pydmps
library to specify a trajectory for the end-effector to follow, in
this case, a circle.

To install the pydmps library, clone https://github.com/studywolf/pydmps
and run 'python setup.py develop'
"""
import traceback
import time
import numpy as np
import sys
import pydmps
import glfw

from threading import Thread
from abr_control.controllers import OSC, Damping, RestingConfig
from dynamics_adaptation import DynamicsAdaptation
from abr_control.utils import transformations as transform
from gauss_path_planner import GaussianPathPlanner
from circle_path_planner import CirclePathPlanner
from arc import Arc as ArcPathPlanner
import control_utils
from abr_analyze import DataHandler
from abr_control.utils import colors

from color_segmentation import colSegmentation

import matplotlib.pyplot as plt
"""
"""
sim = False
plot = False
track_data = False
use_adapt = False
save_weights = False
load_weights = False
debug = False
backend = 'pd'

if 'plot' in sys.argv:
    plot = True
    print(f"{colors.green}Plotting path{colors.endc}")
if 'sim' in sys.argv:
    sim = True
    print(f"{colors.green}Running in sim{colors.endc}")
if 'track' in sys.argv:
    track_data = True
    print(f"{colors.green}Tracking data{colors.endc}")
if 'cpu' in sys.argv:
    use_adapt = True
    backend = 'cpu'
    print(f"{colors.green}Using CPU Adaptation{colors.endc}")
if 'fpga' in sys.argv:
    use_adapt = True
    backend = 'fpga'
    print(f"{colors.green}Using FPGA Adaptation{colors.endc}")
if 'save' in sys.argv:
    save_weights = True
if 'load' in sys.argv:
    load_weights = True
if 'debug' in sys.argv:
    debug = True

if load_weights:
    weights = np.load(f'{backend}_weights.npz')['weights']
else:
    weights = None

dt = 0.001
error_thres = 0.04
target_error_count = 100 # number of steps to maintain sub error_thres error level
grip_steps = 220
# kp = 50
# kv = 7
# ko = 170
kp = 30
kv = 5
ko = 102
axes = 'rxyz'
# for conversion between quat and euler
# the direction in EE local coordinates that the pen tip is facing
local_start_heading = np.array([0, 0, 1])
# writing instrument offset from EE in EE coodrinates
approach_dist = 0.15
# for plotting to improve arrow visibility
# sampling = 25
# steps after path planner reaches end to allow for controller to catch up
step_limit = 4000 if backend != 'pd' else 0
learning_rate=2.5e-5

# rotate base wrt config default
START_ANGLES = np.array(
    # [4.81, 2.79, 2.62, 4.71, 0.0, 3.04], dtype="float32"
    [6, 2.79, 2.62, 4.71, 0.0, 3.04], dtype="float32"
)

if sim:
    from abr_control.arms.mujoco_config import MujocoConfig
    from abr_control.interfaces.mujoco import Mujoco
    robot_config = MujocoConfig("jaco2")
    interface = Mujoco(robot_config, dt=dt)
else:
    import abr_jaco2
    robot_config = abr_jaco2.Config()
    interface = abr_jaco2.Interface(robot_config)

targets = [
    {
        'name': 'jar',
        # 'pos': np.array([0.1, -0.82, 0.62]),
        'pos': np.array([0.15, -0.82, 0.62]),
        'action': 'pickup', # reach with buffer, open, reach, close, back up
        'global_target_heading': np.array([0, -1, 0]),
        'path': {
            'type': 'arc',
            'kwargs': {'n_timesteps': 2000},
        },
        'mass': False
    },
    # {
    #     'name': 'metal_shelf',
    #     'pos': np.array([0.75, 0.0, 0.73]),
    #     'action': 'dropoff',
    #     'global_target_heading': np.array([1, 0, 0]),
    #     'path': {
    #         'type': 'arc',
    #         'kwargs': {'n_timesteps': 1000},
    #     },
    # },
    {
        'name': 'wooden_shelf2',
        'pos': np.array([0.65, 0.0, 0.83]),
        'action': 'dropoff',
        'global_target_heading': np.array([1, 0, 0]),
        'path': {
            'type': 'arc',
            'kwargs': {'n_timesteps': 1000},
        },
        'mass': False
    },
    {
        'name': 'jar',
        # 'pos': np.array([-0.1, -0.82, 0.62]),
        'pos': np.array([-0.05, -0.82, 0.62]),
        'action': 'pickup', # reach with buffer, open, reach, close, back up
        'global_target_heading': np.array([0, -1, 0]),
        'path': {
            'type': 'arc',
            'kwargs': {'n_timesteps': 2000},
        },
        'mass': True
    },
    # {
    #     'name': 'metal_shelf',
    #     'pos': np.array([0.75, 0.2, 0.73]),
    #     'action': 'dropoff',
    #     'global_target_heading': np.array([1, 0, 0]),
    #     'path': {
    #         'type': 'arc',
    #         'kwargs': {'n_timesteps': 1000},
    #     },
    # },
    {
        'name': 'wooden_shelf2',
        'pos': np.array([0.65, 0.2, 0.84]),
        'action': 'dropoff',
        'global_target_heading': np.array([1, 0, 0]),
        'path': {
            'type': 'arc',
            'kwargs': {'n_timesteps': 1000},
        },
        'mass': True
    },
    {
        'name': 'home',
        'pos': np.array([0.0, 0.0, 0.9]),
        'action': 'reach',
        'global_target_heading': np.array([0, 0, 0]),
        'path': {
            'type': 'linear',
            'kwargs': {
                'max_a': 1,
                'max_v': 1,
                'dt': dt,
                'axes': axes
            },
        },
        'mass': False
    }

]

if not sim:
    colseg = colSegmentation(debug, True)
    segmentation_thread = Thread(
        target = colseg.run
        # target=color_segmentation.run,
        # args=(debug, True)
    )
    segmentation_thread.start()

if sim:
    for target in targets:
        # real arm is slightly above ground due to mount
        target['pos'] -= np.array([0, 0, 0.1])

# save_loc = 'kp=%i|kv=%i|ko=%i|dof=%i' % (kp, kv, ko, int(np.sum(ctrlr_dof)))
save_loc = 'test'

try:
    # instantiate robot config and comm interface
    print('Connecting to arm and interface')
    interface.connect()


    if not sim:
        interface.init_position_mode()
        interface.send_target_angles(START_ANGLES)

        # get our joint angles at start position
        # print('Getting start state information')
        # q_start = interface.get_feedback()['q']
        # T_EE = robot_config.T('EE', q_start)
        #NOTE disconnecting during testing, but eventuall will move to the path
        interface.disconnect()

    # ee_track = []
    # q_track = []
    # open_gripper = []

    # for tt, target in enumerate(targets):
    def gen_path(target, q_start, starting_payload):#, T_EE):
        """
        starting_payload: int
            1 for payload, -1 for no payload
        """
        target['seq'] = []
        target['payload_ctx'] = []
        # offset from location of pickup / dropoff
        # use approach vec to get correct +/- x/y direction
        approach_vector = approach_dist * target['global_target_heading']
        # also offset z to lift back and up
        approach_vector[2] = -0.05 # -approach_dist/2
        # where we want to approach from
        # NOTE from hand writing demo
        # print('Getting target quaternion to align headings')
        # target_quat = control_utils.get_target_orientation_from_heading(
        #         local_start_heading,
        #         target['global_target_heading'],
        #         T_EE,
        #         plot=False)
        #
        # target_euler = transform.euler_from_quaternion(target_quat, axes)

        # pickup along x
        if target['global_target_heading'][0] == 1:
            target_euler = [-0.3, np.pi/2, 0]
        # pickup along -x
        elif target['global_target_heading'][0] == -1:
            target_euler = [0, -np.pi/2, np.pi]
        # pickup along y
        elif target['global_target_heading'][1] == 1:
            # target_euler = [-np.pi/2, 0, np.pi/2]
            target_euler = [-np.pi/2, 0, -np.pi/2]
        # pickup along -y
        elif target['global_target_heading'][1] == -1:
            target_euler = [np.pi/2, -0.2, -np.pi/2]
        else:
            target_euler = [0, 0, 0]

        target['euler'] = target_euler

        # get our start EE position and quat orientation
        start_pos = robot_config.Tx('EE', q_start)#, x=approach_buffer)
        start_quat = robot_config.quaternion('EE', q_start)
        start_euler = transform.euler_from_quaternion(start_quat, axes)

        # get our path to the writing start position
        print(f"Generating path to {target['name']}")
        # path_planner = CirclePathPlanner(
        #             max_a=1,
        #             max_v=1,
        #             dt=0.001,
        #             axes=axes
        #     )

        # WAYPOINT 1: generate the path to next target - buffer
        #====ARC PATH
        if target['path']['type'] == 'arc':
            path_planner_arc = ArcPathPlanner(
                # n_timesteps=arc_steps
                **target['path']['kwargs']
            )

            target_position=[target['pos'][0]-approach_vector[0], target['pos'][1]-approach_vector[1], target['pos'][2]-approach_vector[2]]

            pos_path, vel_path = path_planner_arc.generate_path(
                position=[start_pos[0], start_pos[1], start_pos[2]],
                target_position=target_position
            )

            # get orientation path, this is done in the gauss pp, but not in arc
            error = []
            dist = np.linalg.norm(start_pos-target_position)

            for ee in pos_path:
                error.append(np.sqrt(np.sum((pos_path[-1] - ee) ** 2)))
            error /= dist
            error = 1 - error

            orientation_path = []
            quat0 = transform.quaternion_from_euler(
                start_euler[0],
                start_euler[1],
                start_euler[2],
                axes=axes)

            quat1 = transform.quaternion_from_euler(
                target_euler[0],
                target_euler[1],
                target_euler[2],
                axes=axes)

            for step in error:
                quat = transform.quaternion_slerp(
                    quat0=quat0,
                    quat1=quat1,
                    fraction=step
                )
                orientation_path.append(
                        transform.euler_from_quaternion(
                            quat,
                            axes=axes
                        )
                )

            orientation_path = np.asarray(orientation_path)

            target['seq'].append(
                np.hstack((
                    pos_path,
                    orientation_path,
                    np.array([-1]*len(pos_path))[:, np.newaxis]
                ))
            )
            # our payload remains the same for the start as it was
            # at the end of the previous path
            target['payload_ctx'].append(np.array([starting_payload]*len(pos_path)))

            # need this for the approach, grasp/dropoff, backup
            path_planner_linear = GaussianPathPlanner(
                max_v=1,
                max_a=1,
                dt=dt,
                axes=axes
            )

        elif target['path']['type'] == 'linear':
            #====LINEAR PATH
            path_planner_linear = GaussianPathPlanner(
                **target['path']['kwargs']
            )

            path_planner_linear.generate_path(
                    state=np.array([
                        start_pos[0], start_pos[1], start_pos[2],
                        0, 0, 0,
                        start_euler[0], start_euler[1], start_euler[2],
                        0, 0, 0
                    ]),
                    target=np.array([
                        target['pos'][0]-approach_vector[0], target['pos'][1]-approach_vector[1], target['pos'][2]-approach_vector[2],
                        0, 0, 0,
                        target_euler[0], target_euler[1], target_euler[2],
                        0, 0, 0
                    ]),
                    start_v=0,
                    target_v=0,
                    plot=plot,
                    autoadjust_av=True
                )
            target['seq'].append(
                np.hstack((
                    path_planner_linear.position_path,
                    path_planner_linear.orientation_path,
                    np.array([-1]*len(path_planner_linear.position_path))[:, np.newaxis]
                ))
            )

            target['payload_ctx'].append(
                np.array([starting_payload]*len(path_planner_linear.position_path))
            )

        else:
            raise ValueError(f"{target['path']['type']} is not a valid path planner")

        # If just reaching to a point, end here
        # if pickup or dropoff, then break down into parts
        if target['action'] != 'reach':
            # open_gripper = None
            if target['action'] == 'pickup':
                # open before reaching pickup position
                open_gripper = 1
            # elif target['action'] == 'dropoff':
            #     # do nothing
            #     open_gripper = -1

                # WAYPOINT 2: maintain position and open/maintain gripper depending
                # on if picking up or dropping off, respectively
                target['seq'].append(
                    np.array(
                        [[
                            target['seq'][-1][-1][0],
                            target['seq'][-1][-1][1],
                            target['seq'][-1][-1][2],
                            target['seq'][-1][-1][3],
                            target['seq'][-1][-1][4],
                            target['seq'][-1][-1][5],
                            open_gripper
                        ]] * grip_steps
                    )
                )
                # if target is pickup, there is no payload atm (-1)
                target['payload_ctx'].append(np.array([-1]*grip_steps))
                # print(len(target['seq']))

            # WAYPOINT 3: move up buffer dist to pickup/dropoff loc
            path_planner_linear.generate_path(
                    state=np.array([
                        # last seq, last step, dim x, y, z (0, 1, 2)
                        target['seq'][-1][-1][0], target['seq'][-1][-1][1], target['seq'][-1][-1][2],
                        0, 0, 0,
                        target_euler[0], target_euler[1], target_euler[2],
                        0, 0, 0
                    ]),
                    target=np.array([
                        target['seq'][-1][-1][0]+approach_vector[0], target['seq'][-1][-1][1]+approach_vector[1], target['seq'][-1][-1][2]+approach_vector[2],
                        0, 0, 0,
                        target_euler[0], target_euler[1], target_euler[2],
                        0, 0, 0
                    ]),
                    start_v=0,
                    target_v=0,
                    plot=plot,
                    autoadjust_av=True
                )

            target['seq'].append(
                np.hstack((
                    path_planner_linear.position_path,
                    path_planner_linear.orientation_path,
                    np.array([-1]*len(path_planner_linear.position_path))[:, np.newaxis]
                ))
            )
            # have moved to the pickup/dropoff location, atm the payload ctx is the
            # same as it was at the start of the reach, we have either opened the gripper
            # in preparation to pickup a payload, or have kept it shut because we are
            # still holding the payload
            target['payload_ctx'].append(
                np.array([starting_payload]*len(path_planner_linear.position_path))
            )

            if target['action'] == 'pickup':
                # at target, close gripper
                open_gripper = 0
                # closing gripper, but have not lifted payload yet
                pl_ctx = -1
            elif target['action'] == 'dropoff':
                # at target open gripper
                open_gripper = 1
                # have not released yet, but opening gripper
                pl_ctx = 1 if target['mass'] else -1


            # WAYPOINT 4: open/close hand and maintain position depending on whether
            # dropping off or picking up, respectively
            target['seq'].append(
                np.array(
                    [[
                        target['seq'][-1][-1][0],
                        target['seq'][-1][-1][1],
                        target['seq'][-1][-1][2],
                        target['seq'][-1][-1][3],
                        target['seq'][-1][-1][4],
                        target['seq'][-1][-1][5],
                        open_gripper
                    ]] * grip_steps
                )
            )
            target['payload_ctx'].append(np.array([pl_ctx]*grip_steps))

            # WAYPOINT 5: back up buffer dist and maintain gripper
            path_planner_linear.generate_path(
                    state=np.array([
                        # last seq, last step, dim x, y, z (0, 1, 2)
                        target['seq'][-1][-1][0], target['seq'][-1][-1][1], target['seq'][-1][-1][2],
                        0, 0, 0,
                        target_euler[0], target_euler[1], target_euler[2],
                        0, 0, 0
                    ]),
                    target=np.array([
                        target['seq'][-1][-1][0]-approach_vector[0], target['seq'][-1][-1][1]-approach_vector[1], target['seq'][-1][-1][2]-approach_vector[2],
                        0, 0, 0,
                        target_euler[0], target_euler[1], target_euler[2],
                        0, 0, 0
                    ]),
                    start_v=0,
                    target_v=0,
                    plot=plot,
                    autoadjust_av=True
                )

            # if target['action'] == 'pickup':
            #     # backing off, keep gripper closed (do nothing)
            #     open_gripper = -1
            # elif target['action'] == 'dropoff':
            #     # backing off, close gripper
            #     open_gripper = 0
            open_gripper = -1

            target['seq'].append(
                np.hstack((
                    path_planner_linear.position_path,
                    path_planner_linear.orientation_path,
                    np.array([open_gripper]*len(path_planner_linear.position_path))[:, np.newaxis]
                ))
            )
            # backing up and payload is opposite of what it was before the pickup/dropoff
            if target['mass']:
                target['payload_ctx'].append(
                    np.array([-1*pl_ctx]*len(path_planner_linear.position_path))
                )
            else:
                # empty jar, so mark context as "no mass"
                target['payload_ctx'].append(
                    np.array([-1]*len(path_planner_linear.position_path))
                )
            # print(len(target['seq']))

            # WAYPOINT 6: if we just dropped something off, close the hand
            # now that we've backed off
            if target['action'] == 'dropoff':
                target['seq'].append(
                    np.array(
                        [[
                            target['seq'][-1][-1][0],
                            target['seq'][-1][-1][1],
                            target['seq'][-1][-1][2],
                            target['seq'][-1][-1][3],
                            target['seq'][-1][-1][4],
                            target['seq'][-1][-1][5],
                            0
                        ]] * grip_steps
                    )
                )
                target['payload_ctx'].append(np.array([-1]*grip_steps))

            print('LEN SEQ: ', len(target['seq']))
            print('SHAPES')
            for s in target['seq']:
                print(s.shape)
            # Now we are backed away from the pickup/drop off location and
            # can proceed to the next target

    # create opreational space controller
    damping = Damping(robot_config, kv=10)
    resting = RestingConfig(
        robot_config,
        rest_angles=[None, 3.14, None, None, None, None]
    )
    # for returning home, can have lower gains
    # ctrlr_pos = OSC(robot_config, kp=25, ko=0, kv=5, null_controllers=[damping],
    ctrlr_pos = OSC(robot_config, kp=200, ko=0, kv=7, null_controllers=[damping],
                vmax=None,
                ctrlr_dof=[True, True, True, False, False, False])

    ctrlrx = OSC(robot_config, kp=kp, ko=ko, kv=kv, null_controllers=[damping],
                vmax=None,
                # ctrlr_dof=[True, True, True, True, False, True])
                ctrlr_dof=[True, True, True, True, True, True])
    ctrlry = OSC(robot_config, kp=kp, ko=ko, kv=kv, null_controllers=[damping],
                vmax=None,
                ctrlr_dof=[True, True, True, False, True, True])




    # create our adaptive controller
    if use_adapt:
        adapt = DynamicsAdaptation(
            n_neurons=1000,
            n_ensembles=1,
            n_input=3,  # we apply adaptation on the most heavily stressed joints
            n_output=3,
            pes_learning_rate=learning_rate,
            means=[3.14, 3.14, 3.14],
            variances=[3.14, 3.14, 3.14],
            spherical=True,
            backend=backend,
            payload_ctx=True,
            weights=weights
        )
    if track_data:
        q_track = []
        train_track = []
        input_track = []

    interface.connect()

    if not sim:
        interface.init_position_mode()

    interface.send_target_angles(START_ANGLES)

    # if not sim:
    #     interface.init_force_mode()

    for target_count, target in enumerate(targets):
        print('Getting start state information')
        feedback = interface.get_feedback()

        if target_count == 0:
            # generate the first path before turning on force mode
            gen_path(
                target,
                np.copy(feedback['q']),
                targets[target_count-1]['payload_ctx'][-1][-1] if target_count > 0 else -1
            )

            interface.init_force_mode()

        # for proper visualization of gripper in mujoco
        last_grip = None
        print(f"TARGET: {target['name']}")
        for seq_count, seq in enumerate(target['seq']):
            print(f"SEQ: {seq_count}")
            # tracks consecutive steps where we are below error threshold
            at_error_count = 0
            # path counter
            ii = -1
            # take modulus of counter to print occasionally
            print_cnt = 0
            # counts steps after the end of the path to allow for buffer to reach
            step_limit_cnt = 0

            if use_adapt:
                u_adapt = np.zeros(6)

            # Last dim is gripper command
            # if -1 then hand is not opening so use error count
            if seq[-1][-1] == -1:
                at_count = target_error_count
            # otherwise we are opening the hand so just wait however
            # long it takes to open the hand (grip_steps param)
            else:
                at_count = grip_steps

            # for properly closing mujoco sim
            exit_sim = False

            # tracks if all conditions are met to continue to next target
            move_to_next_target = False
            path_gen_thread_started = False
            next_path_ready = False

            # make sure we are at error thres and the next path is ready
            while not move_to_next_target:
                print_cnt += 1
                ii += 1

                # get arm feedback
                feedback = interface.get_feedback()

                # calculate error and track if below threshold
                hand_xyz = robot_config.Tx('EE', feedback['q'])#, x=approach_buffer)
                error = np.linalg.norm(hand_xyz - seq[-1][:3])
                if error < error_thres:
                    at_error_count += 1
                else:
                    at_error_count = 0

                # save joint data to file
                if track_data:
                    q_track.append(feedback['q'])

                if sim:
                    # check for closing sim
                    if interface.viewer.exit:
                        glfw.destroy_window(interface.viewer.window)
                        exit_sim = True
                        break

                # at end of path, start counting towards our limit
                if ii > seq.shape[0]-1:
                    ii = seq.shape[0]-1
                    # print(f"{colors.yellow}AT END OF CURRENT SEQUENCE{colors.endc}")

                    # check if at the end of the target sequence, and not
                    # already at the last target, if so start thread to
                    # generate next path
                    if seq_count == len(target['seq'])-1:
                        # print(f"{colors.yellow}AT END OF SEQUENCE LIST{colors.endc}")
                        if target_count < len(targets)-1:
                            # print(f"{colors.yellow}NOT AT END OF TARGET LIST{colors.endc}")
                            # if not at last target, check if thread started to gen next path
                            if not path_gen_thread_started:
                                # print(f"{colors.yellow}STARTING PATH GEN THREAD{colors.endc}")
                                path_gen_thread_started = True
                                # === START THREAD TO GEN NEXT PATH ===
                                new_thread = Thread(
                                    target=gen_path,
                                    args=(
                                        targets[target_count+1],
                                        np.copy(feedback['q']),
                                        # targets[target_count-1]['payload_ctx'][-1][-1] if target_count > 0 else -1
                                        targets[target_count]['payload_ctx'][-1][-1]
                                    )
                                )
                                new_thread.start()
                            elif new_thread.is_alive():
                                # print(f"{colors.yellow}WAITING FOR PATH TO GENERATE{colors.endc}")
                                next_path_ready = False
                            elif not new_thread.is_alive():
                                # print(f"{colors.yellow}PATH READY{colors.endc}")
                                next_path_ready = True
                                new_thread.handled = True
                        else:
                            # print(f"{colors.yellow}AT END OF TARGET LIST{colors.endc}")
                            # there is no next path, so set to True to exit loop
                            next_path_ready = True
                    else:
                        # still have sequences to go through, so no need to generate a path yet
                        next_path_ready = True

                    # at the end of the current seq path, track how many steps pass
                    step_limit_cnt += 1

                    if next_path_ready:
                        # print(f"{colors.green}PATH READY CHECK: PASS{colors.endc}")
                        # check within error thres for set number of steps
                        if at_error_count > target_error_count:
                            print(f"{colors.green}AT ERROR COUNT CHECK: PASS{colors.endc}")
                            move_to_next_target = True
                        # check if buffer time passed (0 for PD, non zero for adaptive)
                        elif step_limit_cnt > step_limit:
                            move_to_next_target = True
                            print(f"{colors.red}REACHED STEP LIMIT{colors.endc}")

                if print_cnt % 1000 == 0:
                    hand_abg = transform.euler_from_matrix(
                        robot_config.R("EE", feedback["q"]), axes=axes
                    )
                    print(f"pos error: {error}")
                    # print(hand_xyz-target['pos'])
                    print('xyz: ', hand_xyz-seq[-1][:3])
                    print('abg: ', np.asarray(list(hand_abg))-np.asarray(list(target['euler'])))

                # pick/place along x
                if abs(target['global_target_heading'][0]) == 1:
                    u = ctrlrx.generate(
                        q=feedback['q'],
                        dq=feedback['dq'],
                        target=seq[ii][:6],
                        )
                    training_signal = ctrlrx.training_signal[1:4]

                # pick/place along y
                elif abs(target['global_target_heading'][1]) == 1:
                    u = ctrlry.generate(
                        q=feedback['q'],
                        dq=feedback['dq'],
                        target=seq[ii][:6],
                        )
                    training_signal = ctrlry.training_signal[1:4]

                # return home with xyz control only
                else:
                    u = ctrlr_pos.generate(
                        q=feedback['q'],
                        dq=feedback['dq'],
                        target=seq[ii][:6],
                        )
                    training_signal = ctrlr_pos.training_signal[1:4]

                # TODO set dimensions being adapted as list of bools
                if use_adapt:
                    u_adapt[1:4] = adapt.generate(
                        input_signal=np.asarray(feedback["q"][1:4]),
                        training_signal=np.array(training_signal),
                        payload_ctx=target['payload_ctx'][seq_count][ii]
                    )
                    # print(target['payload_ctx'][seq_count][ii])

                    u += u_adapt

                    if track_data:
                        train_track.append(training_signal)
                        input_track.append(feedback["q"][1:4])

                # open / close gripper
                if not sim:
                    if step_limit_cnt == 0:
                        if seq[ii][-1] == 1:
                            interface.open_hand(True)
                        elif seq[ii][-1] == 0:
                            interface.open_hand(False)

                    # apply the control signal
                    interface.send_forces(np.array(u, dtype='float32'))

                else:
                    # sim requires extra work because gripper is simulated
                    # in force mode, whereas the real gripper is in position mode.
                    # Because of this a constant control signal needs to be sent
                    # to the gripper. To account for this keep track of the last
                    # grip command for moments when no open/close command
                    # is sent to the real arm

                    # grip force
                    uf = 4
                    if seq[ii][-1] == 1:
                        u_grip = [uf, uf, uf]
                        last_grip = u_grip
                    elif seq[ii][-1] == 0:
                        u_grip = [-uf, -uf, -uf]
                        last_grip = u_grip
                    else:
                        if last_grip is None:
                            # first target, so leave gripper as is
                            u_grip = [0, 0, 0]
                        else:
                            u_grip = last_grip

                    interface.send_forces(np.hstack((u, u_grip)))

                    # visualization for debugging
                    target_geom_id = interface.sim.model.geom_name2id("target")
                    green = [0, 0.9, 0, 0.5]
                    red = [0.9, 0, 0, 0.5]
                    if u_grip[0] > 0:
                        interface.set_mocap_xyz("target", [0.0, 0, 1.2])
                        interface.sim.model.geom_rgba[target_geom_id] = green
                        last_grip = u_grip
                    elif u_grip[0] < 0:
                        interface.set_mocap_xyz("target", [0.0, 0, 1.2])
                        interface.sim.model.geom_rgba[target_geom_id] = red
                        last_grip = u_grip
                    else:
                        interface.set_mocap_xyz("target", [0, 0, -1.2])

                    interface.set_mocap_xyz("target_orientation", seq[ii][:3])
                    interface.set_mocap_orientation(
                        "target_orientation",
                        transform.quaternion_from_euler(
                            seq[ii][3],
                            seq[ii][4],
                            seq[ii][5],
                            axes=axes
                        )
                    )

            if exit_sim:
                break

except Exception as e:
    print(e)
    print(traceback.format_exc())
finally:
    if not sim:
        interface.init_position_mode()
        interface.send_target_angles(START_ANGLES)
    interface.disconnect()

    # stop segmentation thread
    colseg.running = False

    if save_weights:
        weights = adapt.get_weights()
        np.savez_compressed(f'{backend}_weights.npz', weights=weights)

    if track_data:
        np.savez_compressed(
            'data_track.npz',
            q=q_track,
            input_signal=input_track,
            training_signal=train_track
        )
        plt.figure
        plt.plot(np.asarray(q_track))
        plt.show()
    # np.savez_compressed('arm_results.npz', q=q_track, ee=ee_track)

    # if plot:
    #     # get the next point in the target trajectory from the dmp
    #     print('Plotting 6dof path')
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     # ax = control_utils.plot_6dof_path(
    #     #         pos_path=pos_path,
    #     #         ori_path=ori_path,
    #     #         global_start_heading=control_utils.local_to_global_heading(
    #     #             local_start_heading, T_EE),
    #     #         sampling=sampling,
    #     #         show_axes=False,
    #     #         axes=axes,
    #     #         scale=10,
    #     #         # ax=None,
    #     #         # show=True
    #     #         ax=ax,
    #     #         show=False
    #     # )
    #     # plt.show()
    #
    #     control_utils.plot_6dof_path_from_q(
    #             q_track=q_track,
    #             local_start_heading=local_start_heading,
    #             robot_config=robot_config,
    #             sampling=sampling,
    #             # ax=None,
    #             ax=ax,
    #             show=True,
    #             show_axes=True
    #     )
    #     # plt.show()
    #
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     ee_track = np.asarray(ee_track)
    #     ax.plot(pos_path[:, 0], pos_path[:, 1], pos_path[:, 2], c='b', label='path_planner')
    #     ax.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], c='g', label='ee_trajectory')
    #     plt.legend()
    #     plt.show()
    #
    # data = {}
    # data['pos_err'], data['ori_err'] = control_utils.calc_error(
    #     q_track=q_track,
    #     pos_path=pos_path,
    #     ori_path=ori_path,
    #     robot_config=robot_config,
    #     axes=axes)#,
    #     # offset=pen_buffer)
    #
    # dat = DataHandler('writing_gain_tuning')
    # save_loc = 'pos_err=%.2f|ori_err=%.2f|' % (np.sum(data['pos_err']), np.sum(data['ori_err'])) + save_loc
    # # save_loc = '%s/%s' % (text, save_loc)
    # save_loc = '%s' % (save_loc)
    # dat.save(data, save_loc, overwrite=True)
    #
    # if plot:
    #     plt.figure()
    #     plt.title('Error')
    #     plt.subplot(211)
    #     plt.title('Position Error')
    #     plt.plot(data['pos_err'], label='%.2f' % np.sum(data['pos_err']))
    #     plt.legend()
    #     plt.subplot(212)
    #     plt.title('Orientation Error')
    #     plt.plot(data['ori_err'], label='%.2f' % np.sum(data['ori_err']))
    #     plt.legend()
    #     plt.show()
