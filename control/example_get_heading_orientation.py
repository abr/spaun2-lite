import abr_jaco2
import numpy as np
import control_utils as utils

# getting our target orientation to move our local heading to a global heading
# initialize our robot config for neural controllers
robot_config = abr_jaco2.Config()
# create our interface for the jaco2
interface = abr_jaco2.Interface(robot_config, display_error_level=2)
interface.connect()
interface.init_position_mode()
interface.send_target_angles(robot_config.START_ANGLES)
# get feedback
state = interface.get_feedback()['q']

T_EE = robot_config.T('EE', state)
# we want to align our EE z axis to some global target heading
local_start_heading = [0, 0, 1]
# our target global heading is along the x axis
global_target_heading = [1, 0, 0]

# get the quaternion orientation that will align our local z axis with the global x axis
quat = utils.get_target_orientation_from_heading(local_start_heading, global_target_heading, T_EE, plot=True)

interface.disconnect()


