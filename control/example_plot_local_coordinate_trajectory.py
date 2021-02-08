import abr_jaco2
import numpy as np
import control_utils as utils
# we want our local coordinate heading [1, 1, 1] to be plotted
local_start_heading = [1, 1, 1]
# recorded joint information during arm reach
q_track = np.load('q_track.npz')['q']
# plot and sample trajectory arrows every 50 steps for visibility
utils.plot_path_from_q(
        q_track=q_track,
        local_start_heading=local_start_heading,
        robot_config=abr_jaco2.Config(),
        sampling=50
)
