import numpy as np
from abr_control.utils import transformations as transform
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
import glfw
import time

class MujocoMirror:
    def __init__(self, robot_name, dt=0.001, axes='rxyz', real_to_sim_offset=None):
        self.robot_config = MujocoConfig("jaco2")
        self.interface = Mujoco(self.robot_config, dt=dt)
        self.interface.connect(
            joint_names=[
                'joint0',
                'joint1',
                'joint2',
                'joint3',
                'joint4',
                'joint5',
                'joint_thumb',
                'joint_index',
                'joint_pinky'
            ]
        )
        # self.interface.get_feedback()
        # self.interface.send_forces(np.zeros(9))

        self.axes = axes
        self.dt = dt

        # real arm has a vertical offset due to mounting base
        # this offsets targets so visualization aligns with real arm
        if real_to_sim_offset is None:
            # real_to_sim_offset = [0, 0, -0.1]
            real_to_sim_offset = [0, 0, -0.05]
        self.real_to_sim_offset = real_to_sim_offset

    def threaded_step(self):
        self.running = True
        self.q = None
        self.target = None
        self.filtered_target = None
        while self.running:
            if self.q is not None:
                self.step(self.q, self.target, self.filtered_target)
                self.q = None
            else:
                time.sleep(self.dt)

    def step(self, q, target=None, filtered_target=None):
        # print('simstep')

        # visualization for debugging
        if target is not None:
            # print(target)
            # print(type(target))
            target[0] += self.real_to_sim_offset[0]
            target[1] += self.real_to_sim_offset[1]
            target[2] += self.real_to_sim_offset[2]

            self.interface.set_mocap_xyz(
                "target_orientation",
                target[:3]
            )
            self.interface.set_mocap_orientation(
                "target_orientation",
                transform.quaternion_from_euler(
                    target[3],
                    target[4],
                    target[5],
                    axes=self.axes
                )
            )

        if filtered_target is not None:
            filtered_target[0] += self.real_to_sim_offset[0]
            filtered_target[1] += self.real_to_sim_offset[1]
            filtered_target[2] += self.real_to_sim_offset[2]

            self.interface.set_mocap_xyz(
                "path_planner_orientation",
                filtered_target[:3]
            )
            self.interface.set_mocap_orientation(
                "path_planner_orientation",
                transform.quaternion_from_euler(
                    filtered_target[3],
                    filtered_target[4],
                    filtered_target[5],
                    axes=self.axes
                )
            )
        self.interface.set_joint_state(
            q=q,
            dq=np.zeros(len(q))
        )
        # self.interface.sim.step()
        self.interface.viewer.render()

    def close_sim(self):
        self.interface.disconnect()
        glfw.destroy_window(self.interface.viewer.window)
