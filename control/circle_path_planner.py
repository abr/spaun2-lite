import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611
# from abr_control.controllers.path_planners.path_planner import PathPlanner
from abr_control.controllers import path_planners
from abr_control.utils import transformations as transform
# from gauss_path_planner import GaussPathPlanner

class CirclePathPlanner(path_planners.path_planner.PathPlanner):

    # def __init__(self, n_timesteps, dt, startup_steps=None, NED=False, axes='rxyz'):
    def __init__(self, max_a, max_v, dt, NED=False, axes='rxyz', loops=0):
        """
        NED: boolean, Optional (Default: False)
            whether or not using NED coordinate system where +z is down (use lhr instead or rhr)
        """
        self.loops=loops
        self.dt = dt
        self.axes = axes
        self.max_v = max_v
        self.max_a = max_a
        self.NED = NED


    #TODO move this into a class with different ramps, or to utils
    def _get_gauss_profile(self, max_a, max_v, start_v, target_v):

        # print('start_v: ', start_v)
        # print('path: ', starting_vel_profile[0])
        # print('target_v; ', target_v)
        # print('path: ', ending_vel_profile[-1])

        # ramp_up_time = max_v/max_a
        # TODO if we start at a vel > max_v our number of steps becomes negative
        # when getting x and throws an error in the linspace func
        ramp_up_time = abs(max_v-start_v)/max_a
        # Amplitude of Gaussian is max speed, get our sigma from here
        # s = 1/ (max_v * np.sqrt(np.pi*2))
        s = 1/ (abs(max_v-start_v) * np.sqrt(np.pi*2))
        # Go 3 standard deviations so our tail ends get close to zero
        u = 3*s
        # print('maxv: ', max_v)
        # print('start_v: ', start_v)
        # print('ramp_up_time: ', ramp_up_time)
        x = np.linspace(0, u, int(ramp_up_time/self.dt))
        # print('number of ramp steps: ', len(x))
        starting_vel_profile = 1 * (
                1/(s*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-u)/s)**2)
        ) #+ start_v

        # print(starting_vel_profile)
        # plt.figure()
        # plt.plot(x, starting_vel_profile, label='initial')
        # shift down so our profile starts at zero
        starting_vel_profile -= starting_vel_profile[0]
        # plt.plot(x, starting_vel_profile, label='shift to zero')

        # scale back up so we reach our target v
        starting_vel_profile *= ((max_v-start_v) / starting_vel_profile[-1])
        # plt.plot(x, starting_vel_profile, label='scale_to_max_v')

        # add to our baseline starting velocity
        starting_vel_profile += start_v
        # plt.plot(x, starting_vel_profile, label='add to baseline')
        # plt.legend()
        # plt.show()

        # if we start and end at the same velocity, mirror the vel curve
        if start_v == target_v:
                ending_vel_profile = starting_vel_profile[::-1]
        else:
            ramp_down_time = (max_v-target_v)/max_a
            # Amplitude of Gaussian is max speed, get our sigma from here
            s = 1/ ((max_v-target_v) * np.sqrt(np.pi*2))
            # Go 3 standard deviations so our tail ends get close to zero
            u = 3*s
            x = np.linspace(0, u, int(ramp_down_time/self.dt))
            ending_vel_profile = 1 * (
                    1/(s*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-u)/s)**2)
            ) #+ target_v

            # shift down so our profile starts at zero
            ending_vel_profile -= ending_vel_profile[0]

            # scale back up so we reach our target v
            ending_vel_profile *= ((max_v-target_v) / ending_vel_profile[-1])

            # add to our baseline ending velocity
            ending_vel_profile += target_v

            ending_vel_profile = ending_vel_profile[::-1]

        # ending_vel_profile = np.insert(ending_vel_profile, -1, target_v)
        # print('length start: ', len(starting_vel_profile))
        # print('length end: ', len(ending_vel_profile))

        return starting_vel_profile, ending_vel_profile


    # def _get_linear_profile(self, max_a, max_v, start_v, target_v):
    #     # LINEAR RAMP FOR DEBUGGING
    #     starting_vel_profile = np.linspace(start_v, max_v, int(((max_v-start_v)/max_a)/self.dt))
    #     ending_vel_profile = np.linspace(max_v,  target_v, int(((max_v-target_v)/max_a)/self.dt))
    #
    #     print('length start: ', len(starting_vel_profile))
    #     print('length end: ', len(ending_vel_profile))
    #     return starting_vel_profile, ending_vel_profile



    # def generate_path(self, state, circle_origin, loops=None, plot=False, start_v=None, target_v=None):
    def generate_path(self, state, target, loops=None, plot=False, start_v=0, target_v=0):
        """
        Creates a circular path using a gaussian to get to max velocity for the first
        loop and a constant velocity, then ends in the same gaussian curve to slow down

        NOTE that the path starts along the +x axis, but can be shifted by the axis_offset using the right hand rule

        Parameters
        ----------
        state: np.array of size 3
            start position, used to set z and xy offset [m]
        circle_origin: np.array of size 3
            x and y set the center of the circle
            if circle_origin z does not equal the state z, then
            a helical path will be taken, circling the circle_origin xy,
            from the state z to the target z, with a circle of radius r
            Note that a path from state to the start of the circle is not set
        r: float, Optional, (Default: None)
            radius of circle [m]
            if None the radius is set by the distance from state to circle origin
            If radius is set to be different, than a gaussian path planner will be
            used to get to the start of the circle
        loops: float, Optional (Default: 1)
            number of loop arounds circle
        direction: string either 'cw' or 'ccw'
            sets the direction of movement
            This does not follow the right hand rule when in NED, but the reverse since it is
            more intuitive to think of cw and ccw from the perspective of looking down
        axis_offset: float, Optional (Default: 0)
            the path starts along the +x axis, this value in radians will offset the starting
            point of the circle path using the right hand rule
        """
        #NOTE hack for now for consistent naming
        circle_origin = target[:3]

        #NOTE hack for now, loops should be defined on generate, but passing in to init
        # to simplify the different circle paths with varying loops in the path node
        if loops is None:
            loops = self.loops
        # print('CIRCLE ORIGIN: ', circle_origin)
        # print('STATE: ', state)
        # print('TARGET: ', target)
        self.position_path = []
        state = np.asarray(state)
        circle_origin = np.asarray(circle_origin)

        r = np.linalg.norm(state[:2] - circle_origin[:2])
        # r1 = np.linalg.norm(state[:2] - circle_origin[:2])
        # r2 = np.linalg.norm(target[:2] - circle_origin[:2])
        assert (abs(r)>0, 'distance between start and origin must be > 0')

        self.final_target = circle_origin[:3]
        self.state = state
        # self.target = target

        # cw
        if loops < 0:
            direction = -1
        # ccw
        elif loops > 0:
            direction = 1

        loops = abs(loops)

        if self.NED:
            # NED coordinates have +z point down, so flip our rotation direction
            direction *= -1

        # get our xy velocity profile to determine the path of the circle
        # max_v = 2*np.pi*r/(self.n_timesteps * self.dt) # m/s
        # print('max_v: ', self.max_v)
        # print('r: ', r)
        max_w = self.max_v/r # rad/s
        # print('max_w: ', max_w)

        starting_vel_profile, ending_vel_profile = self._get_gauss_profile(
                max_a=self.max_a,
                max_v=self.max_v,
                # max_v=max_w,
                start_v=start_v,
                target_v=target_v,
                )

        # check how far we travel on startup
        # w = theta/t = v/r | s = r*theta = r*w*t = v*t
        # NOTE this is still a linear velocity profile, not angular velocity
        arc_travelled = (np.sum(starting_vel_profile) + np.sum(ending_vel_profile)) * self.dt
        # we use the same profile to slow down at the end, so determine the number of
        # steps at const speed required
        total_radians = loops*np.pi*2

        if arc_travelled < total_radians:
            remaining_radians = total_radians - arc_travelled
            # print("startup and slowdown covers %f radians" % arc_travelled)
            # print("remaining radians after start and slow down: ", remaining_radians)
            # constant_speed_steps = int(remaining_radians / max_w / self.dt)
            constant_speed_steps = int(remaining_radians / self.max_v / self.dt)
            # print('const vel steps: ', constant_speed_steps)
            circle_vel_profile = np.hstack((
                starting_vel_profile,
                # np.ones(constant_speed_steps) * max_w,
                np.ones(constant_speed_steps) * self.max_v,
                ending_vel_profile
            ))
        else:
            scale = total_radians / (arc_travelled)
            circle_vel_profile = np.hstack((
                scale*starting_vel_profile,
                scale*ending_vel_profile
            # ['stem_bottom', 'circle_down'],
            ))


        total_steps = len(circle_vel_profile)
        total_ramp_steps = len(starting_vel_profile) + len(ending_vel_profile)

        if circle_origin[2] != state[2]:
            z_dist = circle_origin[2] - state[2]
            z_scale = z_dist/(total_radians)
            z_vel_profile = circle_vel_profile * z_scale

            # equation from solving integral under speed up, const speed, slow down curve
            # and solving for v_max which occurs at x = startup_steps
            # max_vz = abs(circle_origin[2] - state[2])/(len(circle_vel_profile)*self.dt)
            # if loops >= 1:
            #     # travelling at least 1 loop, in which case the timesteps set are per loop
            #     max_vz = ((circle_origin[2] - state[2])/self.dt)/(2 + total_ramp_steps*loops)
            # else:
            #     # if travelling a portion of one loop, the timesteps sets the steps to complete this portion
            #     max_vz = ((circle_origin[2] - state[2])/self.dt)/(2 + total_ramp_steps)
            #
            # # TODO this should use z component of start_v
            # starting_z_vel_profile, ending_z_vel_profile = self._get_gauss_profile(
            # # starting_z_vel_profile, ending_z_vel_profile = self._get_linear_profile(
            #         # max_a=self.max_a,
            #         max_a=(max_vz/(len(starting_vel_profile)*self.dt)),
            #         max_v=max_vz,
            #         # max_v=self.max_v,
            #         # start_v=start_v,
            #         start_v=0,
            #         target_v=0,
            #         )
            #
            # z_travelled = (np.sum(starting_z_vel_profile) + np.sum(ending_z_vel_profile)) * self.dt
            # print('startup and slowdown covers %f m in z' % z_travelled)
            #
            # if z_travelled < z_dist:
            #     remaining_z = z_dist - z_travelled
            #     print('after start up and slow down have %fm left' % remaining_z)
            #     constant_z_steps = abs(int(remaining_z / self.max_v / self.dt))
            #     print('%i steps of const speed' % remaining_steps)
            #
            #     z_vel_profile = np.hstack((
            #         starting_z_vel_profile,
            #         np.ones(constant_z_steps) * max_vz,
            #         ending_z_vel_profile
            #     ))
            #
            # else:
            #     scale = z_dist / z_travelled
            #     z_vel_profile = np.hstack((
            #         scale*starting_vel_profile,
            #         scale*ending_vel_profile
            #     ))
            #

        else:
            z_vel_profile = np.zeros(len(circle_vel_profile))
        # print('z: ', len(z_vel_profile))
        # print('xy: ', len(circle_vel_profile))

        #TODO automate axis offset based on state and origin info
        vector_2 = state[:2] - circle_origin[:2]
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        axis_offset = math.atan2(
                unit_vector_2[1], unit_vector_2[0]
                # unit_vector_2[0]-unit_vector_1[0]
                )

        theta = 0
        theta_offset = axis_offset
        theta += theta_offset
        x = []
        y = []
        z = []
        z_pos=state[2]

        for ii, step in enumerate(circle_vel_profile):
            # if abs(theta-theta_offset) >= abs(2 * np.pi * loops):
            #     break
            x.append(r*np.cos(theta) + circle_origin[0])
            y.append(r*np.sin(theta)  + circle_origin[1])
            z_pos += z_vel_profile[ii]*self.dt
            z.append(z_pos)
            theta += step*self.dt*direction

        # z = np.linspace(state[2], circle_origin[2], len(x))
        self.position_path = np.array([x, y, z]).T


        self.velocity_path = np.asarray(np.gradient(self.position_path, self.dt, axis=0))
        self.orientation_path = []
        for pos in self.position_path:
            err = circle_origin[:3] - pos
            yaw = math.atan2(err[1], err[0])
            self.orientation_path.append([0, 0, yaw])
        self.orientation_path = np.asarray(self.orientation_path)

        # TODO have to account for discrepancy around 2pi to 0 jump
        self.ang_velocity_path = np.asarray(
                np.gradient(self.orientation_path, self.dt, axis=0))

        # this goes through and removes spikes in angular velocity from the
        # modulo operation applied on the orientation path, causing jumps
        # when we roll over from 2pi to 0
        # TODO get angular velocity from quaternion path
        for ii, ang_vel in enumerate(self.ang_velocity_path):
            if ii>0:
                for jj in range(0, 3):
                    abs_diff = abs(self.ang_velocity_path[ii, jj] - self.ang_velocity_path[ii-1, jj])
                    if abs_diff > 100:
                        # print('abs_diff: ', abs_diff)
                        self.ang_velocity_path[ii, jj] = self.ang_velocity_path[ii-1, jj]

        self.n_timesteps = len(self.position_path)
        self.n = 0

        self.path = np.hstack(
                    (np.hstack(
                        (np.hstack((self.position_path, self.velocity_path)),
                            self.orientation_path)),
                        self.ang_velocity_path)
                )

        self.n_timesteps = len(self.path)

        # save as a self variable so we don't zero out target vel on our last step if we want non-zero vel
        self.target_v = target_v

        # print('ORI PATH START: ', self.orientation_path[0])
        # print('ORI PATH END: ', self.orientation_path[-1])
        if plot:
            self._plot(state=state, target=target)
        # NOTE should be using length of path here in case it's changed
        self.time_to_converge = total_steps * self.dt
        return self.position_path, self.velocity_path, self.orientation_path

    # def _path_to_circle_start(self, r, start_state):
    #     # we want our final orientation to match the start of the circle path
    #     yaw = self.orientation_path[0][2]
    #     # get the first position of our circle path to use as the target for the gaussian path
    #     circle_start_state = np.hstack((self.position_path[0], self.velocity_path[0], [0, 0, yaw, 0, 0, 0]))
    #
    #     # NOTE scale this by loops if < 1?
    #     # calculated the number of timesteps per m in our circle path to maintain the same dist / m
    #     circumference = 2*np.pi*r
    #     timesteps_per_m = self.n_timesteps/circumference
    #
    #     #TODO add option to offset circle start and specify what axis it starts on
    #
    #     # calculate our number of steps based on the distance to the start of the circle
    #     dist_to_circle = np.linalg.norm(start_state[:3] - circle_start_state[:3])
    #     n_timesteps_to_circle = max(int(timesteps_per_m * dist_to_circle), 10)
    #
    #     # plan our position and linear velocity path
    #     gauss_position_planner = GaussPathPlanner(n_timesteps=n_timesteps_to_circle, dt=self.dt)
    #     gauss_position_planner.generate_path(
    #             position=start_state[:3],
    #             target_position=circle_start_state[:3])
    #
    #     # plan our orientation path
    #     # match the profile of our position planner
    #     orientation_planner = path_planners.Orientation(axes=self.axes, n_timesteps=n_timesteps_to_circle)
    #     orientation_planner.match_position_path(
    #         orientation=transform.quaternion_from_euler(
    #             self.orientation_path[0][0],
    #             self.orientation_path[0][1],
    #             self.orientation_path[0][2],
    #             axes=self.axes),
    #         target_orientation=transform.quaternion_from_euler(
    #             circle_start_state[6],
    #             circle_start_state[7],
    #             circle_start_state[8],
    #             axes=self.axes),
    #         position_path=gauss_position_planner.position_path,
    #     )
    #
    #     # update the number of timesteps to step through with the extra path to the circle start
    #     self.position_path = np.vstack((gauss_position_planner.position_path, self.position_path))
    #     self.velocity_path = np.vstack((gauss_position_planner.velocity_path, self.velocity_path))
    #     self.orientation_path = np.vstack((orientation_planner.orientation_path, self.orientation_path))

    def next(self):
        """ Returns the next target from the generated path
        """
        # position = self.position_path[self.n]  # pylint: disable=E0203
        # velocity = self.velocity_path[self.n]  # pylint: disable=E0203
        # orientation = self.orientation_path[self.n] # pylint: disable=E0203
        path = self.path[self.n]
        if self.n_timesteps is not None:
            self.n = min(self.n + 1, self.n_timesteps - 1)
        else:
            self.n += 1
        # some path planner may not end with zero target velocity depending on
        # their parameters this will assure that you have zero target velocity
        # when the filter is positionally at the final target
        if self.n_timesteps is not None:
            if self.n == self.n_timesteps - 1:
                # if we have non-zero target_v, do not zero out
                if self.target_v == 0:
                    path[3:6] = np.zeros(3)
                    path[9:] = np.zeros(3)

        return path



    # def next(self):
    #     """ Returns the next target from the generated path
    #     """
    #     position = self.position_path[self.n]  # pylint: disable=E0203
    #     velocity = self.velocity_path[self.n]  # pylint: disable=E0203
    #     orientation = self.orientation_path[self.n] # pylint: disable=E0203
    #     self.n = min(self.n + 1, self.n_timesteps - 1)
    #     # some path planner may not end with zero target velocity depending on
    #     # their parameters this will assure that you have zero target velocity
    #     # when the filter is positionally at the final target
    #     if self.n == self.n_timesteps - 1:
    #         velocity = np.zeros(3)
    #
    #     return position, velocity, orientation


    def _plot(self, state, target):
        plt.figure()
        plt.subplot(321)
        plt.title('Position')
        plt.plot(self.position_path)

        steps = self.n_timesteps
        plt.scatter(0, state[0], c='b')
        plt.scatter(0, state[1], c='tab:orange')
        plt.scatter(0, state[2], c='g')
        # plt.scatter(steps, target[0], c='b')
        # plt.scatter(steps, target[1], c='tab:orange')
        plt.scatter(steps, target[2], c='g')

        plt.legend(['x', 'y', 'z'])
        plt.subplot(322)
        plt.title('Velocity')
        plt.plot(self.velocity_path)
        norm = []
        for vel in self.velocity_path:
            norm.append(np.linalg.norm(vel))
        plt.plot(norm)
        plt.legend(['dx', 'dy', 'dz', 'norm'])
        ax = plt.subplot(323)
        ax.set_aspect('equal')
        plt.scatter(self.position_path.T[0], self.position_path.T[1])
        plt.scatter(self.position_path.T[0, 0], self.position_path.T[1, 0], color='y', label='path start')
        ax.scatter(state[0], state[1], c='r', label='start loc')
        ax.scatter(self.final_target.T[0], self.final_target.T[1], c='g', label='circle path origin')
        plt.legend()

        plt.subplot(324)
        plt.plot(self.orientation_path)
        plt.legend(['a', 'b', 'g'])
        plt.title('Orientation')

        plt.subplot(325)
        plt.plot(self.ang_velocity_path)
        plt.legend(['da', 'db', 'dg'])
        plt.title('Angular Velocity')


        plt.figure()
        ax = plt.subplot(111, projection='3d')
        plt.plot(self.position_path.T[0], self.position_path.T[1], self.position_path.T[2])
        ax.scatter(xs=state[0], ys=state[1], zs=state[2], c='r', label='start loc')
        ax.scatter(xs=self.final_target.T[0], ys=self.final_target.T[1], zs=self.final_target.T[2], c='g', label='circle path origin')

        ax.scatter(
            xs=self.position_path[0][0],
            ys=self.position_path[0][1],
            zs=self.position_path[0][2],
            c='b',
            s=10
        )
        plt.legend()
        plt.show()

if __name__ == '__main__':
    path = CirclePathPlanner(
            # max_v=2,
            max_v=0.2,
            max_a=0.2,
            dt=0.005,
            # NED=False,
            NED=True,
            )

    path.generate_path(
            # state=np.array([-1, -1, 0, 0,0,0,0,0,0,0,0,0]),
            # target=np.zeros(12),
            # landing
            # state=np.array(
            #     [-2.74000978, -20.90748405, -95.08695221, 0, 0, 0, 0, 0, 1.57, 0, 0, 0]),
            # target=np.array(
            #     [-2.97851569e-04, -3.63362288e+00, -2.74136233e+00, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            # state=np.array(
            #     [ -2.74000978, -20.90748405, -95.08695221, 0, 0, 0, 0, 0, 1.5707, 0, 0, 0]),
            # target=np.array(
            #     [-2.97851569e-04, -3.63362288e+00, -2.74136233e+00, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            # state=np.array(
            #     [ 0.01199463, 5.47384501, -2.3284204, 0, 0, 0, 0, 0, -1.57079995, 0, 0, 0]),
            # target=np.array([-2.97851569e-04, -3.63362288e+00, -8.89256973e+01]),
            # loops=0.5,

            # takeoff
            state=np.array(
                [0.01199463, 5.47384501, -2.3284204, 0, 0, 0, 0, 0, -1.57, 0, 0, 0]),
            target=np.array(
                [-2.97851569e-04, -3.63362288e+00, -8.89256973e+01, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            loops=10,

            start_v=0,
            target_v=0,
            plot=True
            )
