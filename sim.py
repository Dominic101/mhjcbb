import numpy as np
import gtsam  

def fsg_track_map():

    with open('/home/dominic/Downloads/cones_final.txt','r') as fo:
        line = fo.readline()
        cones_full = np.array( [ [float(cp.split(' ')[0]), float(cp.split(' ')[1])] for cp in line.split(',')[2:]])
        #plt.scatter(cones_full[:,0], cones_full[:,1])
        #plt.show()

    with open('/home/dominic/Downloads/cones_incremental.txt','r') as fo:
        cones_incremental = fo.readlines()
        cones_sample = [ np.array([float(cp.split(' ')[0]), float(cp.split(' ')[1])]) for cp in cones_incremental[0].split(',')[2:]]
        car_x = [float(cones_incremental[i].split(',')[0].split(' ')[0]) for i in range(len(cones_incremental)) if len(cones_incremental[i].split(',')[2][0])<=4]
        car_y = [float(cones_incremental[i].split(',')[0].split(' ')[1]) for i in range(len(cones_incremental)) if len(cones_incremental[i].split(',')[2][0])<=4]
    
    return {'cones':cones_full, 'car':[car_x,car_y]}


class Simulator(object):
    def __init__(self, seed=0):
        self.sigma_x = 0.1
        self.sigma_y = 0.1
        self.sigma_theta = 0.05
        self.sigma_bearing = 0.05
        self.sigma_range = 0.1

        self.max_range = 5.0
        self.max_bearing = np.pi / 3.0
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        # Define env and traj here
        self.env = {}  # l -> gtsam.Point2
        self.traj = []  # gtsam.Pose2

    def reset(self):
        self.random_state = np.random.RandomState(self.seed)

    def make_fsg_map(self):
        data_set = fsg_track_map()
        cones_full=data_set['cones']
        i=0
        for x,y in zip(cones_full[:,0], cones_full[:,1]):
            self.env[i] = gtsam.Point2(x,y)
            i+=1

    def car_path(self):
        data_set = fsg_track_map()['car']
        
        x_pos,y_pos=data_set
    
        pose=gtsam.Pose2(x_pos[0], y_pos[0], 0.0)
        for index,data in enumerate(x_pos):
            self.traj.append(pose)
            if index != 0:
                u= data-x_pos[index-1],y_pos[index]-y_pos[index-1],0.0
                pose = pose.compose(gtsam.Pose2(*u))
        


    def random_map(self, size, limit):
        """
        size: num of landmarks
        limit: l, r, b, t
        """
        self.env = {}
        l, r, b, t = limit
        for i in range(size):
            x = self.random_state.uniform(l, r)
            y = self.random_state.uniform(b, t)
            self.env[i] = gtsam.Point2(x, y)

    def step(self):
        """
        return:
          odom: odom between two poses (initial pose is returned for the first step)
          obs: dict of (landmark -> (bearing, range))
        """
        for i in range(len(self.traj)):
            if i == 0:
                odom = gtsam.Pose2()
            else:
                odom = self.traj[i - 1].between(self.traj[i])
            nx = self.random_state.normal(0.0, self.sigma_x)
            ny = self.random_state.normal(0.0, self.sigma_y)
            nt = self.random_state.normal(0.0, self.sigma_theta)
            odom = odom.compose(gtsam.Pose2(nx, ny, nt))

            obs = {}
            for l, point in self.env.items():
                b = self.traj[i].bearing(point).theta()
                r = self.traj[i].range(point)
                b += self.random_state.normal(0.0, self.sigma_bearing)
                r += self.random_state.normal(0.0, self.sigma_range)

                if 0 < r < self.max_range and abs(b) < self.max_bearing:
                    obs[l] = b, r

            if i == 0:
                yield self.traj[0].compose(odom), obs
            else:
                yield odom, obs
