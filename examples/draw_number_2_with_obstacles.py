
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import pydmps
import pydmps.dmp_discrete


beta = 20.0 / np.pi
gamma = 200
R_halfpi = np.array(
    [
        [np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
        [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)],
    ]
)

ob1 = np.array([0.37,0.60])
ob2 = np.array([0.35,-0.65])
ob3 = np.array([0.62,0.0])
obstacles = np.vstack([ob1,ob2,ob3])


def avoid_obstacles(y, dy, goal):
    p = np.zeros(2)

    for obstacle in obstacles:
        # based on (Hoffmann, 2009)

        # if we're moving
        if np.linalg.norm(dy) > 1e-5:

            # get the angle we're heading in
            phi_dy = -np.arctan2(dy[1], dy[0])
            R_dy = np.array(
                [[np.cos(phi_dy), -np.sin(phi_dy)], [np.sin(phi_dy), np.cos(phi_dy)]]
            )
            # calculate vector to object relative to body
            obj_vec = obstacle - y
            # rotate it by the direction we're going
            obj_vec = np.dot(R_dy, obj_vec)
            # calculate the angle of obj relative to the direction we're going
            phi = np.arctan2(obj_vec[1], obj_vec[0])

            dphi = gamma * phi * np.exp(-beta * abs(phi))
            R = np.dot(R_halfpi, np.outer((obstacle - y)*5, dy))
            pval = -np.nan_to_num(np.dot(R, dy) * dphi)

            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
                pval = 0

            p += pval
    return p




y_des = np.load("2.npz")["arr_0"].T
y_des -= y_des[:, 0][:, None]

# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 25.0)

dmp.imitate_path(y_des=y_des)

plt.figure(1, figsize=(6, 6))

y_track, dy_track, ddy_track = dmp.rollout()
plt.plot(y_track[:, 0], y_track[:, 1], "b--", lw=2, alpha=0.5)

# run while moving the target up and to the right
y_track = []
dmp.reset_state()


for t in range(dmp.timesteps):
    y, _, _ = dmp.step(
        external_force=avoid_obstacles(dmp.y, dmp.dy, dmp.goal)
        )
    y_track.append(np.copy(y))
y_track = np.array(y_track)

plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2)
plt.title("DMP system - draw number 2")



plt.axis("equal")
plt.legend(["original path", "Path with obstacles"])


for obstacle in obstacles:
        (plot_obs,) = plt.plot(obstacle[0], obstacle[1], "rx", mew=3)

plt.plot(dmp.goal[0], dmp.goal[1], "bo", lw=2, alpha=0.5)


plt.show()
