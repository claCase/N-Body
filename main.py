import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import gravitational_constant as G
import matplotlib.animation as anim
import argparse


class Planet:
    def __init__(self, position: np.array, force: np.array, mass: float, name=""):
        self.position = position
        self.mass = mass
        self.trajectory = []
        self.name = name
        self.force = force
        self.update_trajectory()

    def update_position(self, f):
        f /= self.mass
        self.force = self.force + f
        ds = 2 * self.force * dt ** 2
        self.position = self.position + ds

    def update_trajectory(self):
        self.trajectory.append(self.position)


class Simulator:
    def __init__(self, planets: [Planet], t: int, dt: float):
        self.planets = planets
        self.t = t
        self.dt = dt
        self.planets_trajectories = []

    @staticmethod
    def calc_force(planet1, planet2):
        const = planet1.mass * planet2.mass  # G
        d = planet2.position - planet1.position
        norm = np.sqrt(d.T.dot(d)) ** 3
        Fij = const * d / norm
        return Fij

    def calc_tot_forces(self, planets):
        n_planets = len(planets)
        f_dim = planets[0].position.shape[0]
        Fs = np.zeros((n_planets, n_planets, f_dim))

        for i, p1 in enumerate(planets):
            for j, p2 in enumerate(planets):
                if i != j:
                    Fij = self.calc_force(p1, p2)
                    Fs[i, j] = Fij
        tot_forces = np.sum(Fs, 1)
        return tot_forces

    def simulate(self):
        for _ in range(self.t):
            tot_forces = self.calc_tot_forces(self.planets)
            for p, f in zip(self.planets, tot_forces):
                p.update_position(f)
                p.update_trajectory()
        return self.planets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform", action="store_true")
    args = parser.parse_args()
    uniform = args.uniform

    n_planets = 50
    t = 300
    dt = 0.2
    planets_position = np.random.uniform(-100, 100, size=(n_planets, 2))
    planets_position = np.concatenate([[[0, 0]], planets_position], 0)
    if uniform:
        planets_mass = np.random.uniform(5, 20, size=(n_planets,))
        initial_forces = np.random.uniform(0, 2, size=(n_planets, 2))
    else:
        planets_mass = np.asarray([5] * n_planets)
        initial_forces = np.zeros((n_planets, 2))
    planets_mass = np.append(150, planets_mass)
    initial_forces = np.concatenate([[[0, 0]], initial_forces], 0)
    names = np.arange(n_planets + 1)
    planets = []
    for i in range(n_planets + 1):
        planets.append(Planet(planets_position[i], initial_forces[i], planets_mass[i], names[i]))

    simulator = Simulator(planets, t, dt)
    planets = simulator.simulate()
    trajectories = [p.trajectory for p in planets]
    trajectories = np.asarray(trajectories)
    trajectories = np.swapaxes(trajectories, 0, 1)  # TxNxd

    max_lim = (200, 200)#np.max(trajectories, (0, 1))
    min_lim = (-200, -200)#np.min(trajectories, (0, 1))


    def update(i):
        plt.cla()
        plt.title(f"Frame {i}")
        for j, p in enumerate(planets):
            plt.scatter(trajectories[i, j, 0], trajectories[i, j, 1], s=np.maximum(p.mass, 50))
            plt.xlim(min_lim[0], max_lim[0])
            plt.ylim(min_lim[1], max_lim[1])


    fig = plt.figure()
    animation = anim.FuncAnimation(fig, update, frames=t, repeat=True, interval=1)
    print("Saving Animation")
    animation.save(
        "./animation2.gif",
        writer="pillow",
    )
    '''for p in planets:
        trj = p.trajectory
        trj = np.asarray(trj)
        plt.scatter(trj[:,0], trj[:,1])'''

    plt.show()