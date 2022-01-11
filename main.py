from typing import List
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

    def update_position(self, f, dt):
        f /= self.mass
        self.force = self.force + f
        ds = self.force * dt ** 2
        self.position = self.position + ds

    def update_trajectory(self):
        self.trajectory.append(self.position)


class Simulator:
    def __init__(self, planets: List[Planet], t: int, dt: float):
        self.planets = planets
        self.t = t
        self.dt = dt
        self.planets_trajectories = []
        self.e = 1e-2

    def calc_force(self, planet1, planet2):
        const = planet1.mass * planet2.mass  # G
        d = planet2.position - planet1.position
        norm = np.sqrt(d.T.dot(d)) ** 3
        Fij = const * d / (norm + self.e)
        return Fij

    def calc_tot_forces(self):
        n_planets = len(self.planets)
        f_dim = self.planets[0].position.shape[0]
        Fs = np.zeros((n_planets, n_planets, f_dim))

        for i, p1 in enumerate(self.planets):
            for j, p2 in enumerate(self.planets):
                if i != j:
                    Fij = self.calc_force(p1, p2)
                    Fs[i, j] = Fij
        tot_forces = np.sum(Fs, 1)
        return tot_forces

    def simulate(self):
        for _ in range(self.t):
            tot_forces = self.calc_tot_forces()
            for p, f in zip(self.planets, tot_forces):
                p.update_position(f, self.dt)
                p.update_trajectory()
        return self.planets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=str, default="uniform")
    parser.add_argument("--n_planets", type=int, default=10)
    parser.add_argument("--t", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace_len", type=int, default=60)
    parser.add_argument("--skip", type=int, default=10)
    args = parser.parse_args()
    n_planets = args.n_planets
    t = args.t
    dt = args.dt
    initial = args.initial
    save = args.save
    trace = args.trace
    trace_len = args.trace_len
    skip = args.skip

    if initial == "uniform":
        planets_position = np.random.uniform(-100, 100, size=(n_planets, 2))
        planets_mass = np.random.uniform(5, 20, size=(n_planets,))
        initial_forces = np.random.uniform(-10, 10, size=(n_planets, 2))
        planets_mass = np.append(30, planets_mass)
    elif initial == "circle":
        theta = (np.linspace(0, 2 * np.math.pi, n_planets + 1)[:-1]).reshape(-1, 1)
        radious = 30
        x = np.cos(theta) * radious
        y = np.sin(theta) * radious
        force_scale = 25
        dx = -np.sin(theta) * force_scale
        dy = np.cos(theta) * force_scale
        planets_position = np.concatenate([x, y], 1)
        planets_mass = np.tile([10], n_planets)
        initial_forces = np.concatenate([dx, dy], 1)
        planets_mass = np.append(25, planets_mass)
    elif initial == "spiral":
        n_planets = 2
        planets_position = np.asarray([[-20, -20], [-20, 0]])
        planets_mass = np.asarray([10] * n_planets)
        # initial_forces = np.zeros((n_planets, 2))
        initial_forces = np.asarray([[2, 0], [5, 5]])
        planets_mass = np.append(15, planets_mass)

    initial_forces = np.concatenate([[[0, 0]], initial_forces], 0)
    planets_position = np.concatenate([[[0, 0]], planets_position], 0)
    names = np.arange(n_planets + 1)
    planets = []

    for i in range(planets_mass.shape[0]):
        planets.append(Planet(planets_position[i], initial_forces[i], planets_mass[i], names[i]))

    simulator = Simulator(planets, t, dt)
    planets = simulator.simulate()
    trajectories = [p.trajectory for p in planets]
    trajectories = np.asarray(trajectories)
    trajectories = np.swapaxes(trajectories, 0, 1)  # TxNxd

    if initial == "uniform":
        max_lim = (160, 160)
        min_lim = (-160, -160)
    elif initial == "circle":
        max_lim = (40, 40)
        min_lim = (-40, -40)
    else:
        max_lim = (90, 90)
        min_lim = (-40, -40)

    trajectories = trajectories[::skip]
    counter = np.arange(trajectories.shape[0])
    counter = np.minimum(counter, trace_len)


    def update(i, counter):
        plt.cla()
        plt.title(f"Frame {i}")
        for j, p in enumerate(planets):
            plt.scatter(trajectories[i, j, 0], trajectories[i, j, 1], s=np.maximum(p.mass, 50))
            if trace:
                plt.plot(trajectories[i - counter[i]:i, j, 0], trajectories[i - counter[i]:i, j, 1])
            plt.xlim(min_lim[0], max_lim[0])
            plt.ylim(min_lim[1], max_lim[1])


    fig = plt.figure()
    animation = anim.FuncAnimation(fig, update, fargs=(counter,), frames=trajectories.shape[0], repeat=True, interval=1)
    if save:
        print("Saving Animation")
        animation.save(
            "./Animations/animation10.gif",
            writer="pillow",
        )

    plt.show()
