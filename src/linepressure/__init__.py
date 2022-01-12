from typing import Iterable, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as lines
import numpy as np
import tqdm


class PointCollection:
    def __init__(self, points: np.ndarray, wraparound=False):
        self.wraparound = wraparound
        self.points = points
        self.dt = None
        self._d = None

    @classmethod
    def from_circle(cls, center: tuple[int, int], radius: int, n: int = 100):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        x = radius * np.cos(angles) + center[0]
        y = radius * np.sin(angles) + center[1]

        points = np.vstack((x, y))
        return cls(points.transpose(), True)

    @classmethod
    def from_function(cls, points, f):
        points = f(points)
        return cls(points, False)

    def n_points(self) -> int:
        return self.points.shape[0]

    def neighbouring_indexes(self, of: int) -> tuple[Optional[int], Optional[int]]:
        lower = of - 1 if of > 0 else self.n_points() - 1
        higher = of + 1 if of < self.n_points() - 1 else 0

        if not self.wraparound and of == 0:
            lower = None
        elif not self.wraparound and of == self.n_points() - 1:
            print("yo")
            higher = None

        return lower, higher

    def close_to(
        self, point: np.ndarray, radius: float, except_indexes: Iterable[int] = None
    ) -> np.ndarray:
        distances = np.sqrt(
            np.sum((point - self.points) * (point - self.points), axis=1)
        )
        mask = distances < radius

        if except_indexes is not None:
            for i in except_indexes:
                mask[i] = 0
        return self.points[mask]

    def d(self):
        if self._d is not None:
            return self._d
        d = 0
        for i in range(self.n_points() - 1):
            diff = self.points[i] - self.points[i + 1]
            d += np.sqrt(np.sum(diff * diff))
        self._d = d / (self.n_points() - 1)
        return self._d

    def resample(self, k_max, k_min, delta=lambda _: 1):
        i = 0
        while i < self.n_points():
            _, next_i = self.neighbouring_indexes(i)
            if next_i == None:
                print("skipping", next_i, "after", i)
                continue
            p1 = self.points[i]
            p2 = self.points[next_i]
            diff = p1 - p2
            norm = np.sqrt(np.sum(diff * diff))
            d_max = k_max * self.d() * (delta(p1) + delta(p2)) / 2
            d_min = k_min * self.d() * (delta(p1) + delta(p2)) / 2
            if norm > d_max:
                if next_i != 0:
                    self.dt =  np.insert(self.dt, next_i, (self.dt[i] + self.dt[next_i])/2, 0)
                    self.points = np.insert(self.points, next_i, (p1 + p2) / 2, 0)
                else:
                    self.dt = np.vstack((self.dt, (self.dt[i] + self.dt[next_i])/2))
                    self.points = np.vstack((self.points, (p1 + p2) / 2))
                continue
            elif norm < d_min:
                self.dt = np.delete(self.dt, i, axis=0)
                self.points = np.delete(self.points, i, axis=0)
                continue
            i += 1

    def scatter(self):
        return plt.scatter(
            np.concatenate((self.points[:, 0], self.points[:1, 0])),
            np.concatenate((self.points[:, 1], self.points[:1, 1])),
            edgecolors=None,
            facecolor="black",
            alpha=0.6,
            # color='black',
            s=0.2,
        )

    def __iter__(self):
        return self.points.__iter__()

    def indexes_to_update(self):
        if self.wraparound:
            return range(self.n_points())
        else:
            return range(1, self.n_points() - 1)


class LineTensionUpdater:
    def __init__(self, pc: PointCollection, baseline = lambda _: 1.0, stationary_points: np.ndarray = None):
        self.pc = pc
        self.stationary_points = stationary_points

        self.fb = lambda x: baseline(x)*0.5
        self.ff = lambda x: baseline(x)*0.2
        self.fa = lambda x: baseline(x)*1e-2
        self.delta = lambda _: 1.0
        # self.r_1 = 10
        # self.q_ij = 1e-8
        self.k0 = 0.9
        self.k1 = 100

        self.k_max = 4
        self.k_min = 0.25

        self.far1 = 100

    def d(self):
        return self.pc.d()

    def brownian(self, sigma=0.05):
        z = np.random.normal(0, sigma, size=self.pc.points.shape)
        scale = np.apply_along_axis(self.fb, 1, self.pc.points).reshape((-1, 1))
        return scale * z * self.delta(self.pc.points) * self.d()

    def fairing(self, p, p_left, p_right):
        return self.ff(p) * (
            (p_left * self.delta(p_right) + p_right * self.delta(p_left))
            / (self.delta(p_left) + self.delta(p_right))
            - p
        )

    def attraction_repulsion(self, p, i, all_points):
        n_min = 5

        fi = 0
        for j in range(all_points.shape[0]):
            next_index = j + 1
            if next_index >= all_points.shape[0]:
                next_index = 0
            if i == j or next_index == i:
                continue

            xj = self.closest_point_to_line(p, all_points[j], all_points[next_index])
            xj_to_p = p - xj
            if np.all(xj_to_p == 0):
                raise Exception(p, xj)
            diff_norms = np.sqrt(np.sum(xj_to_p * xj_to_p))

            delta_x = self.delta(xj)
            r_1 = self.k1 * self.d()
            mask = (diff_norms < min(self.delta(p), delta_x) * r_1) * (
                max(abs(j - i), abs(j + 1 - i)) > n_min
            )
            fij = (
                (xj_to_p / diff_norms)
                * self.lennard_jones(diff_norms / (self.d() * self.delta(p)))
                * mask
            )
            fi += fij

        return self.fa(p) * fi

    def reject(self, p, i, other_points):
        all_points = other_points
        if self.stationary_points is not None:
            all_points = np.concatenate((all_points, self.stationary_points))
        diff = p - all_points
        normals = np.sqrt(np.sum(diff * diff, axis=1))
        l_i, r_i = self.pc.neighbouring_indexes(i)
        mask = np.ones_like(normals)

        mask[i] = 0
        if l_i is not None:
            mask[l_i] = 0
        if r_i is not None:
            mask[r_i] = 0

        mask *= normals < self.far1

        normals = np.where(mask == 1, normals, 1).reshape((-1, 1))

        forces = diff * (self.far1 / normals - 1) * self.fa(p) * mask.reshape(-1, 1)
        return forces.sum(axis=0)

    def vec_reject(self, points):
        tiled = np.repeat(points[:, :, np.newaxis], len(points) - 1, axis=2)
        for i in range(0, tiled.shape[2]):
            tiled[:, :, i] = np.roll(tiled[:, :, i], -(i+1), axis=0)
        points = points[:, :, np.newaxis]

        diff = points - tiled
        normals = np.sqrt(np.sum(diff*diff, axis=1))

        mask = np.ones_like(normals)
        for i in range(len(points)):
            l_i, r_i = self.pc.neighbouring_indexes(i)

            if l_i is not None:
                mask[i, -1] = 0
            if r_i is not None:
                mask[i, 0] = 0

        mask *= normals < self.far1

        forces = diff * (self.far1 / normals[:, np.newaxis, :] - 1) * self.fa(points) * mask[:, np.newaxis, :]
        return forces.sum(axis=2)


    def lennard_jones(self, r, q_ij=None):
        if q_ij is None:
            q_ij = self.k0
        q_ij_6 = q_ij ** 6
        r_6 = r ** 6
        a = q_ij_6 / r_6
        return a * (a - 1)

    @staticmethod
    def closest_point_to_line(p, a, b):
        a_to_p = p - a
        a_to_b = b - a

        t = a_to_p.dot(a_to_b) / np.sum(a_to_b * a_to_b)

        closest = a + a_to_b * t

        if t < 0:
            closest = a
        elif t > 1:
            closest = b

        return closest

    def update(
        self,
        scatter = None,
        ax = None,
        pbar: tqdm.tqdm = None,
    ):
        if scatter is not None:
            if isinstance(scatter, list) and isinstance(scatter[0], lines.Line2D):
                # plt.plot
                if self.pc.dt is not None:
                    c = np.sqrt(np.sum(self.pc.dt*self.pc.dt, axis=1))
                    c /= np.max(c)
                    scatter[0].set_color([(255, 255, 255, int(255*x)) for x in c])
                scatter[0].set_xdata(np.concatenate((self.pc.points[:, 0], self.pc.points[:1, 0])))
                scatter[0].set_ydata(np.concatenate((self.pc.points[:, 1], self.pc.points[:1, 1])))
            else:
                # plt.scatter
                scatter.set_offsets(self.pc.points)
                # if self.pc.dt is not None:
                #     c = np.sqrt(np.sum(self.pc.dt*self.pc.dt, axis=1))
                #     c /= np.max(c)
                #     scatter.set_facecolor([(255, 255, 255, int(255*x)) for x in c])

        if ax is not None:
            xmin = self.pc.points[:, 0].min()
            xmax = self.pc.points[:, 0].max()
            ymin = self.pc.points[:, 1].min()
            ymax = self.pc.points[:, 1].max()
            x_lim = max(abs(xmin), abs(xmax))
            y_lim = max(abs(ymin), abs(ymax))
            ax.set_xlim(-x_lim-2, x_lim +2)
            ax.set_ylim(-y_lim-2, y_lim +2)

        new_points = self.pc.points.copy()

        b = self.brownian()
        r = self.vec_reject(self.pc.points)
        new_points = b + r

        dt = np.zeros_like(new_points)
        for i in self.pc.indexes_to_update():
            point = self.pc.points[i]
            l_i, r_i = self.pc.neighbouring_indexes(i)
            left, right = self.pc.points[l_i], self.pc.points[r_i]

            f = self.fairing(point, left, right)
            # a = self.attraction_repulsion(point, i, self.pc.points)
            dt[i] = new_points[i] + f # + a
            new_points[i] += point + f # + a

        self.pc.points = new_points
        self.pc.dt = dt

        self.pc.resample(self.k_max, self.k_min)

        if pbar is not None:
            pbar.update(1)
