"""
    @title  宇宙機力学2 第4回課題
    @title  ロケットの打ち上げシミュレーション
    @date   2023/10/17
    @brief  3段無誘導重力ターン式ロケット
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import numpy as np


class RocketSpecs:
    ###################
    # 打ち上げロケット諸元
    ###################
    # 機体質量 [kg]
    total_mass = 289e3
    first_stage_mass = total_mass * 3 / 6
    second_stage_mass = total_mass * 2 / 6
    third_stage_mass = total_mass - first_stage_mass - second_stage_mass

    # 推力 [kN]
    first_stage_thrust = 20000
    second_stage_thrust = 5000
    third_stage_thrust = 1000

    # 燃焼時間 [s]
    MECO = 50              # Main Engine Cut Off
    SECO = MECO + 50       # Second Engine Cut Off
    TEIG = SECO + 250      # Third Engine IGnittion
    TECO = TEIG + 200      # Third Engine Cut Off

    def __init__(self):
        pass


class RocketSimulation:
    def __init__(self):
        ###################
        # 物理定数
        ###################
        # 地球半径 [km]
        self.EARTH_RADIUS = 6378
        # 地心重力定数 [km^3/s^2]
        self.mu = 3.986e5

        ###################
        # 解析条件
        ###################
        # 解析時間 [s]
        self.analysis_time = 10000
        # 射角 [rad]
        self.launch_angle = math.radians(80)
        # 射点緯度 [rad]
        self.launch_site_lat = math.radians(90)

    def sim_main(self):
        # [x1_0, x2_0, x3_0, x4_0]
        X = [self.EARTH_RADIUS, 0, self.launch_site_lat, 0]
        spec = RocketSpecs()

        sol = solve_ivp(self.__eom, [0, self.analysis_time], X, max_step=1,
                        args=(spec,))

        return sol.t, sol.y

    def __eom(self, t, X, spec):
        x_1, x_2, x_3, x_4 = X

        # ゼロ割回避のための例外処理
        if t == 0:
            sin_Gamma = math.sin(self.launch_angle)
            cos_Gamma = math.cos(self.launch_angle)
        else:
            sin_Gamma = x_2 / math.sqrt(x_2**2 + (x_1 * x_4)**2)
            cos_Gamma = - x_1 * x_4 / math.sqrt(x_2**2 + (x_1 * x_4)**2)

        thrust = self.__thrust(t, spec)
        mass = self.__mass(t, spec)

        dx_1 = x_2
        dx_2 = x_1 * x_4**2 + thrust / mass * sin_Gamma\
            - self.mu / x_1**2
        dx_3 = x_4
        dx_4 = - 1 / x_1 *\
            (2 * x_2 * x_4 + thrust / mass * cos_Gamma)

        return [dx_1, dx_2, dx_3, dx_4]

    def __thrust(self, t, spec):
        if t <= spec.MECO:
            thrust = spec.first_stage_thrust
        elif t <= spec.SECO:
            thrust = spec.second_stage_thrust
        elif t <= spec.TEIG:
            thrust = 0
        elif t <= spec.TECO:
            thrust = spec.third_stage_thrust
        else:
            thrust = 0

        return thrust

    def __mass(self, t, spec):
        if t <= spec.MECO:
            mass = spec.total_mass
        elif t <= spec.SECO:
            mass = spec.total_mass - spec.first_stage_mass
        else:
            mass = spec.third_stage_mass

        return mass

    def plot_polar_graph(self, t, sol):
        self.fig = plt.figure(tight_layout=True)

        earth_radius = np.full(100, self.EARTH_RADIUS)
        earth_theta = np.linspace(0, 2 * np.pi, 100)

        ax = self.fig.add_subplot(polar=True)
        ax.plot(sol[2, :], sol[0, :], "C1", label="rocket path")
        ax.plot(earth_theta, earth_radius, "C0", label="earth shape")
        ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        ax.grid(color="black", linestyle="dotted")

        plt.show()

    def plot_raw_graph(self, t, sol):
        self.fig = plt.figure(tight_layout=True)

        ax11 = self.fig.add_subplot(2, 1, 1)
        ax11.plot(t, sol[0, :] - self.EARTH_RADIUS, "C0", label=r"$r$")
        ax11.set_xlabel(r"$t$ [s]")
        ax11.set_ylabel(r"$r$ [km]")
        ax11.grid(color="black", linestyle="dotted")

        ax12 = ax11.twinx()
        ax12.set_ylabel(r"$\dot{r}$ [km/s]")
        ax12.plot(t, sol[1, :], "C1", label=r"$\dot{r}$")

        h1, l1 = ax11.get_legend_handles_labels()
        h2, l2 = ax12.get_legend_handles_labels()
        ax11.legend(h1+h2, l1+l2, bbox_to_anchor=(1.1, 1.1), loc='upper left')

        ax21 = self.fig.add_subplot(2, 1, 2)
        ax21.plot(t, np.degrees(sol[2, :]), "C0", label=r"$\theta$")
        ax21.set_xlabel(r"$t$ [s]")
        ax21.set_ylabel(r"$\theta$ [deg]")
        ax21.grid(color="black", linestyle="dotted")

        ax22 = ax21.twinx()
        ax22.set_ylabel(r"$\dot{\theta}$ [deg/s]")
        ax22.plot(t, np.degrees(sol[3, :]), "C1", label=r"$\dot{\theta}$")

        h1, l1 = ax21.get_legend_handles_labels()
        h2, l2 = ax22.get_legend_handles_labels()
        ax22.legend(h1+h2, l1+l2, bbox_to_anchor=(1.1, 1.1), loc='upper left')

        plt.show()

    def plot_cartesian_graph(self, t, sol):
        self.fig = plt.figure(tight_layout=True)

        ax11 = self.fig.add_subplot(3, 1, 1)
        ax11.plot(t, sol[0, :] * np.cos(sol[2, :]), "C0")
        ax11.set_xlabel(r"$t$ [s]")
        ax11.set_ylabel(r"$x$ [km]")
        ax11.grid(color="black", linestyle="dotted")

        ax21 = self.fig.add_subplot(3, 1, 2)
        ax21.plot(t, sol[0, :] * np.sin(sol[2, :]), "C0")
        ax21.set_xlabel(r"$t$ [s]")
        ax21.set_ylabel(r"$y$ [km]")
        ax21.grid(color="black", linestyle="dotted")

        ax31 = self.fig.add_subplot(3, 1, 3)
        ax31.plot(sol[0, :] * np.cos(sol[2, :]),
                  sol[0, :] * np.sin(sol[2, :]),
                  "C0")
        ax31.set_xlabel(r"$x$ [km]")
        ax31.set_ylabel(r"$y$ [km]")
        ax31.grid(color="black", linestyle="dotted")

        plt.show()

    def save_graph(self, filename):
        self.fig.savefig(f"{filename}.png", dpi=300)


if __name__ == "__main__":
    FILE_NAME = "202310221530"

    rocksim = RocketSimulation()

    time_array, sol_array = rocksim.sim_main()

    rocksim.plot_raw_graph(time_array, sol_array)
    # rocksim.save_graph(FILE_NAME + "_raw")
    rocksim.plot_polar_graph(time_array, sol_array)
    # rocksim.save_graph(FILE_NAME + "_polar")
    rocksim.plot_cartesian_graph(time_array, sol_array)
    # rocksim.save_graph(FILE_NAME + "_xy")
