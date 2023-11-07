"""
    @title      宇宙機力学2 第4回課題 チャレンジ
    @title      ロケットの打ち上げシミュレーション
    @date       2023/11/04
    @brief      L-4-S-5のフライトシミュレーション
    @reference  L-4-T-1, L-4-S-4, L-4-S-5の諸元と飛しょう計画, 秋葉鐐二郎 et al.
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import numpy as np

DEBUG = False
ANALYSIS_TIME = 30000
FILE_NAME = "L-4-S-5_flight_sim"


class RocketSpecs:
    STAGE_NUM = 4

    # 機体質量 [kg]
    mass = np.zeros(STAGE_NUM + 1)
    mass[0] = 1005.0 * 2                          # 補助ブースタ
    mass[1] = 9399.0                              # 1段
    mass[2] = 3417.6                              # 2段
    mass[3] = 943.1                               # 3段
    mass[4] = 111.0                               # 4段

    # 燃料質量 [kg]
    fuel = np.zeros(STAGE_NUM + 1)
    fuel[0] = 624.0 * 2                           # 補助ブースタ
    fuel[1] = 3887.0                              # 1段
    fuel[2] = 1845.0                              # 2段
    fuel[3] = 547.5                               # 3段
    fuel[4] = 87.95                               # 4段

    # 比推力 [sec]
    specific_thrust = np.zeros(STAGE_NUM + 1)
    specific_thrust[0] = 220.0 * 2            # 補助ブースタ
    specific_thrust[1] = 515.0                  # 1段
    specific_thrust[2] = 242.9 * 0.7                   # 2段
    specific_thrust[3] = 249.3 * 0.7                   # 3段
    specific_thrust[4] = 254.0 * 0.7                  # 4段

    # 点火イベント [sec]
    ignition_time = np.zeros(STAGE_NUM + 1)
    ignition_time[0] = 0                          # 補助ブースタ
    ignition_time[1] = 0                          # 1段
    ignition_time[2] = 37.0                       # 2段
    ignition_time[3] = 103.0                      # 3段
    ignition_time[4] = 477.0                      # 4段

    # 燃焼終了イベント [sec]
    burn_out_time = np.zeros(STAGE_NUM + 1)
    burn_out_time[0] = 7.4                        # 補助ブースタ
    burn_out_time[1] = 29.0                       # 1段
    burn_out_time[2] = 75.4                       # 2段
    burn_out_time[3] = 130.0                      # 3段
    burn_out_time[4] = 508.5                      # 4段

    # 分離イベント [sec]
    separation_time = np.zeros(STAGE_NUM + 1)
    separation_time[0] = 8.0                      # 補助ブースタ
    separation_time[1] = 32.0                     # 1段
    separation_time[2] = 100.0                    # 2段
    separation_time[3] = 150.0                    # 3段

    # 燃焼時間 [sec]
    burn_period = np.zeros(STAGE_NUM + 1)
    burn_period = burn_out_time - ignition_time

    # 質量流量 [kg/s]
    mass_flow_rate = np.zeros(STAGE_NUM + 1)
    mass_flow_rate = fuel / burn_period

    # 推力 [kN]
    gravity_constant = 9.81
    thrust = np.zeros(STAGE_NUM + 1)
    thrust = specific_thrust * mass_flow_rate * gravity_constant
    thrust /= 1000

    if DEBUG is True:
        print(thrust)

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
        self.analysis_time = ANALYSIS_TIME
        # 射角 [rad]
        self.launch_angle = math.radians(80)
        # 射点緯度 [rad]
        self.launch_site_lat = math.radians(90)

        if DEBUG is True:
            # 出力用バッファー
            self.time_buf = []
            self.thrust_buf = []
            self.mass_buf = []

    def sim_main(self):
        # [x1_0, x2_0, x3_0, x4_0]
        X = [self.EARTH_RADIUS, 0, self.launch_site_lat, 0]
        spec = RocketSpecs()
        self.stage = 0

        sol = solve_ivp(self.__eom, [0, self.analysis_time], X, max_step=1,
                        args=(spec,))

        if DEBUG is True:
            self.__plot_params_graph(sol.t)

        return sol.t, sol.y

    def __eom(self, t, X, spec):
        x_1, x_2, x_3, x_4 = X

        self.__stage(t, spec)

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

    def __stage(self, t, spec):
        if t > spec.separation_time[self.stage]:
            self.stage += 1
        if self.stage >= 4:
            self.stage = 4

    def __thrust(self, t, spec):
        if self.stage == 0:
            thrust = spec.thrust[0] + spec.thrust[1]
            if t > spec.burn_out_time[0]:
                thrust = spec.thrust[1]
        elif t < spec.burn_out_time[self.stage]:
            thrust = spec.thrust[self.stage]
        else:
            thrust = 0

        if DEBUG is True:
            self.time_buf.append(t)
            self.thrust_buf.append(thrust)

        return thrust

    def __mass(self, t, spec):
        mass = np.sum(spec.mass[self.stage:spec.STAGE_NUM+1])
        if t < spec.burn_out_time[spec.STAGE_NUM]:
            mass -= spec.mass_flow_rate[self.stage] * t -\
                spec.mass_flow_rate[self.stage] *\
                spec.ignition_time[self.stage]

        if DEBUG is True:
            self.mass_buf.append(mass)

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

    # 質量と推力のグラフを表示する（デバッグ用）
    def __plot_params_graph(self, t):
        self.fig = plt.figure(tight_layout=True)

        ax11 = self.fig.add_subplot(2, 1, 1)
        ax11.plot(self.time_buf, self.mass_buf, "C0")
        ax11.set_xlabel(r"$t$ [s]")
        ax11.set_ylabel(r"$m$ [kg]")
        ax11.grid(color="black", linestyle="dotted")

        ax21 = self.fig.add_subplot(2, 1, 2)
        ax21.plot(self.time_buf, self.thrust_buf, "C0")
        ax21.set_xlabel(r"$t$ [s]")
        ax21.set_ylabel(r"$T$ [kN]")
        ax21.grid(color="black", linestyle="dotted")

        self.save_graph("params_log")
        plt.show()

    def save_graph(self, filename):
        self.fig.savefig(f"{filename}.png", dpi=300)


if __name__ == "__main__":
    rocksim = RocketSimulation()

    time_array, sol_array = rocksim.sim_main()

    if DEBUG is False:
        rocksim.plot_raw_graph(time_array, sol_array)
        rocksim.save_graph(FILE_NAME + "_raw")
        rocksim.plot_polar_graph(time_array, sol_array)
        rocksim.save_graph(FILE_NAME + "_polar")
        rocksim.plot_cartesian_graph(time_array, sol_array)
        rocksim.save_graph(FILE_NAME + "_xy")
