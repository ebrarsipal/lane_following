# env/lane_env_speed.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class LaneFollowingCircleSpeedEnv(gym.Env):
    """
    Dairesel pist + ajan hem direksiyon hem de hızı seçiyor.

    State (5 boyutlu):
      s = [radial_error, heading_error, sin(theta), cos(theta), speed]

      - radial_error = r - R        (R = ideal yol yarıçapı)
      - heading_error = psi - (theta + pi/2)
          psi   : aracın heading'i
          theta : atan2(y, x)
      - sin(theta), cos(theta): pist üzerindeki konum
      - speed: ajan'ın seçtiği anlık hız
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # === TRACK ===
        self.track_radius = 5.0
        self.lane_half_width = 0.6
        self.max_radial_error = 2.0

        # === SPEED PARAMS (agent kontrol ediyor) ===
        self.min_speed = 0.05
        self.max_speed = 0.10
        # 3 seviyeli hız (yavaş, orta, hızlı)
        self.speed_levels = np.array([
            self.min_speed,
            (self.min_speed + self.max_speed) / 2.0,
            self.max_speed
        ], dtype=np.float32)

        # === ARAÇ DURUMU ===
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0     # rad
        self.steering = 0.0
        self.speed = self.min_speed

        self.steering_gain = 0.15
        self.max_steering = 0.8
        self.dt = 1.0

        self.step_count = 0
        self.max_steps = 600

        # === ACTION SPACE ===
        # 9 aksiyon = 3 direksiyon x 3 hız
        # index -> speed_idx, steer_idx:
        # speed_idx = action // 3   ∈ {0,1,2}
        # steer_idx = action % 3    ∈ {0,1,2}
        self.action_space = spaces.Discrete(9)

        # === STATE SPACE ===
        # [radial_error, heading_error, sin(theta), cos(theta), speed]
        high = np.array(
            [self.max_radial_error, np.pi, 1.0, 1.0, self.max_speed],
            dtype=np.float32
        )
        low = np.array(
            [-self.max_radial_error, -np.pi, -1.0, -1.0, self.min_speed],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        # === RENDER ===
        self.fig = None
        self.ax = None

    # ----------------- helper fonksiyonlar -----------------

    def _wrap_angle(self, angle):
        """Açıyı [-pi, pi] aralığına sar."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _get_state(self):
        r = np.sqrt(self.x ** 2 + self.y ** 2)
        theta = np.arctan2(self.y, self.x)

        desired_heading = theta + np.pi / 2.0

        radial_error = r - self.track_radius
        heading_error = self._wrap_angle(self.heading - desired_heading)

        state = np.array(
            [
                radial_error,
                heading_error,
                np.sin(theta),
                np.cos(theta),
                self.speed
            ],
            dtype=np.float32
        )
        return state

    # ----------------- Gym API -----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        theta0 = np.random.uniform(-np.pi, np.pi)
        r0 = self.track_radius + np.random.uniform(-0.2, 0.2)

        self.x = r0 * np.cos(theta0)
        self.y = r0 * np.sin(theta0)

        desired_heading = theta0 + np.pi / 2.0
        self.heading = desired_heading + np.random.uniform(-0.1, 0.1)

        self.steering = 0.0
        # başlangıç hızını orta hız yapalım
        self.speed = self.speed_levels[1]

        self.step_count = 0

        state = self._get_state()
        info = {}
        return state, info

    def step(self, action):
        self.step_count += 1

        # --- action -> (speed level, steering command) ---
        action = int(action)
        speed_idx = action // 3      # 0,1,2
        steer_idx = action % 3       # 0,1,2

        # hız seçim
        self.speed = float(self.speed_levels[speed_idx])

        # direksiyon dinamiği
        if steer_idx == 0:          # left
            self.steering -= 0.1
        elif steer_idx == 2:        # right
            self.steering += 0.1
        else:                       # straight
            self.steering *= 0.8

        self.steering += np.random.normal(0, 0.01)
        self.steering = np.clip(self.steering, -self.max_steering, self.max_steering)

        # heading
        self.heading += self.steering_gain * self.steering * self.dt
        self.heading = self._wrap_angle(self.heading)

        # pozisyon (seçilen hızla)
        self.x += self.speed * np.cos(self.heading)
        self.y += self.speed * np.sin(self.heading)

        # --- state & reward ---
        state = self._get_state()
        radial_error, heading_error, _, _, speed = state

        # merkeze yakın + heading doğru:
        reward_center = -(radial_error ** 2)
        reward_heading = -0.1 * (heading_error ** 2)

        # direksiyon + hız için hafif ceza
        reward_steering = -0.01 * (self.steering ** 2)

        # çok yavaş gitmeye hafif ceza, çok hızlı da biraz ceza
        # (orta hız civarı avantajlı olsun)
        speed_mid = (self.min_speed + self.max_speed) / 2.0
        reward_speed = -5.0 * ((speed - speed_mid) ** 2)

        reward = reward_center + reward_heading + reward_steering + reward_speed

        # --- termination ---
        out_of_lane = abs(radial_error) > (self.max_radial_error * 0.8)
        too_long = self.step_count >= self.max_steps

        terminated = bool(out_of_lane)
        truncated = bool(too_long and not terminated)

        return state, reward, terminated, truncated, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()
            plt.show()

        self.ax.clear()

        # pist çizimi
        angles = np.linspace(0, 2 * np.pi, 200)
        outer = self.track_radius + self.lane_half_width
        inner = self.track_radius - self.lane_half_width

        self.ax.plot(outer * np.cos(angles), outer * np.sin(angles),
                     '--', color='gray')
        self.ax.plot(inner * np.cos(angles), inner * np.sin(angles),
                     '--', color='gray')

        # araç
        self.ax.plot(self.x, self.y, 'bo', markersize=8)

        # heading oku
        hx = self.x + 0.8 * np.cos(self.heading)
        hy = self.y + 0.8 * np.sin(self.heading)
        self.ax.arrow(self.x, self.y,
                      hx - self.x, hy - self.y,
                      head_width=0.2, length_includes_head=True, color='b')

        self.ax.set_aspect('equal')
        self.ax.set_xlim(-self.track_radius - 2, self.track_radius + 2)
        self.ax.set_ylim(-self.track_radius - 2, self.track_radius + 2)
        self.ax.set_title(
            f"Circle+Speed | speed={self.speed:.3f}"
        )

        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
