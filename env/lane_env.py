# env/lane_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class LaneFollowingCircleEnv(gym.Env):
    """
    Basit 2D dairesel yol (circle track) ortamı.
    Araç sabit hızla ilerliyor, sadece direksiyon ile kontrol ediliyor.

    State (4 boyutlu):
      s = [radial_error, heading_error, sin(theta), cos(theta)]

      - radial_error = r - R        (R = ideal yol yarıçapı)
      - heading_error = psi - (theta + pi/2)
          psi   : aracın heading'i
          theta : pozisyon açısı (atan2(y, x))
      - sin(theta), cos(theta): aracın yol üzerindeki konumunu gösterir.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # === TRACK PARAMETRELERİ ===
        self.track_radius = 5.0        # dairesel yolun yarıçapı
        self.lane_half_width = 0.6     # şerit genişliği / 2
        self.max_radial_error = 2.0    # güvenlik limiti (render için vs.)

        # === ARAÇ PARAMETRELERİ ===
        self.speed = 0.07              # biraz daha hızlı
        self.steering_gain = 0.18      # direksiyonun heading üzerindeki etkisi
        self.max_steering = 0.8
        self.dt = 1.0

        # === DURUM ===
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.steering = 0.0

        self.step_count = 0
        self.max_steps = 2000

        # === ACTION SPACE ===
        # 0 = left, 1 = straight, 2 = right
        self.action_space = spaces.Discrete(3)

        # === STATE SPACE ===
        # [radial_error, heading_error, sin(theta), cos(theta)]
        high = np.array(
            [self.max_radial_error, np.pi, 1.0, 1.0],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        # === RENDER ===
        self.fig = None
        self.ax = None

    # ----------------- YARDIMCI FONKSİYONLAR -----------------

    def _wrap_angle(self, angle):
        """Açıyı [-pi, pi] aralığına sar."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _get_state(self):
        # Pozisyondan polar koordinatlar
        r = np.sqrt(self.x ** 2 + self.y ** 2)
        theta = np.arctan2(self.y, self.x)

        # ideal heading: circle tangent → theta + pi/2
        desired_heading = theta + np.pi / 2.0

        radial_error = r - self.track_radius
        heading_error = self._wrap_angle(self.heading - desired_heading)

        state = np.array(
            [radial_error,
             heading_error,
             np.sin(theta),
             np.cos(theta)],
            dtype=np.float32
        )
        return state

    # ----------------- GYM API -----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Aracı yolun üzerinde rastgele bir açıya koy
        theta0 = np.random.uniform(-np.pi, np.pi)

        # yol etrafında küçük offset ile
        r0 = self.track_radius + np.random.uniform(-0.15, 0.15)

        self.x = r0 * np.cos(theta0)
        self.y = r0 * np.sin(theta0)

        # heading: yaklaşık tangent yönü + küçük noise
        desired_heading = theta0 + np.pi / 2.0
        self.heading = desired_heading + np.random.uniform(-0.15, 0.15)

        self.steering = 0.0
        self.step_count = 0

        state = self._get_state()
        info = {}
        return state, info

    def step(self, action):
        self.step_count += 1

        # --- Steering update ---
        if action == 0:          # left
            self.steering -= 0.12
        elif action == 2:        # right
            self.steering += 0.12
        else:                    # straight
            self.steering *= 0.7

        # biraz noise
        self.steering += np.random.normal(0, 0.01)
        self.steering = np.clip(self.steering, -self.max_steering, self.max_steering)

        # --- Heading & position dynamics ---
        self.heading += self.steering_gain * self.steering * self.dt
        self.heading = self._wrap_angle(self.heading)

        self.x += self.speed * np.cos(self.heading)
        self.y += self.speed * np.sin(self.heading)

        # --- State & reward ---
        state = self._get_state()
        radial_error, heading_error, _, _ = state

        # Hataları normalize edelim
        radial_norm = radial_error / self.lane_half_width         # ~ [-1,1] içinde tutmaya çalışıyoruz
        heading_norm = heading_error / np.pi                      # [-1,1]

        # Base reward: şeridin içindeyse +1 civarında olsun
        if abs(radial_error) <= self.lane_half_width:
            base = 1.0
        else:
            base = -1.0  # şeridin dışına taşmaya başladıysa temel ceza

        reward_center = - (radial_norm ** 2)          # uzaklaştıkça kare ceza
        reward_heading = - 0.3 * (heading_norm ** 2)  # yön hatasını da cezalandır
        reward_steering = - 0.01 * (self.steering ** 2)

        reward = base + reward_center + reward_heading + reward_steering

        # --- Termination conditions ---
        # Şeridin dış sınırı: track_radius ± lane_half_width
        out_of_lane = abs(radial_error) > self.lane_half_width
        too_long = self.step_count >= self.max_steps

        terminated = bool(out_of_lane)
        truncated = bool(too_long and not terminated)

        if terminated:
            reward -= 5.0  # şeritten çıkınca ekstra büyük ceza

        return state, reward, terminated, truncated, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()
            plt.show()

        self.ax.clear()

        # Çember yolunu çiz
        angles = np.linspace(0, 2 * np.pi, 200)
        outer = self.track_radius + self.lane_half_width
        inner = self.track_radius - self.lane_half_width

        self.ax.plot(outer * np.cos(angles), outer * np.sin(angles),
                     '--', color='gray')
        self.ax.plot(inner * np.cos(angles), inner * np.sin(angles),
                     '--', color='gray')

        # Aracı çiz
        self.ax.plot(self.x, self.y, 'ro', markersize=10)

        # Heading vektörünü göster
        hx = self.x + 0.8 * np.cos(self.heading)
        hy = self.y + 0.8 * np.sin(self.heading)
        self.ax.arrow(self.x, self.y,
                      hx - self.x, hy - self.y,
                      head_width=0.2, length_includes_head=True, color='r')

        self.ax.set_aspect('equal')
        self.ax.set_xlim(-self.track_radius - 2, self.track_radius + 2)
        self.ax.set_ylim(-self.track_radius - 2, self.track_radius + 2)
        radial_error, heading_error, _, _ = self._get_state()
        self.ax.set_title(
            f"Circle track | r_err={radial_error:.2f}, head_err={heading_error:.2f}"
        )


        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
