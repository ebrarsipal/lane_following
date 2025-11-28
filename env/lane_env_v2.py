import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class LaneFollowingTrackEnv(gym.Env):
    """
    3 düz + 2 virajlı basit bir pist.
    Araç pistin merkez çizgisini takip etmeye çalışıyor.

    State (4 boyutlu):
        s = [lateral_error, heading_error, sin(phase), cos(phase)]

        - lateral_error : şerit merkezine göre yan kayma (m)
        - heading_error : aracın yönü - pistin yönü (rad)
        - phase        : pist üzerindeki ilerleme (0..1 arası) → sin/cos ile encode
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # --------- PİST GEOMETRİSİ ---------
        # Merkez çizgisi için nokta listesi (3 düz + 2 viraj gibi)
        # İstersen sonra bu noktaları değiştirip pist şeklini oynayabilirsin.
        self.track_points = np.array([
            [0.0, 0.0],
            [0.0, 4.0],
            [0.0, 8.0],      # 1. düz (yukarı)

            [2.0, 10.0],
            [4.0, 12.0],     # 1. viraj (sağa doğru)

            [8.0, 12.0],
            [12.0, 12.0],    # 2. düz (sağa)

            [14.0, 10.0],
            [16.0, 8.0],     # 2. viraj (aşağı)

            [16.0, 4.0],
            [16.0, 0.0]      # 3. düz (aşağı)
        ], dtype=np.float32)

        # track_points[i] ile track_points[i+1] arasında toplam N-1 segment var
        self.num_segments = self.track_points.shape[0] - 1

        # Şerit yarı genişliği
        self.lane_half_width = 0.7
        self.max_lateral_error = 2.0

        # --------- ARAÇ PARAMETRELERİ ---------
        self.speed = 0.15            # pist üzerinde ilerleme hızı (s boyutunda)
        self.steering_gain = 0.15    # direksiyonun heading_error üzerindeki etkisi
        self.max_steering = 0.8
        self.dt = 1.0

        # --------- DURUM DEĞİŞKENLERİ ---------
        # Pist koordinatında:
        #   s : pist üzerindeki konum (segment index’i gibi)
        #   lateral_error : merkeze göre yan kayma
        #   heading_error : aracın yönü - pist yönü
        self.s = 0.0
        self.lateral_error = 0.0
        self.heading_error = 0.0
        self.steering = 0.0

        self.step_count = 0
        self.max_steps = 600

        # --------- ACTION & STATE SPACE ---------
        # 0 = sol, 1 = düz, 2 = sağ
        self.action_space = spaces.Discrete(3)

        high = np.array(
            [self.max_lateral_error, np.pi, 1.0, 1.0],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        # --------- RENDER ---------
        self.fig = None
        self.ax = None

    # ----------------- YARDIMCI FONKSİYONLAR -----------------

    def _wrap_angle(self, angle):
        """Açıyı [-pi, pi] aralığına sar."""
        return (angle +
