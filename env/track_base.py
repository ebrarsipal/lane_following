import numpy as np

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)

class Track:
    def __init__(self):
        # Basit bir pist (ileride değiştireceğiz)
        self.points = np.array([
            [0, 0],
            [5, 0],
            [10, 0],
            [15, 2],
            [20, 4],
            [25, 4],
            [30, 4],
            [35, 2],
            [40, 0],
            [45, 0],
        ], dtype=np.float32)

        self.n = len(self.points)

        # Tangent & normal vectors
        self.tangents = []
        self.normals = []

        self._compute_vectors()

    def _compute_vectors(self):
        for i in range(self.n - 1):
            p0 = self.points[i]
            p1 = self.points[i + 1]

            t = normalize(p1 - p0)
            n = np.array([-t[1], t[0]])  # rotate +90°

            self.tangents.append(t)
            self.normals.append(n)

        # Son nokta → büyümesin diye tekrar kullan
        self.tangents.append(self.tangents[-1])
        self.normals.append(self.normals[-1])

    def get_reference(self, idx):
        """waypoint index'e göre pist bilgisi döner."""
        idx = int(np.clip(idx, 0, self.n - 1))
        return self.points[idx], self.tangents[idx], self.normals[idx]
