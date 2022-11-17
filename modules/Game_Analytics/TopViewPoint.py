import numpy as np

from util import utils


class TopViewPoint:

    def __init__(self, point, h, f_w=1, f_h=1, color=(0, 0, 0)):
        self.coords = self._convert_perspective_view(point, h, f_w, f_h)
        self.color = color
        self.label = 0

    @staticmethod
    def _convert_perspective_view(point, h, f_w=1, f_h=1):
        x, y = point
        w = 1.0
        p = np.zeros(3)
        p[0], p[1], p[2] = x, y, w

        # flip template in y direction
        m1 = np.asarray([[1, 0, 0],
                         [0, -1, 68],
                         [0, 0, 1]])
        scale = np.asarray([[f_w, 0, 0],
                            [0, f_h, 0],
                            [0, 0, 1]])
        homography_matrix = h @ m1
        homography_matrix = homography_matrix @ scale
        inverted_homography_matrix = np.linalg.inv(homography_matrix)
        q = inverted_homography_matrix @ p

        assert q[2] != 0.0
        projected_x = np.rint(q[0] / q[2]).astype(np.int)
        projected_y = np.rint(q[1] / q[2]).astype(np.int)
        return projected_x, projected_y
