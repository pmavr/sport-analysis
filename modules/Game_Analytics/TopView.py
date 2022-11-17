import cv2
import numpy as np

from modules.Game_Analytics.TopViewPoint import TopViewPoint
from modules.Camera_Estimator.Camera import Camera

from util import utils


class TopViewer:
    def __init__(self, court_dimensions=(920, 592)):
        binary_court = utils.get_binary_court()
        self.court_color = (0, 113, 0)
        self.left_side_color = (0, 0, 0)
        self.right_side_color = (0, 0, 0)
        self.court_dimensions = court_dimensions
        court_w = 106
        court_h = 69
        desired_w, desired_h = self.court_dimensions
        self.f_w = court_w / desired_w
        self.f_h = court_h / desired_h

        self.court = self._draw_top_view_court(binary_court, court_w=920, court_h=592)
        self._paint_top_view_court()

    def convert_boxes_to_topview_points(self, homography, boxes):
        return [TopViewPoint(b.get_field_point(), homography, f_w=self.f_w, f_h=self.f_h, color=b.color) for b in boxes]

    def project_on_topview(self, points, ball_history):
        top_view = np.copy(self.court)
        for point in points:
            cv2.circle(top_view, point.coords, 2, point.color, 3)

        for i in range(1, len(ball_history)):
            point = ball_history[-i]
            v = 255 - i
            cv2.circle(top_view, point, 1, (v, v, v), 1)

        top_view = cv2.resize(top_view, self.court_dimensions)
        return top_view

    def _draw_top_view_court(self, binary_court, court_w, court_h):
        court = np.zeros((court_h, court_w, 3), dtype=np.uint8)
        points = binary_court['points']
        line_segment_indexes = binary_court['line_segment_index']
        homography = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        n = line_segment_indexes.shape[0]
        for i in range(n):
            idx1, idx2 = line_segment_indexes[i][0], line_segment_indexes[i][1]
            p1, p2 = points[idx1], points[idx2]
            q1 = Camera.project_point_on_topview(p1, homography, self.f_w, self.f_h)
            q2 = Camera.project_point_on_topview(p2, homography, self.f_w, self.f_h)
            cv2.line(court, q1, q2, color=(255, 255, 255), thickness=3)
        return court

    def _paint_top_view_court(self):
        self.court[self.court[:, :, 1] < 255] = np.array(self.court_color)
        self.court[208:374, :7, :] = np.array(self.left_side_color)
        self.court[208:374, -7:, :] = np.array(self.right_side_color)

    def set_team_side_colors(self, left_side_color, right_side_color):
        self.left_side_color = left_side_color
        self.right_side_color = right_side_color
        self._paint_top_view_court()
