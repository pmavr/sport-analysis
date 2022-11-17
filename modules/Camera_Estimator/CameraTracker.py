import cv2
import numpy as np

from modules.Camera_Estimator.CameraObject import CameraObject
from modules.Camera_Estimator.Camera import edge_map_from_homography
from util import utils

class CameraTracker(CameraObject):
    def track_camera(self, detected_edge_map, last_estimated_homography):
        pass


class ECCCameraTracker(CameraTracker):
    # @utils.logging_time
    def track_camera(self, edge_map, last_estimated_homography):
        detected_edge_map = cv2.resize(np.copy(edge_map), self.image_resolution)
        refined_homography = self._refine_homography(
            source_h=last_estimated_homography, tar_img=detected_edge_map,
            iterations_num=50)

        return refined_homography


class OpticalCameraTracker(CameraObject):
    def __init__(self):
        super().__init__()
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        margin = 150
        w, h = self.image_resolution
        top_left = (0 + margin, 0 + margin)
        top_right = (w - margin, 0 + margin)
        bot_right = (w - margin, h - margin)
        bot_left = (0 + margin, h - margin)

        self.points = np.array([top_left, top_right, bot_right, bot_left], np.float32)

    def track_camera(self, frame, last_frame, last_estimated_homography):
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

        points, st, err = cv2.calcOpticalFlowPyrLK(
            last_frame, current_frame, self.points, None, **self.lk_params)

        src = np.round(self.points)
        dst = np.round(points)

        homography_update = cv2.getPerspectiveTransform(src, dst)
        refined_homography = homography_update @ last_estimated_homography
        self.points = points

        return refined_homography
