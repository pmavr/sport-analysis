
from copy import deepcopy
import numpy as np
import cv2

from util import utils


class ObjectTracker:

    def track_object(self, frame):
        pass


class CVTracker(ObjectTracker):

    def __init__(self, frame, box, init_tracker_method):
        self.tracker = init_tracker_method()
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        box = [x1, y1, w, h]
        box = [int(v) for v in box]
        self.tracker.init(frame, box)

    def track_object(self, frame):
        success, box = self.tracker.update(frame)
        (x, y, w, h) = [int(v) for v in box]
        return [x, y, x + w, y + h]


class OpticalTracker(ObjectTracker):
    def __init__(self, frame, box):
        self.lk_params = dict(winSize=(50, 50),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        p0 = np.array([box.centroid])
        self.midpoint = np.mean(p0, axis=0).reshape(-1, 1, 2).astype(np.float32)
        self.initial_box = box
        self.initial_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def track_object(self, frame):
        current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        point, st, err = cv2.calcOpticalFlowPyrLK(
            self.initial_gray_frame, current_gray_frame, self.midpoint, None, **self.lk_params)
        self.midpoint = point

        mid_x, mid_y = self.midpoint[0, 0, :]
        box_coords = (
            max(0, int(mid_x - self.initial_box.w // 2)),
            max(0, int(mid_y - self.initial_box.h // 2)),
            max(0, int(mid_x + self.initial_box.w // 2)),
            max(0, int(mid_y + self.initial_box.h // 2))
        )
        self.initial_box.update_box_position(frame, box_coords)
        self.initial_gray_frame = current_gray_frame
        return deepcopy(self.initial_box)
