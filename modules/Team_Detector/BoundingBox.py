import numpy as np
from collections import defaultdict
import cv2

from util import utils

BALL_BBOX_SIZE = 40


def convert_gt_annotations_to_boxes(annotations):
    player_gt = defaultdict(list)
    ball_gt = defaultdict(list)

    for frame_id in annotations.persons:
        frame_annotations = annotations.persons[frame_id]
        player_boxes = []
        for _, h, w, x, y in frame_annotations:
            player_boxes.append(PlayerBoundingBox(box_coords=(x, y, x + w, y + h), color=(0, 255, 0), score=1.))
        player_gt[frame_id].extend(player_boxes)

    for frame_id in annotations.ball_pos:
        frame_annotations = annotations.ball_pos[frame_id]
        ball_boxes = []
        for x, y in frame_annotations:
            x1 = x - BALL_BBOX_SIZE // 2
            x2 = x1 + BALL_BBOX_SIZE
            y1 = y - BALL_BBOX_SIZE // 2
            y2 = y1 + BALL_BBOX_SIZE
            ball_boxes.append(BallBoundingBox(box_coords=(x1, y1, x2, y2), color=(0, 255, 0), score=1.))
        ball_gt[frame_id].extend(ball_boxes)

    return player_gt, ball_gt


class BoundingBox:
    def __init__(self, image=None, box_coords=(0, 0, 0, 0), color=(0, 0, 0), score=1., frame_id=-1):
        self._calculate_box_values(image, box_coords)
        self.color = color
        self.score = score
        self.frame_id = frame_id
        self.history_size = 5
        self.position_history = []
        # self.history_weights = np.array([.5, .25, .125, .0625, .0625])
        # self.history_weights = np.array([1., .0, .0, .0, .0])

    def update_box_position(self, image, box_coords):
        self._calculate_box_values(image, box_coords)

    def _calculate_box_values(self, image, box_coords):
        x1, y1, x2, y2 = box_coords
        self.x1 = max(0, round(x1))
        self.y1 = max(0, round(y1))
        self.x2 = max(0, round(x2))
        self.y2 = max(0, round(y2))
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.centroid = (
            (self.x2 - self.x1) // 2 + self.x1,
            (self.y2 - self.y1) // 2 + self.y1
        )
        if image is not None:
            self.image = image[self.y1:self.y2, self.x1:self.x2, :]

    def get_field_point(self):
        pass

    def update_position_history(self):
        self.position_history.append(self.get_field_point())


class PlayerBoundingBox(BoundingBox):
    def __init__(self, image=None, box_coords=(0, 0, 0, 0), color=(255, 255, 255), score=0.):
        super().__init__(image, box_coords, color, score)

    def get_field_point(self):
        return self.centroid[0], self.y2

    def draw_box(self, image, text='', line_width=2):
        cv2.rectangle(image, (self.x1, self.y1), (self.x1 + self.w, self.y1+self.h), self.color, line_width)
        cv2.putText(image, text, (self.x1 + 5, self.y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        return image


class BallBoundingBox(BoundingBox):
    def __init__(self, image=None, box_coords=(0, 0, 0, 0), color=(255, 255, 255), score=0.):
        super().__init__(image, box_coords, color, score)

    def get_field_point(self):
        return self.centroid

    def draw_box(self, image, text='', line_width=2):
        cv2.circle(image, self.get_field_point(), 8, self.color, 2)
        cv2.putText(image, text, (self.x1 + 5, self.y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, line_width)
        return image


class SportsBallBoundingBox(BoundingBox):
    def __init__(self, image=None, box_coords=(0, 0, 0, 0), color=(0, 255, 0), score=0.):
        super().__init__(image, box_coords, color, score)

    def get_field_point(self):
        return self.centroid[0], self.y2

    def draw_box(self, image, text='', line_width=2):
        cv2.rectangle(image, (self.x1, self.y1), (self.x1 + self.w, self.y1 + self.h), self.color, line_width)
        cv2.putText(image, text, (self.x1 + 5, self.y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        return image

