import numpy as np
import cv2

from util import utils


# @utils.logging_time
def detect(team_detector, frame, masked_court_image):
    if team_detector.object_detector.tracked_frames % team_detector.object_detector.frames_to_track == 0:

        # Access to GPU
        player_boxes, ball_boxes = team_detector.object_detector.detect_objects(masked_court_image)
        ###

        if len(player_boxes) > 0:
            player_boxes = team_detector.player_clustering.infer(player_boxes)

        team_detector.object_detector.player_trackers = [team_detector.object_detector.tracker(frame, box) for box in
                                                         player_boxes]
        team_detector.object_detector.ball_trackers = [team_detector.object_detector.tracker(frame, box) for box in
                                                       ball_boxes]

    else:
        player_boxes, ball_boxes = team_detector.object_detector.track_objects(frame)

    team_detector.object_detector.tracked_frames += 1
    return player_boxes + ball_boxes


class TeamDetector:
    def __init__(self, object_detector, player_clustering):
        self.object_detector = object_detector
        self.player_clustering = player_clustering

    @staticmethod
    def get_ball_color_label():
        return 255, 255, 255

    @staticmethod
    def draw_boxes(frame, boxes, line_width=2):
        image = np.copy(frame)

        for b in boxes:
            image = b.draw_box(image, line_width=line_width)
        return image

    @staticmethod
    def draw_on_field_points(frame, boxes):
        image = np.zeros_like(frame)

        for b in boxes:
            cv2.circle(image, b.get_field_point(), 3, b.color, 5)
        return image
