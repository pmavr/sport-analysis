import cv2
import torch
import numpy as np

from modules.Team_Detector.BoundingBox import PlayerBoundingBox, BallBoundingBox
from modules.Team_Detector.ObjectTracker import OpticalTracker
from models.foot_and_ball.network.footandball import FootAndBall
from models.yolo.Yolo import YoloV3, YoloV5
from models.foot_and_ball.data import augmentation
from util import utils


class PlayerBallDetector:

    def __init__(self, frames_to_track=0):
        self.model = None
        self.tracker = OpticalTracker
        self.player_trackers = []
        self.ball_trackers = []
        self.frames_to_track = 1 + frames_to_track
        self.tracked_frames = 0

    def detect_objects(self, frame, color=(255, 255, 255)):
        pass

    def track_objects(self, frame):
        player_boxes = [tracker.track_object(frame) for tracker in self.player_trackers]
        ball_boxes = [tracker.track_object(frame) for tracker in self.ball_trackers]
        return player_boxes, ball_boxes

    def detect(self, frame, masked_court_image, color=(255, 255, 255)):
        if self.tracked_frames % self.frames_to_track == 0:

            # Access to GPU
            player_boxes, ball_boxes = self.detect_objects(masked_court_image, color)
            ###

            self.player_trackers = [self.tracker(frame, box) for box in player_boxes]
            self.ball_trackers = [self.tracker(frame, box) for box in ball_boxes]

        else:
            player_boxes, ball_boxes = self.track_objects(frame)

        self.tracked_frames += 1
        return player_boxes, ball_boxes

    @staticmethod
    def draw_boxes(frame, boxes, threshold=0., text=''):
        image = np.copy(frame)
        for b in boxes:
            if b.score >= threshold:
                image = b.draw_box(image, text)
        return image


class FootAndBallDetector(PlayerBallDetector):

    def detect_objects(self, frame, color=(255, 255, 255)):
        img_tensor = augmentation.numpy2tensor(frame)

        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(dim=0).to(self.model.device)
            object_detections = self.model.infer(img_tensor)

        player_boxes = [PlayerBoundingBox(image=frame, box_coords=box, score=score, color=color) for box, score in zip(
            object_detections['players']['boxes'].tolist(),
            object_detections['players']['scores'].tolist()
        )]

        ball_boxes = [BallBoundingBox(image=frame, box_coords=box, score=score, color=color) for box, score in zip(
            object_detections['balls']['boxes'].tolist(),
            object_detections['balls']['scores'].tolist()
        )]
        return player_boxes, ball_boxes


class MyFootAndBallDetector(FootAndBallDetector):

    def __init__(self, player_conf=.7, ball_conf=.7, frames_to_track=0):
        super().__init__(frames_to_track)
        model = FootAndBall.initialize(phase='detect', ball_threshold=ball_conf, player_threshold=player_conf,
                                       max_ball_detections=100, max_player_detections=100)
        self.model, _, _ = FootAndBall.load_model(
            f'{utils.get_footandball_model_path()}/object_detector.pth', model)
        self.model.eval()


class OriginalFootAndBallDetector(FootAndBallDetector):

    def __init__(self, player_conf=.7, ball_conf=.7, frames_to_track=0):
        super().__init__(frames_to_track)
        self.model = FootAndBall.initialize(phase='detect', ball_threshold=ball_conf, player_threshold=player_conf,
                                            max_ball_detections=100, max_player_detections=100)
        model_components = torch.load(f'{utils.get_footandball_model_path()}footandball_original.pth')
        self.model.load_state_dict(model_components)
        self.model.eval()


class YOLOv3Detector(PlayerBallDetector):

    def __init__(self, player_conf=.7, ball_conf=.7, frames_to_track=0):
        super().__init__(frames_to_track)
        self.model = YoloV3()
        self.player_conf = player_conf
        self.ball_conf = ball_conf

    def detect_objects(self, frame, color=(255, 255, 255)):
        output = self.model.predict(frame)
        objects = self.model.extract_objects(output)
        object_detections = self.model.merge_overlapping_boxes(objects)
        player_boxes = []
        ball_boxes = []
        for box, score, class_id, _ in object_detections:
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h
            if class_id == self.model.CLASS_PERSON and score >= self.player_conf:
                bbox = PlayerBoundingBox(image=None, box_coords=(x1, y1, x2, y2), score=score, color=color)
                player_boxes.append(bbox)
            elif class_id == self.model.CLASS_BALL and score >= self.ball_conf:
                bbox = BallBoundingBox(image=None, box_coords=(x1, y1, x2, y2), score=score, color=color)
                ball_boxes.append(bbox)

        return player_boxes, ball_boxes


class YOLOv5Detector(PlayerBallDetector):

    def __init__(self, player_conf=.7, ball_conf=.7, frames_to_track=0):
        super().__init__(frames_to_track)
        self.model = YoloV5()
        self.player_conf = player_conf
        self.ball_conf = ball_conf

    def detect_objects(self, frame, color=(255, 255, 255)):
        output = self.model.predict(frame)
        objects = self.model.extract_objects(output)
        object_detections = self.model.merge_overlapping_boxes(objects)
        player_boxes = []
        ball_boxes = []
        for box, score, class_id, _ in object_detections:
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h
            if class_id == self.model.CLASS_PERSON and score >= self.player_conf:
                bbox = PlayerBoundingBox(image=None, box_coords=(x1, y1, x2, y2), score=score, color=color)
                player_boxes.append(bbox)
            elif class_id == self.model.CLASS_BALL and score >= self.ball_conf:
                bbox = BallBoundingBox(image=None, box_coords=(x1, y1, x2, y2), score=score, color=color)
                ball_boxes.append(bbox)

        return player_boxes, ball_boxes


if __name__ == '__main__':
    import sys
    from util.video_handling import VideoHandler
    from modules.CourtDetector import CourtDetector
    import ObjectTracker

    input_videos = [
        'issia6.avi'
        # ,
        # 'aris_aek.mp4'
        # ,
        # 'belgium_japan.mp4'
        # ,
        # 'chelsea_manchester.mp4',
        # 'france_belgium.mp4'
    ]
    court_detector = CourtDetector(output_resolution=(1280, 720))
    object_detector = YOLOv5Detector(frames_to_track=30)

    for i in range(len(input_videos)):
        input_file = input_videos[i]
        path_to = f'{utils.get_project_root()}clips/'
        video_handler = VideoHandler(file=f'{path_to}{input_file}')

        # output_file = f"{input_file.split('.')[0]}_hd_{model_number}_demo.avi"
        # path_to = f'{utils.get_project_root()}tasks/results/PlayerBallDetector/tmp/'
        # video_handler.start_recording(output_file=f'{path_to}{output_file}')
        # number_of_frames_to_track = 8

        while video_handler.has_frames():

            frame = video_handler.get_frame()

            masked_court_image, _ = court_detector.get_masked_and_edge_court(frame)

            player_boxes, ball_boxes = object_detector.detect(frame, masked_court_image)
            boxes = player_boxes + ball_boxes

            output_frame = np.copy(frame)
            output_frame = object_detector.draw_boxes(output_frame, boxes)
            video_handler.show_image('Detections', output_frame)

        video_handler.release()
    sys.exit()
