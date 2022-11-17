import sys
import cv2
import numpy as np

from util.video_handling import VideoHandler

from modules.CourtDetector import CourtDetector
from modules.Camera_Estimator.CameraEstimator import CameraEstimator, estimate
from modules.Team_Detector.TeamDetector import TeamDetector, detect
from modules.Team_Detector.PlayerBallDetector import MyFootAndBallDetector
from modules.Team_Detector.PlayerClustering import SegmentationClassifier, ColorHistogramClassifier
from modules.Game_Analytics.GameAnalytics import GameAnalytics

from util import utils


@utils.logging_time
def process_frame(frame):
    masked_court_image, masked_edge_image = court_detector.get_masked_and_edge_court(frame)

    boxes = detect(team_detector, frame, masked_court_image)
    estimated_edge_map = estimate(camera_estimator, masked_edge_image)

    output_frame = frame
    output_frame = cv2.addWeighted(src1=output_frame,
                                   src2=estimated_edge_map,
                                   alpha=.95, beta=.9, gamma=0.)

    player_positions = team_detector.draw_on_field_points(output_frame, boxes)

    output_frame = cv2.addWeighted(src1=output_frame,
                                   src2=player_positions,
                                   alpha=.95, beta=1, gamma=0.)

    analysis.update(camera_estimator.last_estimated_homography, boxes)
    top_view_with_board = analysis.get_analytics()

    return np.concatenate([output_frame, top_view_with_board], axis=1)


if __name__ == '__main__':
    from time import time

    input_videos = [
        # 'issia6.avi'
        # ,
        # 'aris_aek.mp4'qq
        # ,
        'belgium_japan.mp4'
        # ,
        # 'brazil_germany.mp4'
        # ,
        # 'aek_olympiakos.mp4'
    ]

    for i in range(len(input_videos)):
        input_file = input_videos[i]
        input_path_to = f'{utils.get_project_root()}clips/'

        court_detector = CourtDetector(output_resolution=(1280, 720))

        camera_estimator = CameraEstimator(frames_to_track=200, output_resolution=(1280, 720))

        object_detector = MyFootAndBallDetector(dir='mse_10_0.001', frames_to_track=2)
        team_classifier = ColorHistogramClassifier(num_of_teams=3)
        team_classifier.initialize_using_video(
            f'{input_path_to}{input_file}', object_detector, court_detector, training_frames=50)
        team_detector = TeamDetector(object_detector, team_classifier)

        analysis = GameAnalytics()
        analysis.infer_team_sides(f'{input_path_to}{input_file}', court_detector, team_detector,
                                  camera_estimator, training_frames=1)

        output_file = f"{input_file.split('.')[0]}.avi"
        output_path_to = f'{utils.get_project_root()}tasks/results/game_analytics/'

        video_handler = VideoHandler(file=f'{input_path_to}{input_file}',
                                     output_resolution=(1280, 720))
        video_handler.start_recording(output_file=f'{output_path_to}{output_file}', recording_resolution=(1280 + 920, 720))
        time_per_frame = []
        while video_handler.has_frames():
            frame = video_handler.get_frame()
            start = time()
            output = process_frame(frame)
            time_per_frame.append(time() - start)
            video_handler.show_image('Output', output)
        mean_time_per_frame = sum(time_per_frame) / len(time_per_frame)
        print(f'| Elapsed time:{sum(time_per_frame):.1f} '
              f'| FPS:{1 / mean_time_per_frame:.3f} |')
        video_handler.release()

    sys.exit()
