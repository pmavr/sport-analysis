import sys
import cv2

from modules.CourtDetector import CourtDetector
from modules.Team_Detector.TeamDetector import TeamDetector, detect
from modules.Team_Detector.PlayerBallDetector import MyFootAndBallDetector
from modules.Team_Detector.PlayerClustering import ColorHistogramClassifier, DominantColorClassifier, SegmentationClassifier
from util import utils
from util.video_handling import VideoHandler


if __name__ == '__main__':

    court_detector = CourtDetector(output_resolution=(1280, 720))
    object_detector = MyFootAndBallDetector(frames_to_track=2)
    team_classifier = SegmentationClassifier(num_of_teams=3)

    input_file = 'belgium_japan.mp4'
    input_path_to = f'{utils.get_project_root()}clips/'

    team_classifier.initialize_using_video(
        f'{input_path_to}{input_file}', object_detector, court_detector, training_frames=50)

    team_detector = TeamDetector(object_detector, team_classifier)

    output_file = f"{input_file.split('.')[0]}.avi"
    output_path_to = f'{utils.get_project_root()}'
    video_handler = VideoHandler(file=f'{input_path_to}{input_file}',
                                 output_resolution=(1280, 720))
    video_handler.start_recording(output_file=f'{output_path_to}{output_file}', recording_resolution=(1280 + 920, 720))

    while video_handler.has_frames():
        frame = video_handler.get_frame()

        masked_court_image, _ = court_detector.get_masked_and_edge_court(frame)

        boxes = detect(team_detector, frame, masked_court_image)

        output_frame = frame
        # output_frame = team_classifier.draw_boxes(output_frame, boxes)
        output_frame = team_detector.draw_on_field_points(output_frame, boxes)

        output_frame = cv2.addWeighted(src1=frame,
                                       src2=output_frame,
                                       alpha=.95, beta=.9, gamma=0.)

        video_handler.show_image('Detections', output_frame)

    video_handler.release()

    sys.exit()
