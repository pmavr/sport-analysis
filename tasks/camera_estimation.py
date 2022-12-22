import sys
import cv2

from modules.CourtDetector import CourtDetector
from modules.Camera_Estimator.CameraEstimator import CameraEstimator, estimate

from util import utils
from util.video_handling import VideoHandler


if __name__ == '__main__':
    from time import time

    input_file = 'belgium_japan.mp4'
    input_path_to = f'{utils.get_project_root()}clips/'

    output_file = f"{input_file.split('.')[0]}.avi"
    output_path_to = f'{utils.get_project_root()}'

    video_handler = VideoHandler(file=f'{input_path_to}{input_file}',
                                 output_resolution=(1280, 720))
    video_handler.start_recording(output_file=f'{output_path_to}{output_file}', recording_resolution=(1280 + 920, 720))

    time_per_frame = []

    court_detector = CourtDetector(output_resolution=(1280, 720))
    camera_estimator = CameraEstimator(frames_to_track=200, output_resolution=(1280, 720))

    while video_handler.has_frames():

        frame = video_handler.get_frame()

        _, detected_edge_map = court_detector.get_masked_and_edge_court(frame)
        start = time()
        estimated_edge_map = estimate(camera_estimator, detected_edge_map)
        time_per_frame.append(time() - start)

        output_frame = cv2.addWeighted(src1=frame,
                                       src2=estimated_edge_map,
                                       alpha=.95, beta=.9, gamma=0.)

        video_handler.show_image('Camera Estimate', output_frame)

    mean_time_per_frame = sum(time_per_frame) / len(time_per_frame)
    print(f'| Elapsed time:{sum(time_per_frame):.1f} '
          f'| FPS:{1 / mean_time_per_frame:.3f} |')
    video_handler.release()

    sys.exit()
