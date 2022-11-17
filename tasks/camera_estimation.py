import sys
import cv2

from modules.CourtDetector import CourtDetector
from modules.Camera_Estimator.CameraEstimator import CameraEstimator, estimate

from util import utils
from util.video_handling import VideoHandler


if __name__ == '__main__':
    from time import time

    input_videos = [
        # 'issia3.avi'
        # ,
        # 'aris_aek.mp4'
        # ,
        'belgium_japan.mp4'
        # ,
        # 'chelsea_manchester.mp4',
        # 'france_belgium.mp4'
        # 'pas_oly_master.mp4'
        # 'aek_olympiakos.mp4'
        # 'brazil_germany.mp4'
    ]

    for i in range(len(input_videos)):
        input_file = input_videos[i]
        input_path_to = f'{utils.get_project_root()}clips/'

        output_file = f"{input_file.split('.')[0]}.avi"
        output_path_to = f'{utils.get_project_root()}tasks/results/camera_estimation/'

        video_handler = VideoHandler(file=f'{input_path_to}{input_file}')
        video_handler.start_recording(output_file=f'{output_path_to}{output_file}')

        time_per_frame = []

        court_detector = CourtDetector(output_resolution=video_handler.video_resolution)
        camera_estimator = CameraEstimator(output_resolution=video_handler.video_resolution,
                                           frames_to_track=20)

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
