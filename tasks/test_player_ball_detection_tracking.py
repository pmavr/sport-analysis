import sys

from models.foot_and_ball.data import issia_utils
from modules.Team_Detector.BoundingBox import convert_gt_annotations_to_boxes
from modules.Team_Detector.PlayerBallDetector import MyFootAndBallDetector, OriginalFootAndBallDetector
from util import utils, video_handling, evaluation_metrics

if __name__ == '__main__':

    camera_settings = [
        {'camera_id': 6, 'res': 'hd'},
        # {'camera_id': 6, 'res': 'fhd'}
        # {'camera_id': 5, 'res': 'hd'},
        # {'camera_id': 5, 'res': 'fhd'},
    ]
    tracking_rates = [0,5,10,15,20,25,30]


    for setting in camera_settings:
        issia_camera_id = setting['camera_id']
        res = setting['res']

        if res == 'fhd':
            video_w, video_h = 1920, 1080
            video_res = (1920, 1080)
        elif res == 'hd':
            video_w, video_h = 1280, 720
            video_res = (1280, 720)
        else:
            break

        object_detectors = [
            {
                'name': 'fab',
                'color': (255, 0, 0)},
            # {
            #     'name': 'mae_10_0.001',
            #     'color': (0, 0, 255)},
            # {
            #     'name': 'diou_8_0.001',
            #     'color': (0, 0, 128)},
            # {
            #     'name': 'diou_8_0.0001',
            #     'color': (0, 0, 128)},
            {
                'name': 'mse_10_0.001',
                'color': (0, 0, 64)},
            # {
            #     'name': 'yolo',
            #     'model': YOLOv3Detector(player_conf=.0, ball_conf=.0, frames_to_track=rate),
            #     'color': (255, 0, 255)},
        ]
        for detector in object_detectors:
            for rate in tracking_rates:
                det_name = detector['name']
                # det_model = detector['model']
                det_color = detector['color']

                # Read ground truth for players and ball
                issia_dataset_path = utils.get_issia_dataset_path()
                issia_gt_annotations = issia_utils.read_issia_ground_truth(issia_camera_id, issia_dataset_path,
                                                                           (video_h, video_w))
                player_gts, ball_gts = convert_gt_annotations_to_boxes(issia_gt_annotations)

                ball_dets = utils.load_pickle_file(f'object_detections/{res}_{det_name}_ball_dets_{issia_camera_id}_t{rate}')
                player_dets = utils.load_pickle_file(f'object_detections/{res}_{det_name}_player_dets_{issia_camera_id}_t{rate}')

                # output = issia_utils.evaluate_ball_detection_results(fab_ball_dets, ball_gts, 3)

                # evaluation_metrics.plot_precision_recall_curve(precisions, recalls)
                player_acc = evaluation_metrics.centroid_metric(player_dets, player_gts)
                ball_acc = evaluation_metrics.centroid_metric(ball_dets, ball_gts)
                print(f'| {det_name} |'
                          f' {video_h}p |'
                          f' Rate: {rate} |'
                        f' Player:{player_acc:.3f} |'
                      f' Ball:{ball_acc:.3f}')
            #     player_ap, player_precs, player_recs = evaluation_metrics.eleven_point_interpolated_PR_curve(
            #         player_dets, player_gts, iou_threshold=.5)
            #     ball_ap, ball_precs, ball_recs = evaluation_metrics.eleven_point_interpolated_PR_curve(
            #         ball_dets, ball_gts, iou_threshold=.5)
            #     print(f'| Model:{det_name} |'
            #           f'Resolution: {video_w}x{video_h} |'
            #           f'Track. Rate: {rate} |'
            #           f'Player AP: {player_ap:.3f} |'
            #           f'Ball AP: {ball_ap:.3f} |'
            #           f'mAP: {(player_ap + ball_ap) / 2:.3f} |')
            print('\n')
    print('Distance: 25')
    sys.exit()
