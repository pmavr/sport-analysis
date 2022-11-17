import numpy as np
import cv2
import pyflann
import scipy.io as sio
from PIL import Image

import torch
from torchvision.transforms import ToTensor, Compose, Normalize

from models.siamese.Siamese import Siamese
from modules.Camera_Estimator.Camera import Camera, edge_map_from_homography
from modules.Camera_Estimator.CameraObject import CameraObject
from modules.Camera_Estimator.CameraTracker import ECCCameraTracker, OpticalCameraTracker
from util import utils


# @utils.logging_time
def estimate(camera_estimator, detected_edge_map):
    if camera_estimator.tracked_frames % camera_estimator.frames_to_track == 0:
        camera_estimator.last_estimated_homography = camera_estimator.estimate_camera(detected_edge_map)
        camera_estimator.camera_tracker = ECCCameraTracker(camera_estimator.output_resolution)
        # camera_estimator.camera_tracker = OpticalCameraTracker()
    else:
        camera_estimator.last_estimated_homography = camera_estimator.camera_tracker.track_camera(
            detected_edge_map, camera_estimator.last_estimated_homography)
        # camera_estimator.last_estimated_homography = camera_estimator.camera_tracker.track_camera(
        #     rgb_frame, camera_estimator.rgb_frame, camera_estimator.last_estimated_homography)

    camera_estimator.update_homography_history()
    estimated_homography = camera_estimator.smooth_homography()

    estimated_edge_map = edge_map_from_homography(estimated_homography,
                                                  camera_estimator.binary_court,
                                                  camera_estimator.image_resolution)

    estimated_edge_map = cv2.resize(estimated_edge_map, camera_estimator.output_resolution)
    camera_estimator.tracked_frames += 1
    return estimated_edge_map


class CameraEstimator(CameraObject):
    def __init__(self, output_resolution=None, frames_to_track=1800):
        super().__init__(output_resolution)
        model_filename = f'{utils.get_siamese_model_path()}/siamese_10.pth'
        feature_pose_database_filename = f'{utils.get_camera_edge_map_features_path()}feature-camera_pose_database.mat'

        if feature_pose_database_filename:
            data = sio.loadmat(feature_pose_database_filename)
            self.features_vectors = data['features']
            self.camera_poses = data['cameras']

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.0188], std=[0.128])
        ])

        self.model = Siamese()
        Siamese.load_model(model_filename, self.model)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)

        self.model.eval()
        self.rgb_frame = None
        self.last_estimated_homography = None
        self.camera_tracker = None
        self.frames_to_track = 1 + frames_to_track
        self.skip_frames = 0
        self.tracked_frames = 0

        history_size = 7
        self.homography_history = np.zeros((history_size, 3, 3))
        # self.history_weights = np.array([.5, .25, .125, .0625, .0625])
        self.history_weights = np.array([.4, .2992, .16, .064, .0256, .01024, .004096])

    # @utils.logging_time
    def estimate_camera(self, edge_map):
        detected_edge_map = cv2.resize(np.copy(edge_map), self.image_resolution)
        # Access to GPU
        features = self._infer_features_from_edge_map(detected_edge_map)
        ###

        estimated_camera_params = self._infer_camera_from_features(features)

        estimated_homography = Camera(estimated_camera_params).get_homography()

        refined_homography = self._refine_homography(
            source_h=estimated_homography, tar_img=detected_edge_map, eps_thresh=1e-4)

        # return refined_homography
        return estimated_homography

    def _infer_features_from_edge_map(self, edge_map):
        edge_map = cv2.resize(edge_map, (320, 180))
        edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)

        with torch.no_grad():
            x = Image.fromarray(edge_map)
            x = self.transform(x)
            x = torch.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))
            x = x.clone().detach().requires_grad_(True)
            x = x.to(self.device)
            feat = self.model.feature_numpy(x)

        return feat

    def _infer_camera_from_features(self, features_vector):
        flann = pyflann.FLANN()
        result, _ = flann.nn(
            self.features_vectors, features_vector, num_neighbors=1,
            algorithm="kdtree", trees=4, checks=64
        )

        id = result[0]
        camera_pose = self.camera_poses[id]
        return camera_pose

    def update_homography_history(self):
        self.homography_history[1:] = self.homography_history[:-1]
        self.homography_history[0] = self.last_estimated_homography

    def smooth_homography(self):
        return np.average(self.homography_history, axis=0, weights=self.history_weights)




