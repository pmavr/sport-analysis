import cv2
import numpy as np

from modules.Camera_Estimator.Camera import edge_map_from_homography
from util import utils


class CameraObject:
    def __init__(self, output_resolution):
        self.binary_court = utils.get_binary_court()
        self.image_resolution = (1280, 720)
        self.output_resolution = output_resolution
        self.refinement_dist_thresh = 15

    def _refine_homography(self, source_h, tar_img, **kwargs):
        src_img = edge_map_from_homography(
            source_h, self.binary_court, self.image_resolution)

        target_distance = self._distance_transform(tar_img)
        source_distance = self._distance_transform(src_img)

        homography_adjustment = self._find_transform(source_distance, target_distance, **kwargs)

        refined_h = homography_adjustment @ source_h
        return refined_h

    def _distance_transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.image_resolution)
        _, binary_im = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)

        dist_im = cv2.distanceTransform(binary_im, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_im[dist_im > self.refinement_dist_thresh] = self.refinement_dist_thresh

        return dist_im

    @staticmethod
    # @utils.logging_time
    def _find_transform(im_src, im_dst, iterations_num=1000, eps_thresh=1e-3):
        warp = np.eye(3, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations_num, eps_thresh)
        try:
            _, warp = cv2.findTransformECC(im_src, im_dst, warp, cv2.MOTION_HOMOGRAPHY, criteria, inputMask=None,
                                           gaussFiltSize=1)
        except:
            print('Warning: find transform failed. Set warp as identity')
        return warp
