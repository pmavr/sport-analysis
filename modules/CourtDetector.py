import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

from models.two_pix2pix.network.two_pix2pix_model import TwoPix2PixModel
from models.two_pix2pix.test_options import opt
from util import utils, image_process


class CourtDetector:

    def __init__(self, output_resolution):
        self.opt = opt()
        self.model = TwoPix2PixModel()
        self.model.initialize(self.opt)
        self.transform = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.resolution = output_resolution

    # @utils.logging_time
    def get_masked_and_edge_court(self, frame):
        frame = cv2.resize(frame, self.resolution)
        self._infer(frame)
        mask = self._get_grass_mask(size=self.resolution)

        masked_court_image = self._apply_grass_mask(frame, mask)

        edge_map = self._get_edge_map(size=self.resolution)
        masked_edge_image = self._apply_grass_mask(edge_map, mask)
        return masked_court_image, masked_edge_image

    def _infer(self, image):
        img = Image.fromarray(image)
        img = img.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        img_tensor = self.transform(img)
        img_tensor = torch.stack([img_tensor])
        self.model.infer(img_tensor)

    def _get_court_image(self):
        return utils.tensor2im(self.model.real_A.data)

    def _get_edge_map(self, size=(256, 256)):
        edge_map = utils.tensor2im(self.model.fake_D.data)
        edge_map = cv2.resize(edge_map, size)
        return edge_map

    def _get_grass_mask(self, size=(256, 256)):
        mask = utils.tensor2im(self.model.fake_B.data)
        mask = self._refine_mask(mask)
        mask = cv2.resize(mask, size)
        return mask

    @staticmethod
    def _apply_grass_mask(image, grass_mask):
        return image_process.apply_binary_mask(image, grass_mask)

    @staticmethod
    def _refine_mask(mask):
        """Refine mask from white/black patches"""

        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray_mask, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cmax = max(contours, key=cv2.contourArea)

        width, height = gray_mask.shape
        refined_mask = np.zeros([width, height, 3], dtype=np.uint8)
        cv2.fillPoly(refined_mask, pts=[cmax], color=(255, 255, 255))
        blurred_refined_mask = cv2.GaussianBlur(refined_mask, (21, 21), 0)
        return blurred_refined_mask


if __name__ == '__main__':
    import sys
    import glob

    image_filelist = glob.glob(f"{utils.get_project_root()}datasets/test/court_images/*.jpg")

    estimator = CourtDetector()

    for image_filename in image_filelist:
        print(f'Process image... {image_filename}')
        court_image = cv2.imread(image_filename)

        estimator._infer(court_image)
        output_image = utils.concatenate_multiple_images(
            estimator._get_court_image(),
            estimator._apply_grass_mask(estimator._get_court_image(), estimator._get_grass_mask()),
            estimator._get_edge_map()
            )
        utils.save_image(output_image, f"{utils.get_project_root()}tasks/results/CourtDetector/{image_filename.split('/')[-1]}")

    print(
        f'Results from test images have been saved at directory: {utils.get_project_root()}tasks/results/CourtDetector/')
    sys.exit()
