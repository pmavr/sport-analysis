# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

from torchvision import transforms
from PIL import Image
import numpy as np
import numbers
import random
import cv2

import torchvision.transforms.functional as F

from util import utils

# Labels starting from 0
BALL_LABEL = 1
PLAYER_LABEL = 2

# Size of the ball bbox in pixels (fixed as we detect only ball center)
BALL_BBOX_SIZE = 40

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Tensor to numpy transform

denormalize_trans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1.0 / e for e in NORMALIZATION_STD]),
                                        transforms.Normalize(mean=[-e for e in NORMALIZATION_MEAN], std=[1., 1., 1.])])

normalize_trans = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)])


def tensor2image(image, downscale=None):
    # Convert image encoded as normalized tensor back to numpy (opencv format)
    img = denormalize_trans(image)
    temp = img.permute(1, 2, 0).cpu().numpy()
    # Convert from RBG to BGR (OpenCV default color space) and return as the continuous array
    temp = np.ascontiguousarray(temp[:, :, (2, 1, 0)])
    if downscale is not None:
        temp = cv2.resize(temp, (temp.shape[1] // downscale, temp.shape[0] // downscale))
    return temp


def heatmap2image(tensor, channel=1):
    # Convert 1-channel (or more) heatmap/confidence map to numpy image
    # tensor: (h, w) or (h, w, n_channels) tensor
    # channel: channel to show/convert to image (used if there's more than one channel)

    assert tensor.dim() == 2 or tensor.dim() == 3

    if tensor.dim() == 3:
        tensor = tensor[:, :, channel]

    image = tensor.cpu().numpy().astype(np.uint8)*255
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return heatmap


def image2tensor(image):
    # Convert PIL Image to the tensor (with normalization)
    return normalize_trans(image)


def numpy2tensor(image):
    # Convert OpenCV image to tensor (with normalization)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    return image2tensor(pil_image)


def apply_transform_and_clip(boxes, labels, M, shape):
    """

    :param points:
    :param M: affine transformation matrix
    :param shape: (width, height) tuple
    :return:
    """
    assert len(boxes) == len(labels)

    ones = np.ones((len(boxes), 1))
    ext_pts1 = np.append(boxes[:, :2], ones, 1).transpose()     # Upper right corner
    ext_pts2 = np.append(boxes[:, 2:4], ones, 1).transpose()    # Lower left corner

    transformed_pts1 = np.dot(M[:2], ext_pts1).transpose()
    transformed_pts2 = np.dot(M[:2], ext_pts2).transpose()
    # We need to find out which corner is top right and which is bottom left, after the transform
    transformed_boxes = np.zeros_like(boxes)
    transformed_boxes[:, 0] = np.minimum(transformed_pts1[:, 0], transformed_pts2[:, 0])
    transformed_boxes[:, 1] = np.minimum(transformed_pts1[:, 1], transformed_pts2[:, 1])
    transformed_boxes[:, 2] = np.maximum(transformed_pts1[:, 0], transformed_pts2[:, 0])
    transformed_boxes[:, 3] = np.maximum(transformed_pts1[:, 1], transformed_pts2[:, 1])

    assert boxes.shape == transformed_boxes.shape
    return clip(transformed_boxes, labels, shape)


def clip(boxes, labels, shape):
    """

    :param boxes: list of (x1, y1, x2, y2) coordinates
    :param shape: (width, height) tuple
    :return:
    """
    box_contained = lambda e: 0 <= e[0] < shape[0] and 0 <= e[1] < shape[1] and 0 <= e[2] < shape[0] and 0 <= e[3] < shape[1]
    mask = [box_contained(box) for box in boxes]
    return boxes[mask], labels[mask]


class ColorJitter(object):
    """
    Randomly change brightness, contrast, saturation & hue within given ranges
    """

    def __init__(self, brightness_range=(1., 1.), contrast_range=(1., 1.), saturation_range=(1., 1.), hue_range=(0., 0.)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def __call__(self, sample):
        image, boxes, labels = sample

        brightness = round(random.uniform(self.brightness_range[0], self.brightness_range[1]), 3)
        hue = round(random.uniform(self.hue_range[0], self.hue_range[1]), 3)
        contrast = round(random.uniform(self.contrast_range[0], self.contrast_range[1]), 3)
        saturation = round(random.uniform(self.saturation_range[0], self.saturation_range[1]), 3)

        transformed_image = F.adjust_brightness(image, brightness)
        transformed_image = F.adjust_hue(transformed_image, hue)
        transformed_image = F.adjust_contrast(transformed_image, contrast)
        transformed_image = F.adjust_saturation(transformed_image, saturation)

        return transformed_image, boxes, labels


class RandomAffine:
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, p_hflip=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = (0, 0)

        self.p_hflip = p_hflip  # Horizontal mirror probability

    def get_params(self, h, w):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if self.translate is not None:
            max_dx = self.translate[0] * w
            max_dy = self.translate[1] * h
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0

        if self.shear is not None:
            if len(self.shear) == 2:
                shear = [random.uniform(self.shear[0], self.shear[1]), 0.]
            elif len(self.shear) == 4:
                shear = [random.uniform(self.shear[0], self.shear[1]), random.uniform(self.shear[2], self.shear[3])]
            else:
                assert NotImplementedError('Incorrect shear: {}'.format(self.shear))
        else:
            shear = [0.0]

        return angle, translations, scale, shear

    def __call__(self, sample):
        image, boxes, labels = sample
        height = image.height
        width = image.width
        angle, translate, scale, shear = self.get_params(height, width)

        center = (width * 0.5 + 0.5, height * 0.5 + 0.5)
        coeffs = F._get_inverse_affine_matrix(center, angle, translate, scale, shear)
        inverse_affine_matrix = np.eye(3)
        inverse_affine_matrix[:2] = np.array(coeffs).reshape(2, 3)     # Fill-in first 2 rows of an affine transformation matrix

        if np.random.rand() < self.p_hflip:
            # Post-apply horizontal flip
            # Pre-multiply by [ [-1, 0, width], [0, 1, 0], [0, 0, 1] ] matrix
            flip_matrix = np.eye(3)
            flip_matrix[0, 0] = -1
            flip_matrix[0, 2] = width-1
            # For inverse affine matrix, pre-multiply by a inverse flip matrix (which is the same as a flip matrix)
            inverse_affine_matrix = flip_matrix @ inverse_affine_matrix

        image = image.transform((width, height), Image.AFFINE, inverse_affine_matrix[:2].reshape(6), Image.BILINEAR)

        # Compute affine transform matrix and apply it to keypoints
        affine_matrix = np.linalg.pinv(inverse_affine_matrix)
        boxes, labels = apply_transform_and_clip(boxes, labels, affine_matrix, (width, height))

        return image, boxes, labels


class ToRGB:
    """
    Convert the given PIL Image from BGR to RGB.
    """

    def __call__(self, sample):
        image, boxes, labels = sample
        b, g, r = image.split()
        transformed_image = Image.merge("RGB", (r, g, b))
        return transformed_image, boxes, labels


class CenterCrop:
    def __init__(self, size):
        self.out_h, self.out_w = size

    def get_params(self, h, w):
        if w == self.out_w and h == self.out_h:
            return 0, 0
        i = (h - self.out_h) // 2
        j = (w - self.out_w) // 2
        return i, j

    def __call__(self, sample):
        image, boxes, labels = sample
        i, j = self.get_params(image.height, image.width)
        image = F.crop(image, i, j, self.out_h, self.out_w)
        boxes[:, :2] -= (j, i)
        boxes[:, 2:4] -= (j, i)
        boxes, labels = clip(boxes, labels, (self.out_w, self.out_h))
        return image, boxes, labels


class RandomScale:
    """
    Scale down the given PIL Image using a random factor within range (0, 1).

    Args:
        max_shrink_factor: Maximum allowed scale down for the image
        min_shrink_factor: Minimum allowed scale down for the image
    """

    def __init__(self, max_shrink_factor=.6, min_shrink_factor=1.):
        assert 0 < max_shrink_factor <= min_shrink_factor <= 1.
        self.max_shrink_factor = max_shrink_factor
        self.min_shrink_factor = min_shrink_factor

    def __call__(self, sample):
        image, boxes, labels = sample
        f = round(random.uniform(self.max_shrink_factor, self.min_shrink_factor), 3)

        im_shape = image.size
        resized_shape = (round(im_shape[1] * f), round(im_shape[0] * f))
        resized_im = transforms.Resize(resized_shape)(image)
        resized_shape = resized_im.size

        y_diff = im_shape[0] - resized_shape[0]
        x_diff = im_shape[1] - resized_shape[1]

        pad_x_left = int(np.around(y_diff / 2))
        pad_x_right = int(y_diff - pad_x_left)
        pad_y_top = int(np.around(x_diff / 2))
        pad_y_bottom = int(x_diff - pad_y_top)

        resized_im = F.pad(resized_im, [pad_x_left, pad_y_top, pad_x_right, pad_y_bottom])

        resized_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b
            resized_b = [
                round(x1 * f + pad_x_left),
                round(y1 * f + pad_y_top),
                round(x2 * f + pad_x_left),
                round(y2 * f + pad_y_top)]
            resized_boxes.append(resized_b)

        return resized_im, resized_boxes, labels


class ToTensorAndNormalize(object):
    # Convert image to tensors and normalize the image, ground truth is not changed
    def __init__(self):
        self.image_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)])

    def __call__(self, sample):
        # numpy image: H x W x C
        # torch image: C X H X W
        image, boxes, classes = sample
        return self.image_transforms(image), boxes, classes


class TrainAugmentation(object):
    def __init__(self, size):
        self.size = size
        self.augment = transforms.Compose([
            ToRGB(),
            ColorJitter(
                brightness_range=(1., 2.),
                contrast_range=(.6, 2.),
                saturation_range=(.5, 1.8),
                hue_range=(-.25, .14)
            ),
            RandomAffine(degrees=5, scale=(0.8, 1.2), p_hflip=0.5),
            RandomScale(max_shrink_factor=.7, min_shrink_factor=1.),
            ToTensorAndNormalize()
        ])

    def __call__(self, sample):
        return self.augment(sample)


class NoAugmentation(object):
    def __init__(self, size):
        self.size = size
        self.augment = transforms.Compose([
            CenterCrop(self.size),
            ToTensorAndNormalize()
        ])

    def __call__(self, sample):
        return self.augment(sample)
