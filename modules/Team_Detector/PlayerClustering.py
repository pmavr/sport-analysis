import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from util import utils, image_process
from util.video_handling import get_frames_from_video


class KmeansTeamClassifier:

    def __init__(self, num_of_teams):
        self.num_of_teams = num_of_teams
        self.classifier = None

        assert num_of_teams <= 5

        self.colors = [
            # white color reserved for ball
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (0, 255, 255),
            (255, 0, 255)
        ]

    def initialize_using_video(self, video_file, object_detector, court_detector, training_frames=10):
        image_data = get_frames_from_video(video_file, training_frames)
        samples = self.extract_training_samples(image_data, object_detector, court_detector)

        color_training_samples = self.extract_color_features(samples)
        self.train(color_training_samples)

    @staticmethod
    def extract_training_samples(image_data, object_detector, court_detector):
        output_samples = []
        for frame in image_data:
            masked_court_image, _ = court_detector.get_masked_and_edge_court(frame)
            player_boxes, _ = object_detector.detect_objects(masked_court_image)
            player_boxes = [b.image for b in player_boxes]
            output_samples.extend(player_boxes)

        return output_samples

    def train(self, samples):
        self.classifier = KMeans(n_clusters=self.num_of_teams)
        self.classifier.fit(samples)

    def extract_color_features(self, samples):
        return filter(lambda x: x.size != 0, samples)

    def infer(self, boxes):
        samples = [box.image for box in boxes]
        color_features = self.extract_color_features(samples)

        labels = self.classifier.predict(color_features)

        output = []
        for box, label in zip(boxes, labels):
            box.color = self.colors[label]
            output.append(box)

        return output


class ColorHistogramClassifier(KmeansTeamClassifier):

    def __init__(self, num_of_teams=2, model='hsv', bins=(36, 32, 32)):
        assert model in ['hsv', 'bgr']
        super().__init__(num_of_teams)

        if model == 'hsv':
            self.color_method = image_process.get_hsv_histogram
        elif model == 'bgr':
            self.color_method = image_process.get_bgr_histogram
        self.bins = bins

    def extract_color_features(self, samples):
        samples = super(ColorHistogramClassifier, self).extract_color_features(samples)
        processed_samples = [image_process.remove_green_pixels(image) for image in samples]
        histograms = np.array([image_process.get_hsv_histogram(image, self.bins) for image in processed_samples])
        histograms = normalize(histograms, axis=1)
        return histograms


class DominantColorClassifier(KmeansTeamClassifier):

    def __init__(self, num_of_teams=2, num_of_dominant_colors=2):
        super().__init__(num_of_teams)
        self.num_of_dominant_colors = num_of_dominant_colors

    def extract_color_features(self, samples):
        samples = super(DominantColorClassifier, self).extract_color_features(samples)
        processed_samples = [image_process.remove_green_pixels(image) for image in samples]
        processed_samples = [image_process.remove_black_pixels(image) for image in processed_samples]

        processed_samples = [image_process.bgr_to_hsv_color(img) for img in processed_samples]

        dominant_color_pairs = [self._find_dominant_color_pair(image) for image in processed_samples]

        dominant_colors = [self._get_dominant_from(pair) for pair in dominant_color_pairs]

        return np.array(dominant_colors)

    def _get_dominant_from(self, pair):
        color_pair = []
        hist = np.histogram(pair['labels'], bins=self.num_of_dominant_colors)
        for i in range(self.num_of_dominant_colors):
            color = {'color': pair['dominant_colors'][i], 'frequency': hist[0][i]}
            color_pair.append(color)

        dominant_color = max(color_pair, key=lambda i: i['frequency'])
        return dominant_color['color']

    def _find_dominant_color_pair(self, pixels):
        if pixels.size == 0:
            pixels = np.random.randint(0, 255, size=(self.num_of_dominant_colors, 3))
        color_cluster = KMeans(n_clusters=self.num_of_dominant_colors)
        color_cluster.fit(pixels)

        dominant_colors = color_cluster.cluster_centers_.astype(int)
        labels = color_cluster.labels_

        return {'dominant_colors': dominant_colors, 'labels': labels}


class SegmentationClassifier(KmeansTeamClassifier):

    def __init__(self, num_of_teams, segm_k=3):
        super().__init__(num_of_teams)
        self.segm_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.segm_k = segm_k
        self.segm_attempts = 10

    def extract_color_features(self, samples):
        samples = super(SegmentationClassifier, self).extract_color_features(samples)
        image_foregrounds = [self.get_foreground(image) for image in samples]
        histograms = np.array([image_process.get_hsv_histogram(image) for image in image_foregrounds])
        histograms = normalize(histograms, axis=1)
        return histograms

    def get_foreground(self, sample):
        segments = self.segment_image(sample)
        # dominant_segment = self.get_dominant_segment(segments)
        greenish_segment = self.get_greenish_segment(segments)

        segmented_image = self._get_segmented_image(segments)
        segmented_image = np.reshape(segmented_image, sample.shape)
        # background_mask = cv2.inRange(segmented_image, dominant_segment, dominant_segment)
        background_mask = cv2.inRange(segmented_image, greenish_segment, greenish_segment)
        foreground_mask = cv2.bitwise_not(background_mask, background_mask)
        foreground = cv2.bitwise_and(sample, sample, mask=foreground_mask)
        return foreground

    def segment_image(self, sample):
        img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        vectorized = img.reshape((-1, 3))
        vectorized = np.float32(vectorized)

        _, labels, segment_colors = cv2.kmeans(
            vectorized, self.segm_k, None, self.segm_criteria, self.segm_attempts, cv2.KMEANS_PP_CENTERS)

        segment_colors = np.uint8(segment_colors)
        return {'segment_colors': segment_colors, 'labels': labels}

    def get_dominant_segment(self, segments):
        color_pair = []
        hist = np.histogram(segments['labels'], bins=self.segm_k)
        for i in range(self.segm_k):
            color = {'color': segments['segment_colors'][i], 'frequency': hist[0][i]}
            color_pair.append(color)

        dominant_color = max(color_pair, key=lambda i: i['frequency'])
        return dominant_color['color']

    def get_greenish_segment(self, segments):
        green_color = np.array([0, 255, 0])
        return min(segments['segment_colors'], key=lambda color: np.linalg.norm(color - green_color))

    @staticmethod
    def _get_segmented_image(segments):
        return segments['segment_colors'][segments['labels'].flatten()]
