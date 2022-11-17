if __name__ == '__main__':
    from time import time
    import cv2
    import sys
    import pandas as pd
    import glob
    from modules.Team_Detector.PlayerClustering import ColorHistogramClassifier, DominantColorClassifier, SegmentationClassifier
    from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score
    from util import utils

    datasets = [
        {'dir': 'issia'},
        {'dir': 'aek'},
        {'dir': 'belg'},
        {'dir': 'manch'},
    ]
    classifiers = [
        {'name': 'color_histogram_hsv_1', 'method': ColorHistogramClassifier(num_of_teams=4, model='hsv', bins=(12, 20, 20))},
        {'name': 'color_histogram_hsv_2', 'method': ColorHistogramClassifier(num_of_teams=4, model='hsv', bins=(16, 20, 20))},
        {'name': 'color_histogram_hsv_3', 'method': ColorHistogramClassifier(num_of_teams=4, model='hsv', bins=(20, 20, 20))},
        {'name': 'color_histogram_hsv_4', 'method': ColorHistogramClassifier(num_of_teams=4, model='hsv', bins=(24, 20, 20))},
        {'name': 'color_histogram_hsv_5', 'method': ColorHistogramClassifier(num_of_teams=4, model='hsv', bins=(28, 20, 20))},
        {'name': 'color_histogram_hsv_6', 'method': ColorHistogramClassifier(num_of_teams=4, model='hsv', bins=(32, 20, 20))},
        {'name': 'color_histogram_hsv_7', 'method': ColorHistogramClassifier(num_of_teams=4, model='hsv', bins=(36, 20, 20))},
        {'name': 'color_histogram_hsv_8', 'method': ColorHistogramClassifier(num_of_teams=4, model='hsv', bins=(40, 20, 20))},
        {'name': 'color_histogram_bgr_1', 'method': ColorHistogramClassifier(num_of_teams=4, model='bgr', bins=(32, 32, 32))},
        {'name': 'color_histogram_bgr_2', 'method': ColorHistogramClassifier(num_of_teams=4, model='bgr', bins=(64, 64, 64))},
        {'name': 'color_histogram_bgr_3', 'method': ColorHistogramClassifier(num_of_teams=4, model='bgr', bins=(96, 96, 96))},
        {'name': 'color_histogram_bgr_4', 'method': ColorHistogramClassifier(num_of_teams=4, model='bgr', bins=(128, 128, 128))},
        {'name': 'color_histogram_bgr_5', 'method': ColorHistogramClassifier(num_of_teams=4, model='bgr', bins=(160, 160, 160))},
        {'name': 'color_histogram_bgr_6', 'method': ColorHistogramClassifier(num_of_teams=4, model='bgr', bins=(192, 192, 192))},
        {'name': 'color_histogram_bgr_7', 'method': ColorHistogramClassifier(num_of_teams=4, model='bgr', bins=(256, 256, 256))},
        {'name': 'segmentation_1', 'method': SegmentationClassifier(num_of_teams=4, segm_k=2)},
        {'name': 'segmentation_2', 'method': SegmentationClassifier(num_of_teams=4, segm_k=3)},
        {'name': 'segmentation_3', 'method': SegmentationClassifier(num_of_teams=4, segm_k=4)},
        {'name': 'segmentation_4', 'method': SegmentationClassifier(num_of_teams=4, segm_k=5)},
        {'name': 'dominant_color-1', 'method': DominantColorClassifier(num_of_teams=4, num_of_dominant_colors=1)},
        {'name': 'dominant_color-2', 'method': DominantColorClassifier(num_of_teams=4, num_of_dominant_colors=2)},
        {'name': 'dominant_color-3', 'method': DominantColorClassifier(num_of_teams=4, num_of_dominant_colors=3)}
    ]

    for classifier in classifiers:
        mean_homogeneity = 0.
        mean_completeness = 0.
        mean_v_measure = 0.
        for dataset in datasets:
            path = f"{utils.get_project_root()}datasets/player_clustering/{dataset['dir']}"
            method = classifier['method']
            train_images = glob.glob(f"{path}/train_samples/*.jpg")
            train_samples = [cv2.imread(im) for im in train_images]
            # samples = team_classifier.extract_training_samples(image_data, object_detector, court_detector)
            training_color_features = method.extract_color_features(train_samples)
            method.train(training_color_features)

            test_labels = pd.read_csv(f"{path}/{dataset['dir']}_labels.csv")
            test_images = list(test_labels.values[:, 0])
            test_samples = [cv2.imread(f"{path}/test_samples/{im_filename}") for im_filename in test_images]
            start = time()
            test_color_features = method.extract_color_features(test_samples)
            predicted_labels = method.classifier.predict(test_color_features).tolist()
            elapsed_time = time() - start
            gt_labels = list(test_labels.values[:, 1])

            completeness = completeness_score(gt_labels, predicted_labels)
            homogeneity = homogeneity_score(gt_labels, predicted_labels)
            v_measure = v_measure_score(gt_labels, predicted_labels)

            mean_completeness += completeness / datasets.__len__()
            mean_homogeneity += homogeneity / datasets.__len__()
            mean_v_measure += v_measure / datasets.__len__()


            print(f"| method: {classifier['name']} | dataset:{dataset['dir']} "
                  f"| homogeneity: {homogeneity:.3f} | completeness: {completeness:.3f} "
                  f"| V-measure: {v_measure:.3f}")

        print(f"| Model: {classifier['name']} |"
              f'| Mean homogeneity: {mean_homogeneity:.3f} '
              f'| Mean completeness: {mean_completeness:.3f} '
              f'| Mean V-measure: {mean_v_measure:.3f} '
              f'| Elapsed time: {elapsed_time:.3f} |\n')
    sys.exit()
