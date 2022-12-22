import cv2
from time import time
import numpy as np
from collections import defaultdict, Counter
from collections import namedtuple

import torch
from torchvision.ops.boxes import box_iou

import util.utils as utils
import sys


class BoundingBoxPair:

    def __init__(self, gt, pred, iou, conf):
        self.gt = gt
        self.pred = pred
        self.iou = iou
        self.confidence = conf

    def draw_pair(self, frame, threshold=0., text=''):
        image = np.copy(frame)
        if self.pred.score >= threshold:
            image = self.pred.draw_box(image, text)
            if self.gt is not None:
                image = self.gt.draw_box(image, text)
                image = self.draw_iou(image)
        return image

    def draw_iou(self, image):
        x = min(self.gt.x1, self.pred.x1)
        y = max(self.gt.y2, self.pred.y2)
        cv2.putText(image, f'{round(self.iou, 3)}', (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image


def centroid_metric(detections, ground_truths):
    total_box_pairs = []
    total_gts = get_ground_truth_boxes_num(ground_truths)
    objects_found = 0
    for frame_id in ground_truths:
        bbox_pairs = _find_bbox_pairs(ground_truths[frame_id], detections[frame_id])
        total_box_pairs.extend(bbox_pairs)

    for pair in total_box_pairs:
        objects_found += pair.iou
    accuracy = objects_found/total_gts
    return accuracy


def eleven_point_interpolated_PR_curve(detections, ground_truths, iou_threshold=.5):
    precisions, recalls = PR_curve(detections, ground_truths, iou_threshold=iou_threshold)
    recall_levels = np.array([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    precisions_per_recall_level = np.zeros(11)

    for idx, rec_level in enumerate(recall_levels):
        recalls_above_level = np.argwhere(recalls[:] >= rec_level)
        max_precision = 0
        if recalls_above_level.size > 0:
            max_precision = max(precisions[recalls_above_level.min():])
        precisions_per_recall_level[idx] = max_precision

    average_precision = np.average(precisions_per_recall_level)
    return average_precision, precisions_per_recall_level, recall_levels


def get_ground_truth_boxes_num(gt_box_lists):
    num = 0
    for frame_id in gt_box_lists:
            num += len(gt_box_lists[frame_id])
    return num


def PR_curve(detections, ground_truths, iou_threshold=.5):
    total_box_pairs = []
    total_gts = get_ground_truth_boxes_num(ground_truths)

    for frame_id in ground_truths:
        bbox_pairs = _find_bbox_pairs(ground_truths[frame_id], detections[frame_id])
        total_box_pairs.extend(bbox_pairs)

    # order pairs by confidence
    total_box_pairs = sorted(total_box_pairs, key=lambda pair: pair.confidence, reverse=True)

    # calculate precision/recall arrays
    true_positives = np.array(np.zeros(len(total_box_pairs)))
    false_positives = np.array(np.zeros(len(total_box_pairs)))
    for id, pair in enumerate(total_box_pairs):
        # determine TP or FP
        if pair.iou >= iou_threshold:
            true_positives[id] = 1
        else:
            false_positives[id] = 1

    TP_cumsum = np.cumsum(true_positives, axis=0)
    FP_cumsum = np.cumsum(false_positives, axis=0)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
    recalls = TP_cumsum / total_gts
    return precisions, recalls


def _find_bbox_pairs(gt_list, det_list):
    remaining_gt_list = gt_list.copy()
    pairs = []
    for det in det_list:
        pair, remaining_gt_list = _find_bbox_pair(det, remaining_gt_list)
        pairs.append(pair)
    return pairs


def _find_bbox_pair(pred_box, gt_box_list):
    output_box_list = gt_box_list.copy()
    best_match_id = -1
    best_iou = sys.float_info.min

    for idx in range(len(gt_box_list)):
        # iou = intersection_over_ground_truth(pred_box, gt_box_list[idx])
        # iou = intersection_over_ground_truth(pred_box, gt_box_list[idx])
        # if best_iou < iou:
        #     best_iou = iou
        #     best_match_id = idx

        dist = euclidian_dist(pred_box.centroid, gt_box_list[idx].centroid)
        if dist <= 45:
            best_iou = 1
            best_match_id = idx

    if best_match_id == -1:
        pair = BoundingBoxPair(gt=None,
                               pred=pred_box,
                               iou=0.,
                               conf=pred_box.score)
    else:
        pair = BoundingBoxPair(gt=gt_box_list[best_match_id],
                               pred=pred_box,
                               iou=best_iou,
                               conf=pred_box.score)
        del output_box_list[best_match_id]
    return pair, output_box_list


def intersection_over_union(box1, box2):
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1.h + 1) * (box1.w + 1)
    box2_area = (box2.h + 1) * (box2.w + 1)

    union_area = float(box1_area + box2_area - intersection_area)

    iou = intersection_area / union_area

    return iou


def intersection_over_ground_truth(detection, ground_truth):
    x1 = max(detection.x1, ground_truth.x1)
    y1 = max(detection.y1, ground_truth.y1)
    x2 = min(detection.x2, ground_truth.x2)
    y2 = min(detection.y2, ground_truth.y2)

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    ground_truth_area = (ground_truth.h + 1) * (ground_truth.w + 1)

    iogt = intersection_area / ground_truth_area
    return iogt


def euclidian_dist(x1, x2):
    return np.sqrt((float(x1[0])-float(x2[0]))**2 + (float(x1[1])-float(x2[1]))**2)


def plot_precision_recall_curve(precisions, recalls):
    precisions = np.append(np.ones(1), precisions)
    recalls = np.append(np.zeros(1), recalls)

    import matplotlib.pyplot as plt
    plt.plot(recalls, precisions, linewidth=3, color="red", zorder=0)
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.xlim([0., 1.01])
    plt.ylim([0., 1.01])
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.show()


def plot_interpolated_precision_recall_curve(precisions_per_recall_level, recall_levels):
    recalls = [recall_levels[0]]
    precisions = [precisions_per_recall_level[0]]
    for i in range(1, 10):
        rec_level = recall_levels[i]
        prec_value = precisions_per_recall_level[i]
        if prec_value < precisions[-1]:
            recalls.append(recalls[-1])
            precisions.append(prec_value)
        recalls.append(rec_level)
        precisions.append(prec_value)

    recalls.append(1.)
    precisions.append(precisions[-1])

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    import matplotlib.pyplot as plt
    plt.plot(recalls, precisions, linewidth=3, color="green", zorder=0)
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.xlim([0., 1.01])
    plt.ylim([0., 1.01])
    plt.scatter(recall_levels, precisions_per_recall_level, c='b', marker='o')
    plt.title("11-point Interpolated Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.show()


if __name__ == '__main__':
    from modules.Team_Detector.BoundingBox import BoundingBox


    box_coords_1 = [0, 0, 5, 5]
    box_coords_2 = [1, 1, 6, 6]
    b1 = BoundingBox(box_coords=box_coords_1)
    b2 = BoundingBox(box_coords=box_coords_2)
    iou = intersection_over_union(b1, b2)
    torch_iou = box_iou(torch.Tensor([box_coords_1]), torch.Tensor([box_coords_2])).item()

    prec = np.array([1., 1., .67, .5, .4, .5, .57, .5, .44, .5])
    rec = np.array([.2, .4, .4, .4, .4, .6, .8, .8, .8, 1.])
    tp = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
    fp = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    TP_cumsum = np.cumsum(tp, axis=0)
    FP_cumsum = np.cumsum(fp, axis=0)
    precs = TP_cumsum / (TP_cumsum + FP_cumsum)
    recs = TP_cumsum / 5.

    ap, precs, recs = eleven_point_interpolated_PR_curve(recs, precs)
    # ap, precs, recs, _ = calculate_ap_11_point_interp(recs, precs)

    # assert (np.round(precs, 2) == prec).all()
    # assert (np.round(recs, 2) == rec).all()
    plot_interpolated_precision_recall_curve(recall_levels=recs, precisions_per_recall_level=precs)
