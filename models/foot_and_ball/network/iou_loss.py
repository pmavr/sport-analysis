import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_overlaps_diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious


def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)
    cious = iou - (u + alpha * v)
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    if exchange:
        cious = cious.T
    return cious


def bbox_overlaps_iou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
            bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1 + area2 - inter_area
    ious = inter_area / union
    ious = torch.clamp(ious, min=0, max=1.0)
    if exchange:
        ious = ious.T
    return ious


def bbox_overlaps_giou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
            bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1 + area2 - inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious, min=-1.0, max=1.0)
    if exchange:
        ious = ious.T
    return ious


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    # print(boxes)
    return boxes


class IouLoss(nn.Module):

    def __init__(self, pred_mode='Center', size_sum=True, variances=None, losstype='Giou'):
        super(IouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype

    def forward(self, loc_p, loc_t, prior_data):
        num = loc_p.shape[0]

        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
        else:
            if self.loss == 'Giou':
                loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes, loc_t))
            else:
                if self.loss == 'Diou':
                    loss = torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes, loc_t))
                else:
                    loss = torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))

        if self.size_sum:
            loss = loss
        else:
            loss = loss / num
        return loss


if __name__ == '__main__':
    box_list_1 = torch.Tensor([
        [0, 0, 5, 5],
        [10, 10, 15, 15],
        [20, 20, 25, 25],
        [30, 30, 35, 35],
        [40, 40, 45, 45]
    ])
    box_list_2 = torch.Tensor([
        [1, 1, 6, 6],
        [11, 11, 16, 16],
        [21, 21, 26, 26],
        [31, 31, 36, 36],
        [41, 41, 46, 46]
    ])
    box_list_3 = torch.Tensor([
        [101, 101, 106, 106],
        [110, 110, 116, 116],
        [210, 210, 216, 216],
        [310, 310, 316, 316],
        [410, 410, 416, 416]
    ])
    IoU_Loss = IouLoss(pred_mode='', losstype='Giou')
    GIoU_loss_short = IoU_Loss(box_list_1, box_list_2, None)
    GIoU_loss_long = IoU_Loss(box_list_1, box_list_3, None)

    IoU_Loss = IouLoss(pred_mode='', losstype='Diou')
    DIoU_loss_short = IoU_Loss(box_list_1, box_list_2, None)
    DIoU_loss_long = IoU_Loss(box_list_1, box_list_3, None)

    IoU_Loss = IouLoss(pred_mode='', losstype='Ciou')
    CIoU_loss_short = IoU_Loss(box_list_1, box_list_2, None)
    CIoU_loss_long = IoU_Loss(box_list_1, box_list_3, None)

    IoU_Loss = IouLoss(pred_mode='', losstype='Iou')
    IoU_loss_short = IoU_Loss(box_list_1, box_list_2, None)
    IoU_loss_long = IoU_Loss(box_list_1, box_list_3, None)

    L1_sum_loss_short = F.smooth_l1_loss(box_list_1, box_list_2, reduction='sum')
    L1_sum_loss_long = F.smooth_l1_loss(box_list_1, box_list_3, reduction='sum')

    L2_sum_loss_short = F.mse_loss(box_list_1, box_list_2, reduction='sum')
    L2_sum_loss_long = F.mse_loss(box_list_1, box_list_3, reduction='sum')

    print(f'IoU Loss | short:{IoU_loss_short:.4f} | long:{IoU_loss_long:.4f} \n'
          f'GIoU Loss | short:{GIoU_loss_short:.4f} | long:{GIoU_loss_long:.4f} \n'
          f'DIoU Loss | short:{DIoU_loss_short:.4f} | long:{DIoU_loss_long:.4f} \n'
          f'CIoU Loss | short:{CIoU_loss_short:.4f} | long:{CIoU_loss_long:.4f} \n'
          f'L1 Loss | short:{L1_sum_loss_short:.4f} | long:{L1_sum_loss_long:.4f} \n'
          f'L2 Loss | short:{L2_sum_loss_short:.4f} | long:{L2_sum_loss_long:.4f} ')
