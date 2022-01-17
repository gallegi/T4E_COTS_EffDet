import glob
import os
import cv2
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


def yolo2coco(yolo_anno_list, width=1280, height=720):
    anno_list = []
    for yolo_anno in yolo_anno_list:
        yolo_anno = yolo_anno.split(' ')
        cls = yolo_anno[0]
        x_center, y_center, w, h = float(yolo_anno[1]), float(yolo_anno[2]), float(yolo_anno[3]), float(yolo_anno[4])
        conf_score = yolo_anno[5]
        coco_w, coco_h = w * width, h * height
        coco_x_center, coco_y_center = x_center * width, y_center * height
        coco_x_left = coco_x_center - coco_w/2
        coco_y_left = coco_y_center - coco_h/2
        anno_list.append(np.array([float(conf_score), coco_x_left, coco_y_left, coco_w, coco_h]))
    return np.array(anno_list)

def calc_iou(bboxes1, bboxes2, bbox_mode='xywh'):
    assert len(bboxes1.shape) == 2 and bboxes1.shape[1] == 4
    assert len(bboxes2.shape) == 2 and bboxes2.shape[1] == 4
    
    bboxes1 = bboxes1.copy()
    bboxes2 = bboxes2.copy()
    
    if bbox_mode == 'xywh':
        bboxes1[:, 2:] += bboxes1[:, :2]
        bboxes2[:, 2:] += bboxes2[:, :2]

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def f_beta_bntz(p, r, beta = 2):
    return ((1 + beta**2) * p * r) / (beta**2 * p + r + 1e-16)

def calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th, verbose=False):
    gt_bboxes = gt_bboxes.copy()
    pred_bboxes = pred_bboxes.copy()
    tp = 0
    fp = 0
    for pred_bbox in pred_bboxes:
        ious = calc_iou(gt_bboxes, pred_bbox[None, 1:])
        max_iou = ious.max()
        if max_iou > iou_th:
            tp += 1
            gt_bboxes = np.delete(gt_bboxes, ious.argmax(), axis=0)
        else:
            fp += 1
        if len(gt_bboxes) == 0:
            fp += len(pred_bboxes) - tp
            break

    fn = len(gt_bboxes)
    return tp, fp, fn

def calc_is_correct(gt_bboxes, pred_bboxes, iou_th):
    """
    gt_bboxes: (N, 4) np.array in xywh format
    pred_bboxes: (N, 5) np.array in conf+xywh format
    """
    if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, 0
        return tps, fps, fns
    
    elif len(gt_bboxes) == 0:
        tps, fps, fns = 0, len(pred_bboxes), 0
        return tps, fps, fns
    
    elif len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, len(gt_bboxes)
        return tps, fps, fns
    
    pred_bboxes = pred_bboxes[pred_bboxes[:,0].argsort()[::-1]] # sort by conf
    tp, fp, fn = calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th)
    return tp, fp, fn

def calc_f2_score(image_id_list, gt_bboxes_list, pred_bboxes_list, verbose=False):
    """
    gt_bboxes_list: list of (N, 4) np.array in xywh format
    pred_bboxes_list: list of (N, 5) np.array in conf+xywh format
    """
    if len(gt_bboxes_list) != len(pred_bboxes_list):
        print('Error miss match gt vs predict')
        return None, None, None

    csv_dict = {}
 
    fbeta_arr, log_dict = [], {}
    for iou_th in np.arange(0.3, 0.85, 0.05):
        csv_dict[f'tps@{round(iou_th,2)}'] = []
        csv_dict[f'fps@{round(iou_th,2)}'] = []
        csv_dict[f'fns@{round(iou_th,2)}'] = []
        csv_dict[f'f2@{round(iou_th,2)}'] = []
        tps, fps, fns = 0, 0, 0
        for gt_bboxes, pred_bboxes in zip(gt_bboxes_list, pred_bboxes_list):
            tp, fp, fn = calc_is_correct(gt_bboxes, pred_bboxes, iou_th)
            i_p = tp/(tp+fp + 1e-16)
            i_r = tp/(tp+fn + 1e-16)
            i_fbeta =  f_beta_bntz(i_p, i_r)
            tps += tp
            fps += fp
            fns += fn
            csv_dict[f'tps@{round(iou_th,2)}'].append(tp)
            csv_dict[f'fps@{round(iou_th,2)}'].append(fp)
            csv_dict[f'fns@{round(iou_th,2)}'].append(fn)
            csv_dict[f'f2@{round(iou_th,2)}'].append(i_fbeta)
        
        p = tps/(tps+fps + 1e-16)
        r = tps/(tps+fns + 1e-16)
        fbeta = f_beta_bntz(p, r)
        fbeta_arr.append(fbeta)
        if verbose:
            log_dict[iou_th] = {
                'tps':tps,
                'fps':fps,
                'fns':fns,
                'p':p,
                'r':r,
                'f2':fbeta
            }
        log_csv = pd.DataFrame()
        log_csv['image_id'] = image_id_list
    for d_key, d_value in csv_dict.items():
        log_csv[d_key] = d_value
    return np.mean(np.array(fbeta_arr)), log_dict, log_csv