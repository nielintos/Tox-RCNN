#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Vahid Abrishami
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.timer import Timer
import matplotlib.pyplot as plt
from xml.dom import minidom
import numpy as np
from scipy import misc
import scipy.io as sio
import pandas as pd
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'HEALTHY', 'TOXICITY_AFFECTED')

NETS = {'cytotoxicity': ('cytotoxicity',
                         'cytotoxicity_faster_rcnn_iter_750000.caffemodel')}


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def get_data_from_tag(node, tag):
    return node.getElementsByTagName(tag)[0].childNodes[0].data


def vis_detections(im, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -2] >= thresh)[0]
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im[:, :, 1], aspect='equal', cmap="gray")
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -2]
        class_name = CLASSES[int(dets[i, -1])]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name, xml_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_tmp = misc.imread(im_file)
    w, h = im_tmp.shape
    im = np.empty((w, h, 3), dtype=np.uint16)
    im[:, :, 0] = im_tmp
    im[:, :, 1] = im[:, :, 2] = im[:, :, 0]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for '
          '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    dets_all = np.array([])
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_inds = np.ones(cls_scores.shape[0]) * cls_ind
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis], cls_inds[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if cls_ind == 1:
            dets_all = dets
        else:
            dets_all = np.concatenate((dets_all, dets), axis=0)

    keep = nms(dets_all, NMS_THRESH)
    dets = dets_all[keep, :]
    # Uncomment this part if you want to see detections
    # vis_detections(im, dets, thresh=CONF_THRESH)
    # plt.show()
    return dets


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use',
                        default='cytotoxicity')
    parser.add_argument('--output', dest='output_csv', help='File name for table with predictions', action='store',
                        metavar='FILE')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'cytotoxicity',
                            'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              args.demo_net)

    print(caffemodel)

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print('\n\nLoaded network {:s}'.format(caffemodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)
    master_df = pd.DataFrame(columns=['PATH', 'min_det_X', 'min_det_Y', 'max_det_X', 'max_det_Y',
                                           'Probability_HEALTHY_py'])
    with open(os.path.join(cfg.ROOT_DIR, 'data\\cytotoxicity_devkit\\data\\ImageSets\\test.txt')) as f:
        for im_name in f:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Detection for {}.tif'.format(im_name.strip()))
            image_path = os.path.join(cfg.ROOT_DIR, 'data\\cytotoxicity_devkit\\data\\Images'
                                      , im_name.strip() + '.tif')
            xml_path = os.path.join(cfg.ROOT_DIR, 'data\\cytotoxicity_devkit\\data\\Annotations'
                                    , im_name.strip() + '.xml')
            dets = demo(net, image_path, xml_path)
            image_path = np.repeat(image_path, dets.shape[0])
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            classes = [CLASSES[int(cls_ind)] for cls_ind in dets[:, -1]]
            probability_healthy_py = [cls_ind % 2 for cls_ind in dets[:, -1]]
            raw_data = {'PATH': image_path,
                        'min_det_X': x1,
                        'min_det_Y': y1,
                        'max_det_X': x2,
                        'max_det_Y': y2,
                        'Probability_HEALTHY_py': probability_healthy_py}
            df = pd.DataFrame(raw_data, columns=['PATH', 'min_det_X',
                                                 'min_det_Y', 'max_det_X',
                                                 'max_det_Y', 'Probability_HEALTHY_py'])
            master_df = master_df.append(df).reset_index(drop=True)

    master_df = master_df[['PATH', 'min_det_X', 'min_det_Y', 'max_det_X',
                           'max_det_Y', 'Probability_HEALTHY_py']]
    master_df.to_csv(args.output_csv, index_label='Id_Detect')