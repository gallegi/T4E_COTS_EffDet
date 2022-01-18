#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import argparse
import time
import torch
import torch.nn.parallel
from contextlib import suppress

from effdet import create_model, create_evaluator, create_dataset, create_loader, create_cots_dataset
from effdet.data import resolve_input_config
from timm.utils import AverageMeter, setup_default_logging
from timm.models.layers import set_layer_config
import pickle

from metrics import calc_f2_score

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('root', metavar='DIR',
                    help='path to dataset root')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='val',
                    help='validation split')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')

# custom arguments
parser.add_argument('--im_dir', default='', type=str,
                    help='Path to image_folder')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--conf_thresh', default=0.5, type=float,
                    help='Confidence threshold for prediction')
parser.add_argument('--overlap_algo', default='nms', type=str,
                     help='Algorithm to process overlapping boxes: nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion')
parser.add_argument('--iou_thresh', default=0.5, type=float,
                     help='IOU threshold for overlap processing algorithm')
parser.add_argument('--skip_box_thresh', default=0.01, type=float,
                     help='Confidence threshold to remove the box when using the algorithm to merge overlapping boxes')

import numpy as np

from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion

def post_process_overlap(output, skip_box_thr=0.01, sigma=0.1, iou_thr=0.5, algo_type='nms'):
    
    boxes_list = [output[:,:4]]
    scores_list = [output[:,4]]
    labels_list = [output[:,5]]
    weights = [1]

    if algo_type == 'nms':
        boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    elif algo_type == 'soft_nms':
        boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    elif algo_type == 'non_maximum_weighted':
        boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    elif algo_type == 'weighted_boxes_fusion':
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    else:
        raise ValueError('No such algo_type')

    return boxes, scores, labels

def get_xywh_grountruths(cls, bbox, img_size, img_scale):
    xywh_groundtruths = []

    # filter dummy box in target
    filtering = cls != -1
    cls = cls[filtering]
    bbox = bbox[filtering, :]

    bbox = bbox * img_scale

    for box in bbox:
        y1, x1, y2, x2 = box.cpu().numpy().astype(int)
        w, h = x2-x1, y2 - y1

        xywh_groundtruths.append([x1, y1, w, h])

    xywh_groundtruths = np.array(xywh_groundtruths)

    return xywh_groundtruths

def get_xywh_predictions(output, img_scale, args):
    output[:,:4] = output[:,:4] / args.img_size 
    output = np.clip(output, 0, 1)

    boxes, scores, labels = post_process_overlap(output, args.skip_box_thresh, iou_thr=args.iou_thresh,
                                                algo_type=args.overlap_algo)
    boxes *= args.img_size # back to absolute size

    boxes *= img_scale.cpu().numpy() # back to original size

    # filter by conf score
    filtering = scores >= args.conf_thresh 
    scores = scores[filtering]
    boxes = boxes[filtering]
    labels = labels[filtering]

    xywh_predictions = []
    for score, box in zip(scores, boxes):
        px1, py1, px2, py2 = box
        pw, ph = px2 - px1, py2 - py1
        xywh_predictions.append([score, px1, py1, pw, ph])

    xywh_predictions = np.array(xywh_predictions)
    return xywh_predictions

def validate(args):
    setup_default_logging()

    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    args.pretrained = args.pretrained or not args.checkpoint  # might as well try to validate something
    args.prefetcher = not args.no_prefetcher

    # create model
    with set_layer_config(scriptable=args.torchscript):
        extra_args = {}
        if args.img_size is not None:
            extra_args = dict(image_size=(args.img_size, args.img_size))
        bench = create_model(
            args.model,
            bench_task='predict',
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            redundant_bias=args.redundant_bias,
            soft_nms=args.soft_nms,
            checkpoint_path=args.checkpoint,
            checkpoint_ema=args.use_ema,
            **extra_args,
        )
    model_config = bench.config

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = bench.cuda()

    amp_autocast = suppress
    if args.apex_amp:
        bench = amp.initialize(bench, opt_level='O1')
        print('Using NVIDIA APEX AMP. Validating in mixed precision.')
    elif args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        print('Using native Torch AMP. Validating in mixed precision.')
    else:
        print('AMP not enabled. Validating in float32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

    # dataset = create_dataset(args.dataset, args.root, args.split)
    dataset_train, dataset_eval = create_cots_dataset(name=args.dataset, root=args.root, im_dir=args.im_dir, fold=args.fold)

    input_config = resolve_input_config(args, model_config)
    loader = create_loader(
        # dataset,
        dataset_eval,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    # evaluator = create_evaluator(args.dataset, dataset_eval, pred_yxyx=False)
    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1

    list_image_ids = []
    list_predictions = []
    list_groundtruths = []

    _index = -1
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            with amp_autocast(), torch.no_grad():
                outputs = bench(inputs)

            # get xywh ground truth
            for j in range(len(inputs)):
                _index += 1
                # retrieve im_id
                im_id = dataset_eval._parser.img_ids[_index]
                list_image_ids.append(im_id)

                cls, bbox, img_size, img_scale = targets['cls'][j], targets['bbox'][j], targets['img_size'][j], \
                                            targets['img_scale'][j]
                xywh_groundtruths = get_xywh_grountruths(cls, bbox, img_size, img_scale)  
                list_groundtruths.append(xywh_groundtruths)

                output = outputs[j].detach().cpu().numpy()
                xywh_predictions = get_xywh_predictions(output, img_scale=img_scale, args=args)
                list_predictions.append(xywh_predictions)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0 or i == last_idx:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=inputs.size(0) / batch_time.avg)
                )

    f2, log_dict, detail_df = calc_f2_score(list_image_ids, list_groundtruths, list_predictions, verbose=True)
    print('F2:', f2)

    out_dict = dict()
    for image_id, prediction in zip(list_image_ids, list_predictions):
        out_dict[image_id] = prediction

    print(f'Save result to {args.results}')

    version = args.checkpoint.split('/')[-2]
    os.makedirs(f'{args.results}/{version}', exist_ok=True)
    with open(f'{args.results}/{version}/{version}_predictions_thresh{args.conf_thresh}.pkl', 'wb') as f:
        pickle.dump(out_dict, f)

    with open(f'{args.results}/{version}/{version}_log_dict_thresh{args.conf_thresh}.pkl', 'wb') as f:
        pickle.dump(log_dict, f)

    detail_df.to_csv(f'{args.results}/{version}/{version}_detail_thresh{args.conf_thresh}.csv', index=False)

    return f2

def main():
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()

