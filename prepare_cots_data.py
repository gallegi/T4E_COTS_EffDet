from multiprocessing import Pool
import os
import ast

import pandas as pd
from lxml.etree import Element, SubElement, ElementTree

import argparse

parser = argparse.ArgumentParser(description='Some arguments')
parser.add_argument('--csv_ann', type=str, default='cots_dataset/cots_more_balanced_cv_split_v2.csv',
                    help='Path to csv annotation file')
parser.add_argument('--image_dir', type=str,
                    help='Path to image folder')                   
parser.add_argument('--output_dir', type=str, default='cots_dataset',
                    help='Path to save generated xml annotations')

args = parser.parse_args()

CLASS_NAME = 'cots'
IMG_HEIGHT, IMG_WIDTH = 720, 1280
OUTPUT_DIR = args.output_dir
CSV_FILE = args.csv_ann
IMAGE_DIR = args.image_dir

def create_ann(ele):
    ann_path = f'{OUTPUT_DIR}/labels/train/{ele.imageid}.xml'

    height, width = IMG_HEIGHT, IMG_WIDTH

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    # node_folder.text = 'train'
    node_folder.text = ''
    
    node_filename = SubElement(node_root, 'filename')
    # node_filename.text = f'{ele.imageid}.jpg'
    node_filename.text = ele.image_relative_path
    
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'
    
    for box in ele.boxes:
        x1, y1, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x2, y2 = x1+w, y1+h
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = CLASS_NAME
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(x1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(y1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(x2)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(y2)

    tree = ElementTree(node_root)
    tree.write(ann_path, pretty_print=True, xml_declaration=False)
    
    return None

class ME:
    def __init__(self, image_path, image_relative_path, imageid, boxes):
        self.image_path = image_path
        self.image_relative_path = image_relative_path
        self.imageid = imageid
        self.boxes = boxes

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

if __name__ == '__main__':
    os.makedirs(f'{OUTPUT_DIR}/labels/train', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/folds', exist_ok=True)

    df = pd.read_csv(CSV_FILE)
    df = df.loc[df['has_annotations'] == True].reset_index(drop=True)

    df['annotations'] = df['annotations'].apply(ast.literal_eval)
    df['bboxes'] = df.annotations.apply(get_bbox)

    for fold in range(5):
        tmp_df = df.loc[df['fold'] == fold]

        meles = []
        for _, row in tmp_df.iterrows():
            video_id = row.video_id
            video_frame = row.video_frame
            image_relative_path = f'video_{video_id}/{video_frame}.jpg'
            image_path  = os.path.join(IMAGE_DIR, image_relative_path)
            image_id = row.image_id
            bboxes = row.bboxes

            if len(bboxes) > 0:
                meles.append(ME(image_path, image_relative_path, row['image_id'], bboxes))
        
        p = Pool(8)
        results = p.map(func=create_ann, iterable = meles)
        p.close()

    for fold in range(5):
        val_df = df.loc[df['fold'] == fold].sample(frac=1).reset_index(drop=True)
        train_df = df.loc[df['fold'] != fold].sample(frac=1).reset_index(drop=True)
        
        effdet_tf = open(f"{OUTPUT_DIR}/folds/effdet_train_fold{fold}.txt", "w")
        for _, row in train_df.iterrows():
            effdet_tf.write(row['image_id'] + '\n')
        effdet_tf.close()

        effdet_vf = open(f"{OUTPUT_DIR}/folds/effdet_valid_fold{fold}.txt", "w")
        for _, row in val_df.iterrows():
            effdet_vf.write(row['image_id'] + '\n')
        effdet_vf.close()