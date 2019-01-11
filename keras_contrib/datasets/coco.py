#!/usr/bin/env python
# coding=utf-8
"""
This is a script for downloading and converting the microsoft coco dataset
from mscoco.org. This can be run as an independent executable to download
the dataset or be imported by scripts used for larger experiments.
"""
from __future__ import division, print_function, unicode_literals
import os
import errno
import zipfile
import json
from sacred import Experiment, Ingredient
import numpy as np
from PIL import Image
from keras.utils import get_file
from keras.utils.generic_utils import Progbar
from pycocotools.coco import COCO


def palette():
    max_cid = max(ids()) + 1
    return [(cid, cid, cid) for cid in range(max_cid)]


def cids_to_ids_map():
    return {cid: idx for idx, cid in enumerate(ids())}


def ids():
    return [0,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def id_to_palette_map():
    return {idx: color for idx, color in enumerate(palette())}
    # return {0: (0, 0, 0), idx: (idx, idx, idx)
    # for idx, _ in enumerate(categories())}


def cid_to_palette_map():
    return {ids()[idx]: color for idx, color in enumerate(palette())}


def palette_to_id_map():
    return {color: ids()[idx] for idx, color in enumerate(palette())}
    # return {(0, 0, 0): 0, (idx, idx, idx): idx
    # for idx, _ in enumerate(categories())}


def class_weight(image_segmentation_stats_file=None,
                 weighting_algorithm='total_pixels_p_complement'):
    # weights = defaultdict(lambda: 1.5)
    if image_segmentation_stats_file is None:
        weights = {i: 1.5 for i in ids()}
        weights[0] = 0.5
        return weights
    else:
        with open(image_segmentation_stats_file, 'r') as fjson:
            stats = json.loads(fjson)
            return stats[weighting_algorithm]


def mask_to_palette_map(cid):
    mapper = id_to_palette_map()
    return {0: mapper[0], 255: mapper[cid]}


def categories():  # 80 classes
    return ['background',  # class zero
            'person', 'bicycle', 'car', 'motorcycle',
            'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife',
            'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def id_to_category(category_id):
    return {cid: categories()[idx] for idx, cid in enumerate(ids())}[category_id]


def category_to_cid_map():
    return {category: ids()[idx] for idx, category in enumerate(categories())}


def mkdir_p(path):
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# ============== Ingredient 2: dataset =======================
data_coco = Experiment("dataset")


@data_coco.config
def coco_config():
    # TODO(ahundt) add md5 sums for each file
    verbose = 1
    coco_api = 'https://github.com/pdollar/coco/'
    dataset_root = os.path.join(os.path.expanduser('~'), 'datasets')
    dataset_path = os.path.join(dataset_root, 'coco')
    urls = [
        'coco2014/train2014.zip',
        'coco2014/val2014.zip',
        'coco2014/test2014.zip',
        'coco2015/test2015.zip',
        'annotations-1-0-3/instances_train-val2014.zip',
        'annotations-1-0-3/person_keypoints_trainval2014.zip',
        'annotations-1-0-4/image_info_test2014.zip',
        'annotations-1-0-4/image_info_test2015.zip',
        'annotations-1-0-3/captions_train-val2014.zip'
    ]
    base_url = 'http://msvocds.blob.core.windows.net/'
    urls = [base_url + x for x in urls]
    data_prefixes = [
        'train2014',
        'val2014',
        'test2014',
        'test2015',
    ]
    image_filenames = [prefix + '.zip' for prefix in data_prefixes]
    annotation_filenames = [
        'instances_train-val2014.zip',  # training AND validation info
        'image_info_test2014.zip',  # basic info like download links + category
        'image_info_test2015.zip',  # basic info like download links + category
        'person_keypoints_trainval2014.zip',  # elbows, head, wrist etc
        'captions_train-val2014.zip',  # descriptions of images
    ]
    md5s = [
        '0da8c0bd3d6becc4dcb32757491aca88',  # train2014.zip
        'a3d79f5ed8d289b7a7554ce06a5782b3',  # val2014.zip
        '04127eef689ceac55e3a572c2c92f264',  # test2014.zip
        '65562e58af7d695cc47356951578c041',  # test2015.zip
        '59582776b8dd745d649cd249ada5acf7',  # instances_train-val2014.zip
        '926b9df843c698817ee62e0e049e3753',  # person_keypoints_trainval2014.zip
        'f3366b66dc90d8ae0764806c95e43c86',  # image_info_test2014.zip
        '8a5ad1a903b7896df7f8b34833b61757',  # image_info_test2015.zip
        '5750999c8c964077e3c81581170be65b'   # captions_train-val2014.zip
    ]
    filenames = image_filenames + annotation_filenames
    seg_mask_path = os.path.join(dataset_path, 'seg_mask')
    annotation_json = [
        'annotations/instances_train2014.json',
        'annotations/instances_val2014.json'
    ]
    annotation_paths = [os.path.join(dataset_path, postfix)
                        for postfix in annotation_json]
    # only first two data prefixes contain segmentation masks
    seg_mask_image_paths = [os.path.join(dataset_path, prefix)
                            for prefix in data_prefixes[0:1]]
    seg_mask_output_paths = [os.path.join(seg_mask_path, prefix)
                             for prefix in data_prefixes[0:1]]
    seg_mask_extensions = ['.npy' for prefix in data_prefixes[0:1]]
    image_dirs = [os.path.join(dataset_path, prefix) for prefix in data_prefixes]
    image_extensions = ['.jpg' for prefix in data_prefixes]
    voc_imageset_txt_paths = [os.path.join(dataset_path,
                                           'annotations', prefix + '.txt')
                              for prefix in data_prefixes]


@data_coco.capture
def coco_files(dataset_path, filenames, dataset_root, urls, md5s, annotation_paths):
    print(dataset_path)
    print(dataset_root)
    print(urls)
    print(filenames)
    print(md5s)
    print(annotation_paths)
    return [os.path.join(dataset_path, file) for file in filenames]


@data_coco.command
def print_coco_files(dataset_path, filenames, dataset_root,
                     urls, md5s, annotation_paths):
    coco_files(dataset_path, filenames, dataset_root, urls, md5s, annotation_paths)


@data_coco.command
def coco_download(dataset_path, filenames, dataset_root,
                  urls, md5s, annotation_paths):
    zip_paths = coco_files(dataset_path, filenames, dataset_root,
                           urls, md5s, annotation_paths)
    for url, filename, md5 in zip(urls, filenames, md5s):
        path = get_file(filename, url, md5_hash=md5,
                        extract=True, cache_subdir=dataset_path)
        # TODO(ahundt) check if it is already extracted, don't re-extract. see
        # https://github.com/fchollet/keras/issues/5861
        zip_file = zipfile.ZipFile(path, 'r')
        zip_file.extractall(path=dataset_path)
        zip_file.close()


@data_coco.command
def coco_json_to_segmentation(seg_mask_output_paths,
                              annotation_paths, seg_mask_image_paths, verbose):
    for (seg_mask_path, annFile, image_path) in zip(
            seg_mask_output_paths, annotation_paths, seg_mask_image_paths):
        print('Loading COCO Annotations File: ', annFile)
        print('Segmentation Mask Output Folder: ', seg_mask_path)
        print('Source Image Folder: ', image_path)
        print('\n'
              'WARNING: Each pixel can have multiple classes! That means'
              'class data overlaps. Also, single objects can be outlined'
              'multiple times because they were labeled by different people!'
              'In other words, even a single object may be segmented twice.'
              'This means the .png files are missing entire objects.\n\n'
              'Use of categorical one-hot encoded .npy files is recommended,'
              'but .npy files also have limitations, because the .npy files'
              'only have one label per pixel for each class,'
              'and currently take the union of multiple human class labels.'
              'Improving how your data is handled will improve your results'
              'so remember to consider that limitation. There is still'
              'an opportunity to improve how this training data is handled &'
              'integrated with your training scripts and utilities...')
        coco = COCO(annFile)

        print('Converting Annotations to Segmentation Masks...')
        mkdir_p(seg_mask_path)
        total_imgs = len(coco.imgToAnns.keys())
        progbar = Progbar(total_imgs + len(coco.getImgIds()), verbose=verbose)
        # 'annotations' was previously 'instances' in an old version
        for img_num in range(total_imgs):
            # Both [0]'s are used to extract the element from a list
            img = coco.loadImgs(
                coco.imgToAnns[coco.imgToAnns.keys()[img_num]][0]['image_id'])[0]
            h = img['height']
            w = img['width']
            name = img['file_name']
            root_name = name[:-4]
            filename = os.path.join(seg_mask_path, root_name + ".png")
            file_exists = os.path.exists(filename)
            if file_exists:
                progbar.update(img_num, [('file_fraction_already_exists', 1)])
                continue
            else:
                progbar.update(img_num, [('file_fraction_already_exists', 0)])
                print(filename)

            MASK = np.zeros((h, w), dtype=np.uint8)
            np.where(MASK > 0)
            for ann in coco.imgToAnns[coco.imgToAnns.keys()[img_num]]:
                mask = coco.annToMask(ann)
                idxs = np.where(mask > 0)
                MASK[idxs] = ann['category_id']

            im = Image.fromarray(MASK)
            im.save(filename)

        print('\nConverting Annotations to one hot encoded'
              'categorical .npy Segmentation Masks...')
        img_ids = coco.getImgIds()
        use_original_dims = True  # not target_shape
        for idx, img_id in enumerate(img_ids):
            img = coco.loadImgs(img_id)[0]
            name = img['file_name']
            root_name = name[:-4]
            filename = os.path.join(seg_mask_path, root_name + ".npy")
            file_exists = os.path.exists(filename)
            if file_exists:
                progbar.add(1, [('file_fraction_already_exists', 1)])
                continue
            else:
                progbar.add(1, [('file_fraction_already_exists', 0)])

            if use_original_dims:
                target_shape = (img['height'], img['width'], max(ids()) + 1)
            ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            mask_one_hot = np.zeros(target_shape, dtype=np.uint8)
            mask_one_hot[:, :, 0] = 1  # every pixel begins as background
            # mask_one_hot = cv2.resize(mask_one_hot,
            #                           target_shape[:2],
            #                           interpolation=cv2.INTER_NEAREST)

            for ann in anns:
                mask_partial = coco.annToMask(ann)
                # mask_partial = cv2.resize(mask_partial,
                #                           (target_shape[1], target_shape[0]),
                #                           interpolation=cv2.INTER_NEAREST)
                # # width and height match
                # assert mask_one_hot.shape[:2] == mask_partial.shape[:2]
                #    print('another shape:',
                #          mask_one_hot[mask_partial > 0].shape)
                mask_one_hot[mask_partial > 0, ann['category_id']] = 1
                mask_one_hot[mask_partial > 0, 0] = 0

            np.save(filename, mask_one_hot)


@data_coco.command
def coco_to_pascal_voc_imageset_txt(voc_imageset_txt_paths, image_dirs,
                                    image_extensions):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # Get some image/annotation pairs for example
    for imgset_path, img_dir, t_ext in zip(
            voc_imageset_txt_paths, image_dirs, image_extensions):
        with open(imgset_path, 'w') as txtfile:
            [txtfile.write(os.path.splitext(os.path.basename(file))[0] + '\n')
             for file in os.listdir(img_dir) if file.endswith(t_ext)]


@data_coco.command
def coco_image_segmentation_stats(seg_mask_output_paths, annotation_paths,
                                  seg_mask_image_paths, verbose):
    for (seg_mask_path, annFile, image_path) in zip(
            seg_mask_output_paths, annotation_paths, seg_mask_image_paths):
        print('Loading COCO Annotations File: ', annFile)
        print('Segmentation Mask Output Folder: ', seg_mask_path)
        print('Source Image Folder: ', image_path)
        stats_json = os.path.join(seg_mask_path,
                                  'image_segmentation_class_stats.json')
        print('Image stats will be saved to:', stats_json)
        cat_csv = os.path.join(seg_mask_path,
                               'class_counts_over_sum_category_counts.csv')
        print('Category weights will be saved to:', cat_csv)
        coco = COCO(annFile)
        print('Annotation file info:')
        coco.info()
        print('category ids, not including 0 for background:')
        print(coco.getCatIds())
        # display COCO categories and supercategories
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('categories: \n\n', ' '.join(nms))

        nms = set([cat['supercategory'] for cat in cats])
        print('supercategories: \n', ' '.join(nms))
        img_ids = coco.getImgIds()
        use_original_dims = True  # not target_shape
        max_ids = max(ids()) + 1  # add background category
        # 0 indicates no category (not even background) for counting bins
        max_bin_count = max_ids + 1
        bin_count = np.zeros(max_bin_count)
        total_pixels = 0

        print('Calculating image segmentation stats...')
        progbar = Progbar(len(img_ids), verbose=verbose)
        i = 0
        for idx, img_id in enumerate(img_ids):
            img = coco.loadImgs(img_id)[0]
            i += 1
            progbar.update(i)
            ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            target_shape = (img['height'], img['width'], max_ids)
            # print('\ntarget_shape:', target_shape)
            mask_one_hot = np.zeros(target_shape, dtype=np.uint8)

            # Note to only count background pixels once, we define a temporary
            # null class of 0, and shift all class category ids up by 1
            mask_one_hot[:, :, 0] = 1  # every pixel begins as background

            for ann in anns:
                mask_partial = coco.annToMask(ann)
                above_zero = mask_partial > 0
                mask_one_hot[above_zero, ann['category_id']] = ann['category_id'] + 1
                mask_one_hot[above_zero, 0] = 0

            # print( mask_one_hot)
            # print('initial bin_count shape:', np.shape(bin_count))
            # flat_mask_one_hot = mask_one_hot.flatten()
            bincount_result = np.bincount(mask_one_hot.flatten())
            # print('bincount_result TYPE:', type(bincount_result))
            # np.array(np.ndarray.flatten(np.bincount(np.ndarray.
            # flatten(np.array(mask_one_hot)).astype(int))).resize(max_bin_count))
            # print('bincount_result:', bincount_result)
            # print('bincount_result_shape', np.shape(bincount_result))
            length = int(np.shape(bincount_result)[0])
            zeros_to_add = max_bin_count - length
            z = np.zeros(zeros_to_add)
            # print('zeros_to_add TYPE:', type(zeros_to_add))
            # this is a workaround because for some strange reason the
            # output type of bincount couldn't interact with other numpy arrays
            bincount_result_long = bincount_result.tolist() + z.tolist()
            # bincount_result = bincount_result.resize(max_bin_count)
            # print('bincount_result2:', bincount_result_long)
            # print('bincount_result2_shape',bincount_result_long)
            bin_count = bin_count + np.array(bincount_result_long)
            total_pixels += (img['height'] * img['width'])

        print('Final Tally:')
        # shift categories back down by 1
        bin_count = bin_count[1:]
        category_ids = range(bin_count.size)
        sum_category_counts = np.sum(bin_count)

        # sum will be =1 as a pixel can be in multiple categories
        category_counts_over_sum_category_counts = \
            np.true_divide(bin_count.astype(np.float64), sum_category_counts)
        np.savetxt(cat_csv, category_counts_over_sum_category_counts)

        # sum will be >1 as a pixel can be in multiple categories
        category_counts_over_total_pixels = \
            np.true_divide(bin_count.astype(np.float64), total_pixels)

        # less common categories have more weight, sum = 1
        category_counts_p_complement = \
            [1 - x if x > 0.0 else 0.0
             for x in category_counts_over_sum_category_counts]

        # less common categories have more weight, sum > 1
        total_pixels_p_complement = \
            [1 - x if x > 0.0 else 0.0
             for x in category_counts_over_total_pixels]

        print(bin_count)
        stat_dict = {
            'total_pixels': total_pixels,
            'category_counts': dict(zip(category_ids, bin_count)),
            'sum_category_counts': sum_category_counts,
            'category_counts_over_sum_category_counts':
                dict(zip(category_ids,
                         category_counts_over_sum_category_counts)),
            'category_counts_over_total_pixels':
                dict(zip(category_ids, category_counts_over_total_pixels)),
            'category_counts_p_complement':
                dict(zip(category_ids, category_counts_p_complement)),
            'total_pixels_p_complement':
                dict(zip(category_ids, total_pixels_p_complement)),
            'ids': ids(),
            'categories': categories()
        }
        print(stat_dict)
        with open(stats_json, 'w') as fjson:
            json.dump(stat_dict, fjson, ensure_ascii=False)


@data_coco.command
def coco_setup(dataset_root, dataset_path, data_prefixes,
               filenames, urls, md5s, annotation_paths,
               image_dirs, seg_mask_output_paths, verbose,
               image_extensions):
    # download the dataset
    coco_download(dataset_path, filenames, dataset_root,
                  urls, md5s, annotation_paths)
    # convert the relevant files to a more useful format
    coco_json_to_segmentation(seg_mask_output_paths, annotation_paths)
    coco_to_pascal_voc_imageset_txt(voc_imageset_txt_paths, image_dirs,
                                    image_extensions)


@data_coco.automain
def main(dataset_root, dataset_path, data_prefixes,
         filenames, urls, md5s, annotation_paths,
         image_dirs, seg_mask_output_paths):
    coco_config()
    coco_setup(data_prefixes, dataset_path, filenames, dataset_root, urls,
               md5s, annotation_paths, image_dirs,
               seg_mask_output_paths)
