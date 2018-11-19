from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from utils import core as flags_core
from utils import utils
from utils import distribution_utils
from model import yolo_model
from model import yolo_run_loop
from data_processor import coco_dataset

import multiprocessing
from time import time
import imgaug


DEFAULT_DATASET_YEAR = '2017'


def define_coco_flags():
    """Key flags for coco."""
    yolo_run_loop.define_yolo_flags()
    flags.adopt_module_key_flags(yolo_run_loop)
    flags_core.set_defaults(data_dir='/home/hume/Deep-learning/dataset/coco',
                            model_dir='/home/hume/Deep-learning/model/yolov3/coco_model/',
                            train_epochs=150,
                            epochs_between_evals=10,
                            batch_size=4,
                            num_classes=1+80,
                            threshold=0.5,
                            confidence_score=0.7,
                            learning_rate=0.001,
                            dtype='fp32',
                            backbone='darknet53',
                            image_size=416,
                            image_channels=3,
                            anchors_path='../data_processor/anchors.txt',
                            data_format='channels_last',
                            image_bytes_as_serving_input=False)
    return flags.FLAGS


class CocoModel(yolo_model.YoloDetection):
    """Model class with appropriate defaults for COCO data."""
    def __init__(self):

        anchors = utils.get_anchors(flags.FLAGS.anchors_path)
        num_anchors = len(anchors)
        anchors = np.array(anchors, dtype=np.float32)

        super(CocoModel, self).__init__(
            image_size=flags.FLAGS.image_size,
            image_channels=flags.FLAGS.image_channels,
            num_classes=flags.FLAGS.num_classes,
            anchors=anchors,
            batch_size=distribution_utils.per_device_batch_size(flags.FLAGS.batch_size,
                                                                flags.FLAGS.num_gpus),
            num_anchors=num_anchors,
            learning_rate=flags.FLAGS.learning_rate,
            backbone=flags.FLAGS.backbone,
            norm=flags.FLAGS.norm,
            threshold=flags.FLAGS.threshold,
            max_num_boxes_per_image=flags.FLAGS.max_num_boxes_per_image,
            confidence_score=flags.FLAGS.confidence_score,
            data_format=flags.FLAGS.data_format,
            dtype=flags_core.get_tf_dtype(flags.FLAGS)
        )


def coco_model_fn(features, labels, mode, params):
    """Model function for coco."""
    features = tf.reshape(features, [-1, params['image_size'], params['image_size'],
                          flags.FLAGS.image_channels])

    learning_rate_fn = yolo_run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'],  batch_denom=32,
        num_images=params['train'], boundary_epochs=[101],
        decay_rates=[1, 0.1], base_lr=params['learning_rate']
    )

    weight_decay = 1e-4

    def loss_filter_fn(_):
        return True

    return yolo_run_loop.yolo_model_fn(
        features=features,
        labels=labels,
        mode=mode,
        model_class=CocoModel,
        weight_decay=weight_decay,
        learning_rate_fn=learning_rate_fn,
        batch_size=params['batch_size'],
        data_format=params['data_format'],
        loss_scale=params['loss_scale'],
        image_size=params['image_size'],
        num_classes=params['num_classes'],
        loss_filter_fn=loss_filter_fn,
        dtype=tf.float32,
        fine_tune=params['fine_tune'],
        anchors=params['anchors'],
        num_anchors=params['num_anchors'],
        max_num_boxes_per_image=params['max_num_boxes_per_image'],
        threshold=params['threshold']
    )


def get_detector_heatmap_each_scale(boxes_data_, best_anchors_, anchors_mask, grid_size, num_classes):

    num_anchors = len(anchors_mask)
    boxes_data_shape = boxes_data_.shape

    best_anchors_mask = np.isin(best_anchors_, anchors_mask, invert=True)
    best_anchors_ = best_anchors_ * 1
    best_anchors_ -= min(anchors_mask)
    best_anchors_[best_anchors_mask] = 0

    boxes_data_mask = np.ones_like(best_anchors_)
    boxes_data_mask[best_anchors_mask] = 0
    boxes_data_mask = np.expand_dims(boxes_data_mask, -1)
    boxes_data_ = boxes_data_ * boxes_data_mask

    i__ = np.floor(boxes_data_[:, 1] * grid_size[0]).astype('int32')
    j = np.floor(boxes_data_[:, 0] * grid_size[1]).astype('int32')

    boxes_data_ = boxes_data_.reshape([-1, boxes_data_.shape[-1]])
    best_anchors_ = best_anchors_.reshape([-1, 1])
    i__ = i__.reshape([-1, 1])
    j = j.reshape([-1, 1])

    classes = boxes_data_[:, -1].reshape([-1]).astype(np.int)
    one_hot_array = np.zeros([boxes_data_.shape[0], num_classes])
    one_hot_array[np.arange(boxes_data_.shape[0]), classes] = 1

    boxes_data_mask = boxes_data_[:, 2] > 0
    boxes_data_[boxes_data_mask, 4] = 1
    boxes_data_ = np.concatenate([boxes_data_, one_hot_array], axis=-1)

    y_true = np.zeros([int(grid_size[0]) * int(grid_size[1]) * num_anchors, 5 + num_classes])

    grid_offset = num_anchors * (grid_size[0] * i__ + j)

    indexing_array = np.array(grid_offset + best_anchors_, dtype=np.int32)
    indexing_array = indexing_array[boxes_data_mask, :]
    indexing_array = indexing_array.reshape([-1])

    y_true[indexing_array, :] = boxes_data_[boxes_data_mask]
    y_true = y_true.reshape(
            [int(grid_size[0]) * int(grid_size[1]) * num_anchors, num_classes + 5])
    boxes_data_ = boxes_data_.reshape([boxes_data_shape[0], -1])

    return y_true, boxes_data_[..., 0:4]


def parse_fn(image_id, dataset, anchors_path, augmentation=None, dtype=np.float32, max_num_boxes_per_image=20,
             image_size=416):

    """Load and return ground truth data for an image (image, mask, bounding boxes)."""

    image = dataset.load_image(image_id)
    # original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=0,
        min_scale=0,
        max_dim=image_size,
        mode='square')

    mask, class_ids = dataset.load_mask(image_id)

    mask = utils.resize_mask(mask, scale, padding, crop)

    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    if mask.shape[-1] > max_num_boxes_per_image:
        ids = np.random.choice(
            np.arange(mask.shape[-1]), max_num_boxes_per_image, replace=False)
        class_ids = class_ids[ids]
        bbox = bbox[ids, :]

    # confs = np.ones((bbox.shape[0], 1), dtype=dtype)
    # bbox = np.concatenate([bbox, confs], axis=-1)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    # active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    # source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    # active_class_ids[source_class_ids] = 1

    # image_meta = utils.compose_image_meta(image_id, original_shape, image.shape,
    #                                       window, scale, active_class_ids)
    # image_meta.astype(dtype)

    # gt_mask = np.zeros((mask.shape[0], mask.shape[1], 20), mask.dtype)
    gt_class_ids = np.zeros(max_num_boxes_per_image, class_ids.dtype)
    gt_bbox = np.zeros((max_num_boxes_per_image, bbox.shape[1]), bbox.dtype)
    # gt_data = np.zeros((max_num_boxes_per_image, bbox.shape[1] + dataset.num_classes), dtype=dtype)

    if class_ids.shape[0] > 0:
        gt_class_ids[:class_ids.shape[0]] = class_ids
        # gt_mask[:, :, :mask.shape[-1]] = mask
        gt_bbox[:bbox.shape[0], :] = bbox

    gt_class_ids = np.expand_dims(gt_class_ids, axis=-1).astype(dtype)

    gt_bbox = np.concatenate([gt_bbox, gt_class_ids], axis=-1)

    anchors = utils.get_anchors(anchors_path)
    anchors = np.array(anchors, dtype=np.float32)

    boxes_yx = (gt_bbox[:, 0:2] + gt_bbox[:, 2:4]) // 2
    boxes_hw = gt_bbox[:, 2:4] - gt_bbox[:, 0:2]

    gt_bbox[:, 0] = boxes_yx[..., 1] / image_size
    gt_bbox[:, 1] = boxes_yx[..., 0] / image_size
    gt_bbox[:, 2] = boxes_hw[..., 1] / image_size
    gt_bbox[:, 3] = boxes_hw[..., 0] / image_size

    hw = np.expand_dims(boxes_hw, -2)
    anchors_broad = np.expand_dims(anchors, 0)

    anchor_maxes = anchors_broad / 2.
    anchor_mins = -anchor_maxes
    box_maxes = hw / 2.
    box_mins = -box_maxes
    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_hw = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]
    box_area = hw[..., 0] * hw[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    best_anchors = np.argmax(iou, axis=-1)

    # TODO: write a function to calculate the stride automatically.
    large_obj_image_size = image_size // 32
    medium_obj_image_size = image_size // 16
    small_obj_image_size = image_size // 8

    large_obj_detectors, large_obj_boxes = get_detector_heatmap_each_scale(gt_bbox,
                                                                           best_anchors_=best_anchors,
                                                                           anchors_mask=[6, 7, 8],
                                                                           grid_size=(large_obj_image_size,
                                                                                      large_obj_image_size),
                                                                           num_classes=dataset.num_classes)

    medium_obj_detectors, medium_obj_boxes = get_detector_heatmap_each_scale(gt_bbox,
                                                                             best_anchors_=best_anchors,
                                                                             anchors_mask=[3, 4, 5],
                                                                             grid_size=(medium_obj_image_size,
                                                                                        medium_obj_image_size),
                                                                             num_classes=dataset.num_classes)

    small_obj_detectors, small_obj_boxes = get_detector_heatmap_each_scale(gt_bbox,
                                                                           best_anchors_=best_anchors,
                                                                           anchors_mask=[0, 1, 2],
                                                                           grid_size=(small_obj_image_size,
                                                                                      small_obj_image_size),
                                                                           num_classes=dataset.num_classes)

    yolo_true_data = np.concatenate([large_obj_detectors, medium_obj_detectors, small_obj_detectors],
                                    axis=0).reshape([-1])
    yolo_true_boxes = np.concatenate([large_obj_boxes, medium_obj_boxes, small_obj_boxes], axis=0).reshape([-1])

    yolo_gt = np.concatenate([yolo_true_data, yolo_true_boxes], axis=-1)

    return image.astype(dtype), yolo_gt.astype(dtype)


def input_fn(data_set, is_training, batch_size, anchors_path, num_epochs=1, augmentation=None,
             dtype=tf.float32, max_num_boxes_per_image=20, image_size=416, datasets_num_private_threads=None,
             num_parallel_batches=1):
    """Input function which provides batches for train or eval.

        Args:
            is_training: A boolean denoting whether the input is for training.
            data_set: A dataset obj.
            batch_size: The number of samples per batch.
            anchors_path: The path of anchors file.
            num_epochs: The number of epochs to repeat the dataset.
            augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
                For example, passing imgaug.augmenters.Fliplr(0.5) flips images
                right/left 50% of the time.
            dtype: Data type to use for images/features
            max_num_boxes_per_image: Max num of boxes per scale.
            image_size: Input image size of yolo model.
            datasets_num_private_threads: Number of private threads for tf.data.
            num_parallel_batches: Number of parallel batches for tf.data.

        Returns:
            A dataset that can be used for iteration.
      """

    id_list = data_set.image_ids
    num_images = data_set.num_images
    # id_list = [info['path'] for info in cocodataset.image_info]
    # ann_list = [info['annotations'] for info in cocodataset.image_info]
    # info_list = list(zip(path_list, ann_list))

    dataset = tf.data.Dataset.from_tensor_slices(id_list)

    return yolo_run_loop.process_dataset(
        dataset=dataset,
        data_set=data_set,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=num_images,
        parse_fn=parse_fn,
        anchors_path=anchors_path,
        num_epochs=num_epochs,
        dtype=dtype,
        max_num_boxes_per_image=max_num_boxes_per_image,
        image_size=image_size,
        datasets_num_private_threads=datasets_num_private_threads,
        augmentation=augmentation,
        num_parallel_batches=num_parallel_batches
    )


def run_coco(flag_obj, is_training):
    dataset = coco_dataset.CocoDataset()

    if is_training:
        subset = 'train'
    else:
        subset = 'val'

    dataset.load_coco(flag_obj.data_dir, subset, DEFAULT_DATASET_YEAR)
    dataset.prepare()

    if is_training:
        augmentation = imgaug.augmenters.Fliplr(0.5)
    else:
        augmentation = None

    yolo_run_loop.yolo_main(
        flag_obj, model_function=coco_model_fn, input_function=input_fn, dataset=dataset,
        augmentation=augmentation)


def test(_):
    tf.enable_eager_execution()
    flag_obj = define_coco_flags()
    cocodataset = coco_dataset.CocoDataset()

    cocodataset.load_coco('/home/hume/Deep-learning/dataset/coco', 'train', DEFAULT_DATASET_YEAR)
    cocodataset.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)

    input_iter = input_fn(cocodataset,
                          is_training=True,
                          batch_size=distribution_utils.per_device_batch_size(flag_obj.batch_size,
                                                                              flags_core.get_num_gpus(flag_obj)),
                          anchors_path=flag_obj.anchors_path,
                          num_epochs=flag_obj.train_epochs,
                          dtype=tf.float32,
                          max_num_boxes_per_image=flag_obj.max_num_boxes_per_image,
                          image_size=flag_obj.image_size,
                          augmentation=augmentation,
                          num_parallel_batches=flag_obj.datasets_num_parallel_batches,
                          datasets_num_private_threads=multiprocessing.cpu_count() - 3
                          )
    coco_iter = input_iter.make_one_shot_iterator()
    starttime = time()
    imgs, y_gt = coco_iter.get_next()
    print('cost {}ms\n'.format((time() - starttime) * 1000))

    print(imgs.shape)
    print(y_gt.shape)


def test_parse(_):
    flag_obj = define_coco_flags()
    cocodataset = coco_dataset.CocoDataset()

    cocodataset.load_coco('/home/hume/Deep-learning/dataset/coco', 'train', DEFAULT_DATASET_YEAR)
    cocodataset.prepare()

    image_id = cocodataset.image_ids[1]

    img, y_gt = parse_fn(image_id, dataset=cocodataset, anchors_path=flag_obj.anchors_path,
                         augmentation=None,
                         dtype=np.float32, max_num_boxes_per_image=20, image_size=416)

    print(img.shape, y_gt.shape)


def main(_):
    flag_obj = define_coco_flags()
    run_coco(flag_obj, is_training=True)


if __name__ == '__main__':
    absl_app.run(main)
