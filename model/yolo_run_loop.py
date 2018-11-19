from utils import core as flags_core
from utils import model_helpers
from utils import distribution_utils
from utils import utils
from model import yolo_model

import numpy as np
import math
import functools

import multiprocessing
import os

from absl import flags
import tensorflow as tf
from tensorflow.contrib.data.python.ops import threadpool


DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


def define_yolo_flags():
    flags_core.define_base()
    flags_core.define_performance(num_parallel_calls=True,
                                  tf_gpu_thread_mode=True,
                                  datasets_num_private_threads=True,
                                  datasets_num_parallel_batches=True)
    flags_core.define_image()
    flags.adopt_module_key_flags(flags_core)

    flags.DEFINE_enum(
        name='backbone', short_name='bb', default='darknet53',
        enum_values=['darknet53'],
        help='Backbone of yolo. (currently provide darknet53 only).')

    flags.DEFINE_enum(
        name='fine_tune', short_name='ft', default='all',
        enum_values=['all', 'yolo_body', 'yolo_head'],
        help='If True do not train any parameters except for the final layer.')

    flags.DEFINE_string(
        name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
        help='If not None initialize all the network except the final layer with '
             'these values')

    flags.DEFINE_boolean(
        name='eval_only', default=False,
        help='Skip training and only perform evaluation on '
             'the latest checkpoint.')

    flags.DEFINE_boolean(
        name='image_bytes_as_serving_input', default=False,
        help='If True exports savedmodel with serving signature that accepts '
             'JPEG image bytes instead of a fixed size [HxWxC] tensor that '
             'represents the image. The former is easier to use for serving at '
             'the expense of image resize/cropping being done as part of model '
             'inference.')

    flags.DEFINE_string(
        name='norm', default='batch',
        help='Weather and how normalization method to be applied.'
    )

    flags.DEFINE_float(
        name='learning_rate', short_name='lr', default=0.001,
        help='Learning rate for gradient descent.'
    )

    flags.DEFINE_integer(
        name='num_classes', short_name='nc', default=None,
        help='Number of classes for dataset.'
    )

    flags.DEFINE_float(
        name='threshold', default=0.5,
        help='IOU threshold for detection, bounding box '
             'with iou less than this value will be treated'
             'as BG.'
    )

    flags.DEFINE_float(
        name='confidence_score', default=0.7,
        help='Detected obj with confidence less than this value '
             'will be ignored while display.'
    )

    flags.DEFINE_integer(
        name='image_size', default=416,
        help='Size of input image.'
    )

    flags.DEFINE_integer(
        name='image_channels', default=3,
        help='Channels of input image.'
    )

    flags.DEFINE_integer(
        name='max_num_boxes_per_image', short_name='mnbpi', default=20,
        help='Maximum number of objects for detector to detect per image.'
    )

    flags.DEFINE_string(
        name='anchors_path', short_name='ap', default='data_processor/anchors.txt',
        help='The path that points towards where '
             'the anchor values for the model are stored.'
    )


def learning_rate_with_decay(batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
                             base_lr=0.1, warmup=False):
    """Get a learning rate that decays step-wise as training progresses.

        Args:
            batch_size: the number of examples processed in each training batch.
            batch_denom: this value will be used to scale the base learning rate.
                `0.1 * batch size` is divided by this number, such that when
                batch_denom == batch_size, the initial learning rate will be 0.1.
            num_images: total number of images that will be used for training.
            boundary_epochs: list of ints representing the epochs at which we
                decay the learning rate.
            decay_rates: list of floats representing the decay rates to be used
                for scaling the learning rate. It should have one more element
                than `boundary_epochs`, and all elements should have the same type.
            base_lr: Initial learning rate scaled based on batch_denom.
            warmup: Run a 5 epoch warmup to the initial lr.
        Returns:
            Returns a function that takes a single argument - the number of batches
            trained so far (global_step)- and returns the learning rate to be used
            for training the next batch.
    """
    initial_learning_rate = base_lr * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        """Builds scaled learning rate function with 5 epoch warm up."""
        lr = tf.train.piecewise_constant(global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (
                    initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
            return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
        return lr

    return learning_rate_fn


def coords_to_boxes(yolo_boxes_out, num_classes):

    x_pred, y_pred, w_pred, h_pred, confs_pred, classes_pred = tf.split(yolo_boxes_out,
                                                                        [1, 1, 1, 1, 1,
                                                                         num_classes],
                                                                        axis=-1)

    up_left_x = x_pred - w_pred / 2.0
    up_left_y = y_pred - h_pred / 2.0
    down_right_x = x_pred + w_pred / 2.0
    down_right_y = y_pred + h_pred / 2.0

    detections = tf.concat([up_left_x, up_left_y, down_right_x, down_right_y, confs_pred, classes_pred], axis=-1)

    return detections


def draw_true_boxes(y_true_data, num_classes, batch_size, features):

    yolo_true_data = coords_to_boxes(y_true_data, num_classes)

    yolo_true_images = []

    for i in range(batch_size):
        conf_true = yolo_true_data[i, :, 4]
        boxes_true = yolo_true_data[i, :, 0: 4]

        up_left_x, up_left_y, down_right_x, down_right_y = tf.split(boxes_true, [1, 1, 1, 1], axis=-1)

        boxes_true = tf.concat([up_left_y, up_left_x, down_right_y, down_right_x], axis=-1)

        top_k_scores, top_k_indices = tf.nn.top_k(conf_true, k=20)

        boxes_true = tf.gather(boxes_true, top_k_indices)

        boxes_true = tf.expand_dims(boxes_true, axis=0)

        yolo_true_images.append(tf.image.draw_bounding_boxes(tf.expand_dims(features[i, :, :, :], axis=0),
                                                             boxes_true))

    yolo_true_images = tf.concat(yolo_true_images, axis=0)

    return yolo_true_images


def yolo_model_fn(features, labels, mode, model_class,
                  weight_decay, learning_rate_fn, batch_size, num_anchors,
                  data_format, loss_scale, image_size, num_classes,
                  anchors, max_num_boxes_per_image, threshold,
                  loss_filter_fn=None, dtype=yolo_model.DEFAULT_DTYPE,
                  fine_tune=False):

    def yolo_v3_loss(yolo_out, y_gt):

        def yolo_loss_for_each_scale(yolo_layer_outputs, conv_layer_outputs,
                                     yolo_true, yolo_true_boxes, ignore_thresh, anchors,
                                     num_classes=3, h=416, w=416, batch_size=16):

            def iou(yolo_out_pred, yolo_true_boxes_, shape_, batch_size_=16):
                yolo_true_boxes_ = tf.reshape(yolo_true_boxes_, [batch_size_, -1, 4])
                yolo_true_boxes_ = tf.expand_dims(yolo_true_boxes_, axis=1)
                true_coords_xy = yolo_true_boxes_[:, :, :, 0: 2]
                true_coords_wh = yolo_true_boxes_[:, :, :, 2: 4]
                true_up_left = true_coords_xy - true_coords_wh * 0.5
                true_down_right = true_coords_xy + true_coords_wh * 0.5

                true_area = true_coords_wh[:, :, :, 0] * true_coords_wh[:, :, :, 1]

                yolo_out_pred = tf.reshape(yolo_out_pred, [-1, shape_[1] * shape_[2] * shape_[3], 4])
                yolo_out_pred = tf.expand_dims(yolo_out_pred, axis=-2)
                pred_coords_xy = yolo_out_pred[:, :, :, 0: 2]
                pred_coords_wh = yolo_out_pred[:, :, :, 2: 4]
                pred_up_left = pred_coords_wh - pred_coords_wh * 0.5
                pred_down_right = pred_coords_xy + pred_coords_wh * 0.5

                pred_area = pred_coords_wh[:, :, :, 0] * pred_coords_wh[:, :, :, 1]

                intersects_up_left = tf.maximum(true_up_left, pred_up_left)
                intersects_down_right = tf.minimum(true_down_right, pred_down_right)

                intersects_wh = tf.maximum(intersects_down_right - intersects_up_left, 0.0)

                intersects_area = intersects_wh[:, :, :, 0] * intersects_wh[:, :, :, 1]

                iou_ = intersects_area / (pred_area + true_area - intersects_area)

                return tf.reduce_max(iou_, axis=-1)

            num_anchors = len(anchors)
            shape = yolo_layer_outputs.get_shape().as_list()

            yolo_out_pred_rela = yolo_layer_outputs[..., 0: 4] / tf.cast(tf.constant([w, h, w, h]), tf.float32)

            conv_layer_outputs = tf.reshape(conv_layer_outputs, [-1, shape[1], shape[2], shape[3], shape[4]])
            pred_conf = conv_layer_outputs[..., 4: 5]
            # pred_class = conv_layer_outputs[..., 5:]

            yolo_true = tf.reshape(yolo_true, [-1, shape[1], shape[2], shape[3], shape[4]])
            percent_x, percent_y, percent_w, percent_h, obj_mask, classes = tf.split(yolo_true,
                                                                                     [1, 1, 1, 1, 1, num_classes],
                                                                                     axis=-1)

            clustroid_x = tf.tile(tf.reshape(tf.range(shape[2], dtype=tf.float32), [1, -1, 1, 1]), [shape[2], 1, 1, 1])
            clustroid_y = tf.tile(tf.reshape(tf.range(shape[1], dtype=tf.float32), [-1, 1, 1, 1]), [1, shape[1], 1, 1])
            converted_x_true = percent_x * shape[2] - clustroid_x
            converted_y_true = percent_y * shape[1] - clustroid_y

            anchors = tf.constant(anchors, dtype=tf.float32)
            anchors_w = tf.reshape(anchors[:, 0], [1, 1, 1, num_anchors, 1])
            anchors_h = tf.reshape(anchors[:, 1], [1, 1, 1, num_anchors, 1])

            converted_w_true = tf.log((percent_w / anchors_w) * w)
            converted_h_true = tf.log((percent_h / anchors_h) * h)

            yolo_raw_box_true = tf.concat([converted_x_true, converted_y_true, converted_w_true, converted_h_true],
                                          axis=-1)
            yolo_raw_box_true = tf.where(tf.is_inf(yolo_raw_box_true),
                                         tf.zeros_like(yolo_raw_box_true),
                                         yolo_raw_box_true)

            box_loss_scale = 2 - yolo_true[..., 2: 3] * yolo_true[..., 3: 4]

            coords_xy_loss = (tf.nn.sigmoid_cross_entropy_with_logits(labels=yolo_raw_box_true[..., 0: 2],
                                                                      logits=conv_layer_outputs[..., 0: 2])
                              * obj_mask * box_loss_scale)
            coords_xy_loss = tf.reduce_sum(coords_xy_loss)

            coords_wh_loss = tf.square(yolo_raw_box_true[..., 2: 4]
                                       - conv_layer_outputs[..., 2: 4]) * 0.5 * obj_mask * box_loss_scale

            coords_wh_loss = tf.reduce_sum(coords_wh_loss)

            coords_loss = coords_xy_loss + coords_wh_loss

            box_iou = iou(yolo_out_pred_rela, yolo_true_boxes, shape, batch_size)

            ignore_mask = tf.cast(tf.less(box_iou, ignore_thresh * tf.ones_like(box_iou)), tf.float32)
            ignore_mask = tf.reshape(ignore_mask, [-1, shape[1], shape[2], num_anchors])
            ignore_mask = tf.expand_dims(ignore_mask, -1)

            back_loss = ((1 - obj_mask)
                         * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=pred_conf) * ignore_mask)
            back_loss = tf.reduce_sum(back_loss)

            fore_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=pred_conf)

            fore_loss = tf.reduce_sum(fore_loss)

            conf_loss = back_loss + fore_loss

            # cls_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=classes, logits=pred_class)
            # cls_loss = tf.reduce_sum(cls_loss)

            return coords_loss + conf_loss

        num_anchors_per_detector = len(anchors) // 3

        num_large_detectors = int((image_size / 32) * (image_size / 32) * num_anchors_per_detector)
        num_medium_detectors = int((image_size / 16) * (image_size / 16) * num_anchors_per_detector)

        y_true_data = y_gt[:, :-max_num_boxes_per_image * num_anchors_per_detector * 4]
        y_true_boxes = y_gt[:, -max_num_boxes_per_image * num_anchors_per_detector * 4:]

        y_true_data = tf.reshape(y_true_data, [batch_size, -1, 5 + num_classes])
        y_true_boxes = tf.reshape(y_true_boxes, [batch_size, max_num_boxes_per_image * num_anchors_per_detector, 4])

        origin_images = draw_true_boxes(y_true_data, num_classes, batch_size, features)
        tf.summary.image('origin_images', origin_images)

        large_yolo_true_raw = y_true_data[:, :num_large_detectors, :]
        medium_yolo_true_raw = y_true_data[:, num_large_detectors: num_medium_detectors + num_large_detectors, :]
        small_yolo_true_raw = y_true_data[:, num_medium_detectors + num_large_detectors:, :]

        large_yolo_true_boxes = y_true_boxes[:, :max_num_boxes_per_image, :]
        medium_yolo_true_boxes = y_true_boxes[:, max_num_boxes_per_image: 2*max_num_boxes_per_image, :]
        small_yolo_true_boxes = y_true_boxes[:, 2*max_num_boxes_per_image:, :]

        yolo_layer_outputs_ = yolo_out[:3]
        conv_layer_outputs_ = yolo_out[3:]

        large_yolo_pred_boxes = yolo_layer_outputs_[0]
        medium_yolo_pred_boxes = yolo_layer_outputs_[1]
        small_yolo_pred_boxes = yolo_layer_outputs_[2]

        large_yolo_pred_raw = conv_layer_outputs_[0]
        medium_yolo_pred_raw = conv_layer_outputs_[1]
        small_yolo_pred_raw = conv_layer_outputs_[2]

        large_obj_loss = yolo_loss_for_each_scale(large_yolo_pred_boxes, large_yolo_pred_raw,
                                                  large_yolo_true_raw, large_yolo_true_boxes,
                                                  ignore_thresh=threshold,
                                                  anchors=anchors[num_anchors_per_detector*2:],
                                                  num_classes=num_classes, h=image_size,
                                                  w=image_size,
                                                  batch_size=batch_size
                                                  )

        medium_obj_loss = yolo_loss_for_each_scale(medium_yolo_pred_boxes, medium_yolo_pred_raw,
                                                   medium_yolo_true_raw, medium_yolo_true_boxes,
                                                   ignore_thresh=threshold,
                                                   anchors=anchors[num_anchors_per_detector:
                                                                   2*num_anchors_per_detector],
                                                   num_classes=num_classes, h=image_size,
                                                   w=image_size,
                                                   batch_size=batch_size
                                                   )

        small_obj_loss = yolo_loss_for_each_scale(small_yolo_pred_boxes, small_yolo_pred_raw,
                                                  small_yolo_true_raw, small_yolo_true_boxes,
                                                  ignore_thresh=threshold,
                                                  anchors=anchors[:num_anchors_per_detector],
                                                  num_classes=num_classes, h=image_size,
                                                  w=image_size,
                                                  batch_size=batch_size
                                                  )

        return (large_obj_loss + medium_obj_loss + small_obj_loss) / batch_size

    assert features.dtype == dtype

    model = model_class()
    yolo_out_ = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    yolo_out = [tf.cast(y_out, tf.float32) for y_out in yolo_out_]

    predictions = {
        'large_obj_box_detections': yolo_out[0],
        'medium_obj_box_detections': yolo_out[1],
        'small_obj_box_detections': yolo_out[2]
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    yolo_loss = yolo_v3_loss(yolo_out, labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(yolo_loss, name='yolo_loss')
    tf.summary.scalar('yolo_loss', yolo_loss)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = yolo_loss + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        )

        def _yolo_body_grad_filter(gvs):
            """Only apply gradient updates to the yolo body layer.

            This function is used for fine tuning.

            Args:
                gvs: list of tuples with gradients and variable info
            Returns:
                filtered gradients so that only the dense layer remains
            """
            return [(g, v) for g, v in gvs if 'yolo_v3' in v.name]

        def _yolo_head_grad_filter(gvs):
            """Only apply gradient updates to the yolo head layer.
            This function is used for fine tuning.
            Args:
                gvs: list of tuples with gradients and variable info
            Returns:
                filtered gradients so that only the dense layer remains
            """
            if data_format == 'NHWC':
                return [(g, v) for g, v in gvs if v.shape[-1] == (5+num_classes) * (num_anchors // 3)]
            else:
                return [(g, v) for g, v in gvs if v.shape[1] == (5+num_classes) * (num_anchors // 3)]

        if loss_scale != 1:
            # When computing fp16 gradients, often intermediate tensor values are
            # so small, they underflow to 0. To avoid this, we multiply the loss by
            # loss_scale to make these tensor values loss_scale times bigger.
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            if fine_tune == 'yolo_body':
                scaled_grad_vars = _yolo_body_grad_filter(scaled_grad_vars)
            elif fine_tune == 'yolo_head':
                scaled_grad_vars = _yolo_head_grad_filter(scaled_grad_vars)

            # Once the gradient computation is complete we can scale the gradients
            # back to the correct scale before passing them to the optimizer.
            unscaled_grad_vars = [(grad / loss_scale, var)
                                  for grad, var in scaled_grad_vars]
            minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)

        else:
            grad_vars = optimizer.compute_gradients(loss)
            if fine_tune == 'yolo_body':
                grad_vars = _yolo_body_grad_filter(grad_vars)
            elif fine_tune == 'yolo_head':
                grad_vars = _yolo_head_grad_filter(grad_vars)
            minimize_op = optimizer.apply_gradients(grad_vars, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


def override_flags_and_set_envars_for_gpu_thread_pool(flags_obj):
    """Override flags and set env_vars for performance.

    These settings exist to test the difference between using stock settings
    and manual tuning. It also shows some of the ENV_VARS that can be tweaked to
    squeeze a few extra examples per second.  These settings are defaulted to the
    current platform of interest, which changes over time.

    On systems with small numbers of cpu cores, e.g. under 8 logical cores,
    setting up a gpu thread pool with `tf_gpu_thread_mode=gpu_private` may perform
    poorly.

    Args:
        flags_obj: Current flags, which will be adjusted possibly overriding
        what has been set by the user on the command-line.
    """
    cpu_count = multiprocessing.cpu_count()
    tf.logging.info('Logical CPU cores: %s', cpu_count)

    # Sets up thread pool for each GPU for op scheduling.
    per_gpu_thread_count = 1
    total_gpu_thread_count = per_gpu_thread_count * flags_obj.num_gpus
    os.environ['TF_GPU_THREAD_MODE'] = flags_obj.tf_gpu_thread_mode
    os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    tf.logging.info('TF_GPU_THREAD_COUNT: %s', os.environ['TF_GPU_THREAD_COUNT'])
    tf.logging.info('TF_GPU_THREAD_MODE: %s', os.environ['TF_GPU_THREAD_MODE'])

    # Reduces general thread pool by number of threads used for GPU pool.
    main_thread_count = cpu_count - total_gpu_thread_count
    flags_obj.inter_op_parallelism_threads = main_thread_count

    # Sets thread count for tf.data. Logical cores minus threads assign to the
    # private GPU pool along with 2 thread per GPU for event monitoring and
    # sending / receiving tensors.
    num_monitoring_threads = 2 * flags_obj.num_gpus
    flags_obj.datasets_num_private_threads = (cpu_count - total_gpu_thread_count
                                              - num_monitoring_threads)


def yolo_main(flags_obj, model_function, input_function, dataset, augmentation):
    """Shared main loop for yolo Models.

    Args:
        flags_obj: An object containing parsed flags. See define_yolo_flags()
            for details.
        model_function: the function that instantiates the Model and builds the
            ops for train/eval. This will be passed directly into the estimator.
        input_function: the function that processes the dataset and returns a
            dataset that the estimator can train on. This will be wrapped with
            all the relevant flags for running and passed to estimator.
        dataset: A dataset for training and evaluation.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.
      """

    model_helpers.apply_clean(flags_obj)

    # Ensures flag override logic is only executed if explicitly triggered.
    if flags_obj.tf_gpu_thread_mode:
        override_flags_and_set_envars_for_gpu_thread_pool(flags_obj)

    # Creates session config. allow_soft_placement = True, is required for
    # multi-GPU and is not harmful for other modes.
    session_config = tf.ConfigProto(
        log_device_placement=True,
        inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
        intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
        allow_soft_placement=True)

    session_config.gpu_options.allow_growth = True

    distribution_strategy = distribution_utils.get_distribution_strategy(
        flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        session_config=session_config,
        save_checkpoints_secs=60 * 60 * 24)

    # Initializes model with all but the dense layer from pretrained ResNet.
    if flags_obj.pretrained_model_checkpoint_path is not None:
        warm_start_settings = tf.estimator.WarmStartSettings(
            flags_obj.pretrained_model_checkpoint_path,
            vars_to_warm_start='^(?!.*dense)')
    else:
        warm_start_settings = None

    anchors = np.array(utils.get_anchors(flags_obj.anchors_path))

    detector = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
        warm_start_from=warm_start_settings, params={
            'num_classes': flags_obj.num_classes,
            'data_format': flags_obj.data_format,
            'batch_size': flags_obj.batch_size,
            'image_size': int(flags_obj.image_size),
            'loss_scale': flags_core.get_loss_scale(flags_obj),
            'dtype': flags_core.get_tf_dtype(flags_obj),
            'fine_tune': flags_obj.fine_tune,
            'anchors': anchors,
            'num_anchors': len(anchors),
            'max_num_boxes_per_image': flags_obj.max_num_boxes_per_image,
            'threshold': flags_obj.threshold,
            'train': dataset.num_images,
            'learning_rate': flags_obj.learning_rate
        })

    # if flags_obj.use_synthetic_data:
    #     dataset_name = dataset_name + '-synthetic'

    def input_fn_train(num_epochs):
        return input_function(
            data_set=dataset,
            is_training=True,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            anchors_path=flags_obj.anchors_path,
            num_epochs=num_epochs,
            augmentation=augmentation,
            dtype=tf.float32,
            max_num_boxes_per_image=flags_obj.max_num_boxes_per_image,
            image_size=flags_obj.image_size,
            datasets_num_private_threads=flags_obj.datasets_num_private_threads,
            num_parallel_batches=flags_obj.datasets_num_parallel_batches)

    '''
    def input_fn_eval():
        return input_function(
            is_training=False,
            data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=1,
            dtype=flags_core.get_tf_dtype(flags_obj))
            '''

    if flags_obj.eval_only or not flags_obj.train_epochs:
        # If --eval_only is set, perform a single loop with zero train epochs.
        schedule, n_loops = [0], 1
    else:
        # Compute the number of times to loop while training. All but the last
        # pass will train for `epochs_between_evals` epochs, while the last will
        # train for the number needed to reach `training_epochs`. For instance if
        #   train_epochs = 25 and epochs_between_evals = 10
        # schedule will be set to [10, 10, 5]. That is to say, the loop will:
        #   Train for 10 epochs and then evaluate.
        #   Train for another 10 epochs and then evaluate.
        #   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
        n_loops = math.ceil(flags_obj.train_epochs / flags_obj.epochs_between_evals)
        schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
        schedule[-1] = flags_obj.train_epochs - sum(schedule[:-1])  # over counting.

    for cycle_index, num_train_epochs in enumerate(schedule):
        tf.logging.info('Starting cycle: %d/%d', cycle_index, int(n_loops))

        if num_train_epochs:
            detector.train(input_fn=lambda: input_fn_train(num_train_epochs),
                           max_steps=flags_obj.max_train_steps)
    '''
    if flags_obj.export_dir is not None:
        # Exports a saved model for the given classifier.
        export_dtype = flags_core.get_tf_dtype(flags_obj)
        if flags_obj.image_bytes_as_serving_input:
            input_receiver_fn = functools.partial(
                image_bytes_serving_input_fn, shape, dtype=export_dtype)
        else:
            input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
                shape, batch_size=flags_obj.batch_size, dtype=export_dtype)
        detector.export_savedmodel(flags_obj.export_dir, input_receiver_fn,
                                   strip_default_attrs=True)
                                   '''


def process_dataset(dataset,
                    data_set,
                    is_training,
                    batch_size,
                    shuffle_buffer,
                    parse_fn,
                    anchors_path,
                    num_epochs=1,
                    dtype=tf.float32,
                    max_num_boxes_per_image=20,
                    image_size=416,
                    datasets_num_private_threads=None,
                    augmentation=None,
                    num_parallel_batches=1):
    """Given a Dataset with raw records, return an iterator over the records.

        Args:
            dataset: A Dataset representing raw records
            data_set: A dataset obj.
            is_training: A boolean denoting whether the input is for training.
            batch_size: The number of samples per batch.
            shuffle_buffer: The buffer size to use when shuffling records. A larger
                value results in better randomness, but smaller values reduce startup
                time and use less memory.
            parse_fn: A function that takes a raw record and returns the
                corresponding (image, label) pair.
            anchors_path: Path to the anchors file.
            num_epochs: The number of epochs to repeat the dataset.
            dtype: Data type to use for images/features.
            max_num_boxes_per_image: Max num boxes per scale.
            image_size: Input image size for yolo.
            datasets_num_private_threads: Number of threads for a private
                threadpool created for all datasets computation.
            augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
                For example, passing imgaug.augmenters.Fliplr(0.5) flips images
                right/left 50% of the time.
            num_parallel_batches: Number of parallel batches for tf.data.

        Returns:
            Dataset of (image, label) pairs ready for iteration.
      """
    # Prefetches a batch at a time to smooth out the time taken to load input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size * 10)

    if is_training:
        # Shuffles records before repeating to respect epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)

    if dtype == DEFAULT_DTYPE:
        _dtype = np.float32
    else:
        _dtype = np.float16

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda img_id:
                tf.py_func(
                    func=functools.partial(parse_fn, dataset=data_set, augmentation=augmentation, dtype=_dtype,
                                           anchors_path=anchors_path, max_num_boxes_per_image=max_num_boxes_per_image,
                                           image_size=image_size),
                    inp=[img_id],
                    Tout=[dtype, dtype]
                ),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=False
        )
    )

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    # dataset = dataset.prefetch(buffer_size=200)

    # Defines a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
        tf.logging.info('datasets_num_private_threads: %s',
                        datasets_num_private_threads)
        dataset = threadpool.override_threadpool(
            dataset,
            threadpool.PrivateThreadPool(
                datasets_num_private_threads,
                display_name='input_pipeline_thread_pool'
            )
        )

    return dataset
