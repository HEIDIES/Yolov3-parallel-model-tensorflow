import tensorflow as tf
from model.darknet import Darknet
from model.yolo_v3 import Yolov3

DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class YoloDetection:
    def __init__(self, image_size, image_channels,
                 anchors,
                 batch_size=16,
                 num_anchors=9,
                 learning_rate=0.001,
                 num_classes=1 + 80,
                 backbone=None,
                 norm='batch',
                 threshold=0.5,
                 max_num_boxes_per_image=20,
                 confidence_score=0.7,
                 dtype=DEFAULT_DTYPE,
                 data_format=None,
                 ):
        self.confidence_score = confidence_score
        self.image_size = image_size
        self.image_channels = image_channels
        self.anchors = anchors
        self.batch_size = batch_size
        self.num_anchors = num_anchors
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.dtype = dtype

        if not data_format:
            data_format = 'channel_first' if tf.test.is_built_with_cuda else 'channel_last'

        self.data_format = data_format

        # self.max_num_boxes_per_image = ((self.image_size // 32) *
        #                                 (self.image_size // 32)
        #                                 + (self.image_size // 16)
        #                                 * (self.image_size // 16)
        #                                 + (self.image_size // 8)
        #                                 * (self.image_size // 8))
        self.max_num_boxes_per_image = max_num_boxes_per_image
        self.backbone = backbone
        self.norm = norm
        self.threshold = threshold
        # self.is_training = tf.placeholder_with_default(True, [], name='is_training')
        self.num_anchors_per_detector = self.num_anchors // 3
        self.num_detectors_per_image = self.num_anchors_per_detector * ((self.image_size // 32) *
                                                                        (self.image_size // 32)
                                                                        + (self.image_size // 16)
                                                                        * (self.image_size // 16)
                                                                        + (self.image_size // 8)
                                                                        * (self.image_size // 8))

        # self.X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
        # with tf.variable_scope('labels', c)
        self.Y_true_data = tf.placeholder(dtype=tf.float32,
                                          shape=[None, self.num_detectors_per_image, self.num_classes + 5])

        self.Y_true_boxes = tf.placeholder(dtype=tf.float32,
                                           shape=[None,
                                                  self.max_num_boxes_per_image * self.num_anchors_per_detector, 4])

        self.darknet = Darknet('darknet', backbone=self.backbone,
                               norm=self.norm, dtype=self.dtype)

        self.yolo_v3 = Yolov3('yolo_v3',
                              num_classes=self.num_classes, anchors=self.anchors,
                              norm=self.norm, image_size=self.image_size,
                              dtype=self.dtype)

    def _custom_dtype_getter(self, getter, name, shape=None,
                             *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary.

        This function is a custom getter. A custom getter is a function with the
        same signature as tf.get_variable, except it has an additional getter
        parameter. Custom getters can be passed as the `custom_getter` parameter of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to change
        the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would have
        been called if no custom getter was used. Custom getters typically get a
        variable with `getter`, then modify it in some way.

        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.

        Args:
            getter: The underlying variable getter, that has the same signature as
                tf.get_variable and returns a variable.
            name: The name of the variable to get.
            shape: The shape of the variable to get.
            *args: Additional arguments to pass unmodified to getter.
            **kwargs: Additional keyword arguments to pass unmodified to getter.

        Returns:
            A variable which is cast to fp16 if necessary.
        """

        if self.dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=self.dtype, name=name + '_cast')
        else:
            return getter(name, shape, self.dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Returns a variable scope that the model should be created under.

        If self.dtype is a castable type, model variable will be created in fp32
        then cast to self.dtype before being used.

        Returns:
            A variable scope for the model.
        """

        return tf.variable_scope('x_input',
                                 custom_getter=self._custom_dtype_getter)

    def yolo_v3_optimizer(self, yolo_loss):
        def make_optimizer(loss, variables, name='Adam'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = self.learning_rate
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, name=name).
                minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step
        trainable_var_list = tf.trainable_variables()
        last_layer_var_list = [i for i in trainable_var_list if
                               i.shape[-1] == (5 + self.num_classes) * self.num_anchors_per_detector]
        last_layer_optimizer = make_optimizer(yolo_loss, last_layer_var_list)
        yolo_optimizer = make_optimizer(yolo_loss, trainable_var_list)

        return last_layer_optimizer, yolo_optimizer

    def coords_to_boxes(self, yolo_boxes_out, is_pred=True):

        x_pred, y_pred, w_pred, h_pred, confs_pred, classes_pred = tf.split(yolo_boxes_out,
                                                                            [1, 1, 1, 1, 1,
                                                                             self.num_classes],
                                                                            axis=-1)
        if is_pred:
            x_pred = x_pred / self.image_size
            y_pred = y_pred / self.image_size
            w_pred = w_pred / self.image_size
            h_pred = h_pred / self.image_size

        up_left_x = x_pred - w_pred / 2.0
        up_left_y = y_pred - h_pred / 2.0
        down_right_x = x_pred + w_pred / 2.0
        down_right_y = y_pred + h_pred / 2.0

        detections = tf.concat([up_left_x, up_left_y, down_right_x, down_right_y, confs_pred, classes_pred], axis=-1)

        return detections

    def draw_boxes(self, yolo_out):
        yolo_boxes_out = yolo_out[:3]

        yolo_boxes_out_large_obj = yolo_boxes_out[0]
        yolo_boxes_out_medium_obj = yolo_boxes_out[1]
        yolo_boxes_out_small_obj = yolo_boxes_out[2]

        yolo_boxes_out_large_obj = tf.reshape(yolo_boxes_out_large_obj, [self.batch_size, -1, self.num_classes + 5])
        yolo_boxes_out_medium_obj = tf.reshape(yolo_boxes_out_medium_obj, [self.batch_size, -1,
                                                                           self.num_classes + 5])
        yolo_boxes_out_small_obj = tf.reshape(yolo_boxes_out_small_obj, [self.batch_size, -1, self.num_classes + 5])

        yolo_boxes_out = tf.concat([yolo_boxes_out_large_obj, yolo_boxes_out_medium_obj, yolo_boxes_out_small_obj],
                                   axis=1)

        detections = self.coords_to_boxes(yolo_boxes_out)
        confs_pred = detections[:, :, 4]

        conf_mask = tf.cast(tf.expand_dims(tf.greater(confs_pred,
                                                      tf.ones_like(confs_pred) * self.confidence_score), -1),
                            tf.float32)
        predictions = detections * conf_mask

        pred_images = []

        for i in range(self.batch_size):
            conf_pred = predictions[i, :, 4]
            boxes_pred = predictions[i, :, 0: 4]
            up_left_x, up_left_y, down_right_x, down_right_y = tf.split(boxes_pred, [1, 1, 1, 1], axis=-1)

            boxes_pred = tf.concat([up_left_y, up_left_x, down_right_y, down_right_x], axis=-1)

            top_k_scores, top_k_indices = tf.nn.top_k(conf_pred, k=60)
            boxes_pred = tf.gather(boxes_pred, top_k_indices)

            desired_indices = tf.image.non_max_suppression(boxes_pred, top_k_scores, max_output_size=6)

            desired_boxes = tf.gather(boxes_pred, desired_indices)

            desired_boxes = tf.expand_dims(desired_boxes, axis=0)

            desired_boxes = tf.where(tf.less(desired_boxes, tf.zeros_like(desired_boxes)),
                                     tf.zeros_like(desired_boxes), desired_boxes)

            desired_boxes = tf.where(tf.greater(desired_boxes, tf.ones_like(desired_boxes)),
                                     tf.ones_like(desired_boxes), desired_boxes)

            pred_images.append(tf.image.draw_bounding_boxes(tf.expand_dims(self.X[i, :, :, :], axis=0), desired_boxes))

        pred_images = tf.concat(pred_images, axis=0)

        return pred_images

    def draw_true_boxes(self):

        yolo_true_data = self.coords_to_boxes(self.Y_true_data, is_pred=False)

        yolo_true_images = []

        for i in range(self.batch_size):
            conf_true = yolo_true_data[i, :, 4]
            boxes_true = yolo_true_data[i, :, 0: 4]

            up_left_x, up_left_y, down_right_x, down_right_y = tf.split(boxes_true, [1, 1, 1, 1], axis=-1)

            boxes_true = tf.concat([up_left_y, up_left_x, down_right_y, down_right_x], axis=-1)

            top_k_scores, top_k_indices = tf.nn.top_k(conf_true, k=20)

            boxes_true = tf.gather(boxes_true, top_k_indices)

            boxes_true = tf.expand_dims(boxes_true, axis=0)

            yolo_true_images.append(tf.image.draw_bounding_boxes(tf.expand_dims(self.X[i, :, :, :], axis=0),
                                                                 boxes_true))

        yolo_true_images = tf.concat(yolo_true_images, axis=0)

        return yolo_true_images

    def __call__(self, ipt, is_training):

        self.is_training = is_training

        with self._model_variable_scope():
            if self.data_format == 'channels_first':
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                ipt = tf.transpose(ipt, [0, 3, 1, 2])

            self.X = ipt

        dark_out, dark_route_1, dark_route_2 = self.darknet(self.X, self.is_training)

        yolo_out = self.yolo_v3(dark_out, dark_route_1, dark_route_2, self.is_training)

        origin_images = self.draw_true_boxes()

        tf.summary.image('origin_images', origin_images)

        pred_images = self.draw_boxes(yolo_out)

        tf.summary.image('prediction_images', pred_images)

        return yolo_out
