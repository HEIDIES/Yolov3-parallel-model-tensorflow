import tensorflow as tf
from model import layer

DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Yolov3:
    def __init__(self, name, num_classes, anchors, norm='batch', image_size=416,
                 dtype=DEFAULT_DTYPE):
        self.name = name
        self.num_classes = num_classes
        self.norm = norm
        self.reuse = False
        self.anchors = anchors
        self.image_size = image_size

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.dtype = dtype

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
            var = getter(name, shape, *args, **kwargs)
            return tf.cast(var, dtype=self.dtype, name=name + '_cast')
        else:
            return getter(name, shape, *args, **kwargs)

    def _model_variable_scope(self):
        """Returns a variable scope that the model should be created under.

        If self.dtype is a castable type, model variable will be created in fp32
        then cast to self.dtype before being used.

        Returns:
            A variable scope for the model.
        """

        return tf.variable_scope(self.name,
                                 custom_getter=self._custom_dtype_getter)

    def __call__(self, dark_out, dark_route_1, dark_route_2, is_training):
        self.is_training = is_training
        with self._model_variable_scope():
            large_obj_raw_detections, yolo_route = layer.yolo_1(dark_out, num_classes=self.num_classes,
                                                                reuse=self.reuse, is_training=self.is_training,
                                                                norm=self.norm)

            medium_obj_raw_detections, yolo_route = layer.yolo_2(yolo_route, dark_route_2,
                                                                 num_classes=self.num_classes,
                                                                 reuse=self.reuse,
                                                                 is_training=self.is_training,
                                                                 norm=self.norm)

            small_obj_raw_detections = layer.yolo_3(yolo_route, dark_route_1,
                                                    num_classes=self.num_classes,
                                                    reuse=self.reuse,
                                                    is_training=self.is_training,
                                                    norm=self.norm)

            large_obj_box_detections = layer.yolo_layer(large_obj_raw_detections, name='yolo_0',
                                                        anchors=self.anchors[6:9],
                                                        num_classes=self.num_classes,
                                                        image_size=self.image_size)

            medium_obj_box_detections = layer.yolo_layer(medium_obj_raw_detections, name='yolo_1',
                                                         anchors=self.anchors[3:6],
                                                         num_classes=self.num_classes,
                                                         image_size=self.image_size)

            small_obj_box_detections = layer.yolo_layer(small_obj_raw_detections, name='yolo_2',
                                                        anchors=self.anchors[0:3],
                                                        num_classes=self.num_classes,
                                                        image_size=self.image_size)

        self.reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.is_training:
            with tf.variable_scope('reshape_yolo_0'):
                large_obj_box_detections = tf.reshape(large_obj_box_detections,
                                                      [-1,
                                                       large_obj_box_detections.shape[1]
                                                       * large_obj_box_detections.shape[2]
                                                       * large_obj_box_detections.shape[3],
                                                       self.num_classes + 5])

            with tf.variable_scope('reshape_yolo_1'):
                medium_obj_box_detections = tf.reshape(medium_obj_box_detections,
                                                       [-1,
                                                        medium_obj_box_detections.shape[1]
                                                        * medium_obj_box_detections.shape[2]
                                                        * medium_obj_box_detections.shape[3],
                                                        self.num_classes + 5])

            with tf.variable_scope('reshape_yolo_2'):
                small_obj_box_detections = tf.reshape(small_obj_box_detections,
                                                      [-1,
                                                       small_obj_box_detections.shape[1]
                                                       * small_obj_box_detections.shape[2]
                                                       * small_obj_box_detections.shape[3],
                                                       self.num_classes + 5])

            return large_obj_box_detections, medium_obj_box_detections, small_obj_box_detections

        else:

            return large_obj_box_detections, medium_obj_box_detections, small_obj_box_detections, \
              large_obj_raw_detections, medium_obj_raw_detections, small_obj_raw_detections
