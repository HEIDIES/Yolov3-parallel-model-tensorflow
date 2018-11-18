import tensorflow as tf
from model import ops


def c3s1k32(ipt, name='c3s1k32', reuse=False, is_training=True, norm='batch'):
    return ops.conv2d(ipt, 32, 3, 1, 1, norm=norm, activation=ops.leaky_relu,
                      name=name, reuse=reuse, is_training=is_training)


def c3s2k64(ipt, name='c3s2k64', reuse=False, is_training=True, norm='batch'):
    return ops.conv2d(ipt, 64, 3, 1, 2, norm=norm, activation=ops.leaky_relu,
                      name=name, reuse=reuse, is_training=is_training)


def dark_net_conv_1(ipt, i, name='dark_net_conv_1', reuse=False, is_training=True, norm='batch'):
    c1s1k32 = ops.conv2d(ipt, 32, 1, 0, 1, norm=norm, activation=ops.leaky_relu,
                         name=name + str(i), reuse=reuse, is_training=is_training)
    c3s1k64 = ops.conv2d(c1s1k32, 64, 3, 1, 1, norm=norm, activation=ops.leaky_relu,
                         name=name + str(i + 1), reuse=reuse, is_training=is_training)
    return tf.add(c3s1k64, ipt, name='shortcut_0')


def c3s2k128(ipt, name='c3s2k128', reuse=False, is_training=True, norm='batch'):
    return ops.conv2d(ipt, 128, 3, 1, 2, norm=norm, activation=ops.leaky_relu,
                      name=name, reuse=reuse, is_training=is_training)


def dark_net_conv_2(ipt, i, j, name='dark_net_conv_2', reuse=False, is_training=True, norm='batch'):
    c1s1k64 = ops.conv2d(ipt, 64, 1, 0, 1, norm=norm, activation=ops.leaky_relu,
                         name=name + str(i), reuse=reuse, is_training=is_training)
    c3s1k128 = ops.conv2d(c1s1k64, 128, 3, 1, 1, norm=norm, activation=ops.leaky_relu,
                          name=name + str(i+1), reuse=reuse, is_training=is_training)
    if j == 0:
        return tf.add(c3s1k128, ipt, name='shortcut_1')
    else:
        return tf.add(c3s1k128, ipt, name='shortcut_2')


def c3s2k256(ipt, name='c3s2k256', reuse=False, is_training=True, norm='batch'):
    return ops.conv2d(ipt, 256, 3, 1, 2, norm=norm, activation=ops.leaky_relu,
                      name=name, reuse=reuse, is_training=is_training)


def dark_net_conv_3(ipt, i, j, name='dark_net_conv_3', reuse=False, is_training=True, norm='batch'):
    c1s1k128 = ops.conv2d(ipt, 128, 1, 0, 1, norm=norm, activation=ops.leaky_relu,
                          name=name + str(i), reuse=reuse, is_training=is_training)
    c3s1k256 = ops.conv2d(c1s1k128, 256, 3, 1, 1, norm=norm, activation=ops.leaky_relu,
                          name=name + str(i+1), reuse=reuse, is_training=is_training)
    return tf.add(c3s1k256, ipt, name='shotcut_' + str(j+3))


def c3s2k512(ipt, name='c3s2k512', reuse=False, is_training=True, norm='batch'):
    return ops.conv2d(ipt, 512, 3, 1, 2, norm=norm, activation=ops.leaky_relu,
                      name=name, reuse=reuse, is_training=is_training)


def dark_net_conv_4(ipt, i, j, name='dark_net_conv_4', reuse=False, is_training=True, norm='batch'):
    c1s1k256 = ops.conv2d(ipt, 256, 1, 0, 1, norm=norm, activation=ops.leaky_relu,
                          name=name + str(i), reuse=reuse, is_training=is_training)
    c3s1k512 = ops.conv2d(c1s1k256, 512, 3, 1, 1, norm=norm, activation=ops.leaky_relu,
                          name=name + str(i+1), reuse=reuse, is_training=is_training)
    return tf.add(c3s1k512, ipt, name='shortcut_' + str(j + 11))


def c3s2k1024(ipt, name='c3s2k1024', reuse=False, is_training=True, norm='batch'):
    return ops.conv2d(ipt, 1024, 3, 1, 2, norm=norm, activation=ops.leaky_relu,
                      name=name, reuse=reuse, is_training=is_training)


def dark_net_conv_5(ipt, i, j, name='dark_net_conv_5', reuse=False, is_training=True, norm='batch'):
    c1s1k512 = ops.conv2d(ipt, 512, 1, 0, 1, norm=norm, activation=ops.leaky_relu,
                          name=name + str(i), reuse=reuse, is_training=is_training)
    c3s1k1024 = ops.conv2d(c1s1k512, 1024, 3, 1, 1, norm=norm, activation=ops.leaky_relu,
                           name=name + str(i+1), reuse=reuse, is_training=is_training)
    return tf.add(c3s1k1024, ipt, name='shortcut_' + str(j + 19))


def yolo_1(ipt, num_classes, reuse=False, is_training=True, norm='bath',
           activation=ops.leaky_relu):
    c1s1k512 = ops.conv2d(ipt, 512, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_52', reuse=reuse, is_training=is_training)
    c3s1k1024 = ops.conv2d(c1s1k512, 1024, 3, 1, 1, norm=norm, activation=activation,
                           name='conv_53', reuse=reuse, is_training=is_training)
    c1s1k512 = ops.conv2d(c3s1k1024, 512, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_54', reuse=reuse, is_training=is_training)
    c3s1k1024 = ops.conv2d(c1s1k512, 1024, 3, 1, 1, norm=norm, activation=activation,
                           name='conv_55', reuse=reuse, is_training=is_training)
    c1s1k512 = ops.conv2d(c3s1k1024, 512, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_56', reuse=reuse, is_training=is_training)
    yolo_route = c1s1k512
    c3s1k1024 = ops.conv2d(c1s1k512, 1024, 3, 1, 1, norm=norm, activation=activation,
                           name='conv_57', reuse=reuse, is_training=is_training)
    large_obj_raw_detections = ops.conv2d(c3s1k1024, 3 * (num_classes + 5), 1, 0, 1, norm=None, activation=None,
                                          use_bias=True,
                                          name='conv_58', reuse=reuse, is_training=is_training)
    return large_obj_raw_detections, yolo_route


def yolo_2(ipt1, ipt2, num_classes, reuse=False, is_training=True, norm='batch',
           activation=ops.leaky_relu):
    c1s1k256 = ops.conv2d(ipt1, 256, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_59', reuse=reuse, is_training=is_training)
    with tf.variable_scope('upsample_0'):
        upsample = tf.image.resize_nearest_neighbor(c1s1k256, (c1s1k256.shape[1] * 2, c1s1k256.shape[2] * 2))
    with tf.variable_scope('route_0'):
        route = tf.concat([upsample, ipt2], axis=3)
    c1s1k256 = ops.conv2d(route, 256, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_60', reuse=reuse, is_training=is_training)
    c3s1k512 = ops.conv2d(c1s1k256, 512, 3, 1, 1, norm=norm, activation=activation,
                          name='conv_61', reuse=reuse, is_training=is_training)
    c1s1k256 = ops.conv2d(c3s1k512, 256, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_62', reuse=reuse, is_training=is_training)
    c3s1k512 = ops.conv2d(c1s1k256, 512, 3, 1, 1, norm=norm, activation=activation,
                          name='conv_63', reuse=reuse, is_training=is_training)
    c1s1k256 = ops.conv2d(c3s1k512, 256, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_64', reuse=reuse, is_training=is_training)
    yolo_route = c1s1k256
    c3s1k512 = ops.conv2d(c1s1k256, 512, 3, 1, 1, norm=norm, activation=activation,
                          name='conv_65', reuse=reuse, is_training=is_training)
    medium_obj_raw_detections = ops.conv2d(c3s1k512, 3 * (num_classes + 5), 1, 0, 1, norm=None, activation=None,
                                           use_bias=True,
                                           name='conv_66', reuse=reuse, is_training=is_training)
    return medium_obj_raw_detections, yolo_route


def yolo_3(ipt1, ipt2, num_classes, reuse=False, is_training=True, norm='batch',
           activation=ops.leaky_relu):
    c1s1k128 = ops.conv2d(ipt1, 128, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_67', reuse=reuse, is_training=is_training)
    with tf.variable_scope('upsample_1'):
        upsample = tf.image.resize_nearest_neighbor(c1s1k128, (c1s1k128.shape[1] * 2, c1s1k128.shape[2] * 2))
    with tf.variable_scope('route_1'):
        route = tf.concat([upsample, ipt2], axis=3)
    c1s1k128 = ops.conv2d(route, 128, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_68', reuse=reuse, is_training=is_training)
    c3s1k256 = ops.conv2d(c1s1k128, 256, 3, 1, 1, norm=norm, activation=activation,
                          name='conv_69', reuse=reuse, is_training=is_training)
    c1s1k128 = ops.conv2d(c3s1k256, 128, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_70', reuse=reuse, is_training=is_training)
    c3s1k256 = ops.conv2d(c1s1k128, 256, 3, 1, 1, norm=norm, activation=activation,
                          name='conv_71', reuse=reuse, is_training=is_training)
    c1s1k128 = ops.conv2d(c3s1k256, 128, 1, 0, 1, norm=norm, activation=activation,
                          name='conv_72', reuse=reuse, is_training=is_training)

    c3s1k256 = ops.conv2d(c1s1k128, 256, 3, 1, 1, norm=norm, activation=activation,
                          name='conv_73', reuse=reuse, is_training=is_training)
    small_obj_raw_detections = ops.conv2d(c3s1k256, 3 * (num_classes + 5), 1, 0, 1, norm=None, activation=None,
                                          use_bias=True,
                                          name='conv_74', reuse=reuse, is_training=is_training)
    return small_obj_raw_detections


def yolo_layer(ipt, name, anchors, num_classes, image_size):
    with tf.variable_scope(name):
        inputs_shape = ipt.get_shape().as_list()
        stride_x = image_size // inputs_shape[2]
        stride_y = image_size // inputs_shape[1]

        num_anchors = len(anchors)
        anchors = tf.constant([[a[0] / stride_x, a[1] / stride_y] for a in anchors], dtype=tf.float32)
        anchors_w = tf.reshape(anchors[:, 0], [1, 1, 1, num_anchors, 1])
        anchors_h = tf.reshape(anchors[:, 1], [1, 1, 1, num_anchors, 1])

        clustriod_x = tf.tile(tf.reshape(tf.range(inputs_shape[2], dtype=tf.float32), [1, -1, 1, 1]),
                              [inputs_shape[2], 1, 1, 1])
        clustriod_y = tf.tile(tf.reshape(tf.range(inputs_shape[1], dtype=tf.float32), [-1, 1, 1, 1]),
                              [1, inputs_shape[1], 1, 1])

        ipt = tf.reshape(ipt, [-1, inputs_shape[1], inputs_shape[2], num_anchors, 5 + num_classes])
        delta_x, delta_y, delta_w, delta_h, conf_obj, class_obj = tf.split(ipt, [1, 1, 1, 1, 1, num_classes], axis=-1)

        box_x = (clustriod_x + tf.sigmoid(delta_x)) * stride_x
        box_y = (clustriod_y + tf.sigmoid(delta_y)) * stride_y

        box_w = anchors_w * tf.exp(delta_w) * stride_x
        box_h = anchors_h * tf.exp(delta_h) * stride_y

        conf_obj = tf.nn.sigmoid(conf_obj)
        class_obj = tf.nn.sigmoid(class_obj)

        output = tf.concat([box_x, box_y, box_w, box_h, conf_obj, class_obj], axis=-1)

    return output
