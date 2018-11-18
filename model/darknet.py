import tensorflow as tf
from model import layer

DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Darknet:
    def __init__(self, name, backbone=None,
                 norm='batch', dtype=DEFAULT_DTYPE):
        self.name = name
        if not backbone or backbone == 'darknet53':
            self.num_id1 = 1
            self.num_id2 = 2
            self.num_id3 = 8
            self.num_id4 = 8
            self.num_id5 = 4
        self.norm = norm
        self.dtype = dtype
        self.reuse = False

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

        return tf.variable_scope(self.name,
                                 custom_getter=self._custom_dtype_getter)

    def __call__(self, ipt, is_training):
        self.is_training = is_training
        with self._model_variable_scope():
            c3s1k32 = layer.c3s1k32(ipt, name='conv_0', reuse=self.reuse, is_training=self.is_training, norm=self.norm)
            c3s2k64 = layer.c3s2k64(c3s1k32, name='conv_1',
                                    reuse=self.reuse, is_training=self.is_training, norm=self.norm)

            dark_net_conv_1 = []
            for i in range(self.num_id1):
                dark_net_conv_1.append(layer.dark_net_conv_1(dark_net_conv_1[-1] if i
                                                             else c3s2k64,
                                                             i=2 * i + 2,
                                                             name='conv_',
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            c3s2k128 = layer.c3s2k128(dark_net_conv_1[-1], name='conv_4', reuse=self.reuse,
                                      is_training=self.is_training,
                                      norm=self.norm)

            dark_net_conv_2 = []
            for i in range(self.num_id2):
                dark_net_conv_2.append(layer.dark_net_conv_2(dark_net_conv_2[-1] if i
                                                             else c3s2k128,
                                                             i=2*i + 3 + self.num_id1*2,
                                                             j=i,
                                                             name='conv_',
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            c3s2k256 = layer.c3s2k256(dark_net_conv_2[-1], name='conv_9',
                                      reuse=self.reuse, is_training=self.is_training,
                                      norm=self.norm)

            dark_net_conv_3 = []
            for i in range(self.num_id3):
                dark_net_conv_3.append(layer.dark_net_conv_3(dark_net_conv_3[-1] if i
                                                             else c3s2k256,
                                                             i=4 + 2*self.num_id1 + 2*self.num_id2 + 2*i,
                                                             j=i,
                                                             name='conv_',
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            c3s2k512 = layer.c3s2k512(dark_net_conv_3[-1], name='conv_26',
                                      reuse=self.reuse, is_training=self.is_training,
                                      norm=self.norm)

            dark_net_route_1 = dark_net_conv_3[-1]

            dark_net_conv_4 = []
            for i in range(self.num_id5):
                dark_net_conv_4.append(layer.dark_net_conv_4(dark_net_conv_4[-1] if i
                                                             else c3s2k512,
                                                             i=5+2*(self.num_id1+self.num_id2+self.num_id3+i),
                                                             j=i,
                                                             name='conv_',
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            c3s2k1024 = layer.c3s2k1024(dark_net_conv_4[-1], name='conv_43',
                                        reuse=self.reuse, is_training=self.is_training,
                                        norm=self.norm)

            dark_net_route_2 = dark_net_conv_4[-1]

            dark_net_conv_5 = []
            for i in range(self.num_id5):
                dark_net_conv_5.append(layer.dark_net_conv_5(dark_net_conv_5[-1] if i
                                                             else c3s2k1024,
                                                             i=6+2*(self.num_id1+self.num_id2+self.num_id4 +
                                                                    self.num_id3+i),
                                                             j=i,
                                                             name='conv_',
                                                             reuse=self.reuse,
                                                             is_training=self.is_training,
                                                             norm=self.norm))

            dark_out = dark_net_conv_5[-1]

        self.reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return dark_out, dark_net_route_1, dark_net_route_2
