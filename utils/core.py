from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import multiprocessing
import functools

import sys

from absl import app as absl_app
from absl import flags
import tensorflow as tf

DTYPE_MAP = {
    "fp16": (tf.float16, 128),
    "fp32": (tf.float32, 1),
}


def define_base(data_dir=True, model_dir=True, clean=True, train_epochs=True,
                epochs_between_evals=True, stop_threshold=True, batch_size=True,
                num_gpus=True, export_dir=True):
    """Register base flags.

        Args:
            data_dir: Create a flag for specifying the input data directory.
            model_dir: Create a flag for specifying the model file directory.
            clean: If set, model_dir will be removed if it exists.
            train_epochs: Create a flag to specify the number of training epochs.
            epochs_between_evals: Create a flag to specify the frequency of testing.
            stop_threshold: Create a flag to specify a threshold accuracy or other
                eval metric which should trigger the end of training.
            batch_size: Create a flag to specify the batch size.
            num_gpus: Create a flag to specify the number of GPUs used.
            export_dir: Create a flag to specify where a SavedModel should be exported.

        Returns:
            A list of flags for core.py to marks as key flags.
    """
    key_flags = []
    if data_dir:
        flags.DEFINE_string(name='data_dir', short_name='dd',
                            default='/home/hume/Deep-learning/dataset/',
                            help='The location of the input data.')
        key_flags.append('data_dir')

    if model_dir:
        flags.DEFINE_string(name='model_dir', short_name='md',
                            default='/home/hume/Deep-learning/model/',
                            help='The location of the model checkpoint files.')
        key_flags.append('model_dir')

    if clean:
        flags.DEFINE_bool(name='clean',
                          default=False,
                          help='If set, model_dir will be removed if it exists.')
        key_flags.append('clean')

    if train_epochs:
        flags.DEFINE_integer(name='train_epochs', short_name='te',
                             default=1,
                             help='The number of epochs used to train.')
        key_flags.append('train_epochs')

    if epochs_between_evals:
        flags.DEFINE_integer(name='epochs_between_evals', short_name='ebe',
                             default=1,
                             help='The number of training epochs to run between '
                                  'evaluations.')
        key_flags.append('epochs_between_evals')

    if stop_threshold:
        flags.DEFINE_float(
            name='stop_threshold', short_name='st',
            default=None,
            help="If passed, training will stop at the earlier of "
                 "train_epochs and when the evaluation metric is  "
                 "greater than or equal to stop_threshold."
        )
        key_flags.append('stop_threshold')

    if batch_size:
        flags.DEFINE_integer(
            name="batch_size", short_name="bs", default=32,
            help="Batch size for training and evaluation. When using "
                 "multiple gpus, this is the global batch size for "
                 "all devices. For example, if the batch size is 32 "
                 "and there are 4 GPUs, each GPU will get 8 examples on "
                 "each step."
        )
        key_flags.append("batch_size")

    if num_gpus:
        flags.DEFINE_integer(
            name="num_gpus", short_name="ng",
            default=-1 if tf.test.is_gpu_available() else 0,
            help="How many GPUs to use with the DistributionStrategies API. The "
                 "default is 1 if TensorFlow can detect a GPU, and 0 otherwise."
        )
        key_flags.append('num_gpus')

    if export_dir:
        flags.DEFINE_string(
            name="export_dir", short_name="ed", default=None,
            help="If set, a SavedModel serialization of the model will "
                 "be exported to this directory at the end of training. "
                 "See the README for more details and relevant links."
        )
        key_flags.append("export_dir")

    return key_flags


def define_performance(num_parallel_calls=True, inter_op=True, intra_op=True,
                       synthetic_data=True, max_train_steps=True, dtype=True,
                       all_reduce_alg=True, tf_gpu_thread_mode=False,
                       datasets_num_private_threads=False,
                       datasets_num_parallel_batches=False):
    """Register flags for specifying performance tuning arguments.

        Args:
            num_parallel_calls: Create a flag to specify parallelism of data loading.
            inter_op: Create a flag to allow specification of inter op threads.
            intra_op: Create a flag to allow specification of intra op threads.
            synthetic_data: Create a flag to allow the use of synthetic data.
            max_train_steps: Create a flags to allow specification of maximum number
                of training steps
            dtype: Create flags for specifying dtype.
            all_reduce_alg: If set forces a specific algorithm for multi-gpu.
            tf_gpu_thread_mode: gpu_private triggers us of private thread pool.
            datasets_num_private_threads: Number of private threads for datasets.
            datasets_num_parallel_batches: Determines how many batches to process in
            parallel when using map and batch from tf.data.

        Returns:
            A list of flags for core.py to marks as key flags.
    """

    key_flags = []
    if num_parallel_calls:
        flags.DEFINE_integer(
            name="num_parallel_calls", short_name="npc",
            default=multiprocessing.cpu_count(),
            help="The number of records that are  processed in parallel "
                 "during input processing. This can be optimized per "
                 "data set but for generally homogeneous data sets, "
                 "should be approximately the number of available CPU "
                 "cores. (default behavior)"
        )
        key_flags.append('num_parallel_calls')

    if inter_op:
        flags.DEFINE_integer(
            name="inter_op_parallelism_threads", short_name="inter", default=0,
            help="Number of inter_op_parallelism_threads to use for CPU. "
                 "See TensorFlow config.proto for details."
        )
        key_flags.append('inter_op_parallelism_threads')

    if intra_op:
        flags.DEFINE_integer(
            name="intra_op_parallelism_threads", short_name="intra", default=0,
            help="Number of intra_op_parallelism_threads to use for CPU. "
                 "See TensorFlow config.proto for details."
        )
        key_flags.append('intra_op_parallelism_threads')

    if synthetic_data:
        flags.DEFINE_bool(
            name="use_synthetic_data", short_name="synth", default=False,
            help="If set, use fake data (zeroes) instead of a real dataset. "
                 "This mode is useful for performance debugging, as it removes "
                 "input processing steps, but will not learn anything."
        )
        key_flags.append('use_synthetic_data')

    if max_train_steps:
        flags.DEFINE_integer(
            name="max_train_steps", short_name="mts", default=None,
            help="The model will stop training if the global_step reaches this "
                 "value. If not set, training will run until the specified number "
                 "of epochs have run as usual. It is generally recommended to set "
                 "--train_epochs=1 when using this flag."
            )
        key_flags.append('max_train_steps')

    if dtype:
        flags.DEFINE_enum(
            name="dtype", short_name="dt", default="fp32",
            enum_values=DTYPE_MAP.keys(),
            help="The TensorFlow datatype used for calculations. "
                 "Variables may be cast to a higher precision on a "
                 "case-by-case basis for numerical stability."
        )
        key_flags.append('dtype')

        flags.DEFINE_integer(
            name="loss_scale", short_name="ls", default=None,
            help="The amount to scale the loss by when the model is run. Before "
                 "gradients are computed, the loss is multiplied by the loss scale, "
                 "making all gradients loss_scale times larger. To adjust for this, "
                 "gradients are divided by the loss scale before being applied to "
                 "variables. This is mathematically equivalent to training without "
                 "a loss scale, but the loss scale helps avoid some intermediate "
                 "gradients from underflowing to zero. If not provided the default "
                 "for fp16 is 128 and 1 for all other dtypes."
        )
        key_flags.append('loss_scale')

        loss_scale_val_msg = "loss_scale should be a positive integer."

        @flags.validator(flag_name="loss_scale", message=loss_scale_val_msg)
        def _check_loss_scale(loss_scale):  # pylint: disable=unused-variable
            if loss_scale is None:
                return True  # null case is handled in get_loss_scale()

            return loss_scale > 0

    if all_reduce_alg:
        flags.DEFINE_string(
            name="all_reduce_alg", short_name="ara", default=None,
            help="Defines the algorithm to use for performing all-reduce."
                 "See tf.contrib.distribute.AllReduceCrossTowerOps for "
                 "more details and available options."
        )
        key_flags.append('all_reduce_alg')

    if tf_gpu_thread_mode:
        flags.DEFINE_string(
            name="tf_gpu_thread_mode", short_name="gt_mode", default='gpu_private',
            help="Whether and how the GPU device uses its own threadpool."
        )
        key_flags.append('tf_gpu_thread_mode')

    if datasets_num_private_threads:
        flags.DEFINE_integer(
            name="datasets_num_private_threads",
            default=None,
            help="Number of threads for a private threadpool created for all"
                 "datasets computation.."
        )
        key_flags.append('datasets_num_private_threads')

    if datasets_num_parallel_batches:
        flags.DEFINE_integer(
            name="datasets_num_parallel_batches",
            default=None,
            help="Determines how many batches to process in parallel when using "
                 "map and batch from tf.data."
        )
        key_flags.append('datasets_num_parallel_batches')

    return key_flags


def define_image(data_format=True):
    """Register image specific flags.

    Args:
        data_format: Create a flag to specify image axis convention.

    Returns:
        A list of flags for core.py to marks as key flags.
    """

    key_flags = []

    if data_format:
        flags.DEFINE_enum(
            name="data_format", short_name="df", default=None,
            enum_values=["channels_first", "channels_last"],
            help="A flag to override the data format used in the model. "
                 "channels_first provides a performance boost on GPU but is not "
                 "always compatible with CPU. If left unspecified, the data format "
                 "will be chosen automatically based on whether TensorFlow was "
                 "built for CPU or GPU.")
        key_flags.append("data_format")

    return key_flags


def get_tf_dtype(flags_obj):
    return DTYPE_MAP[flags_obj.dtype][0]


def get_loss_scale(flags_obj):
    if flags_obj.loss_scale is not None:
        return flags_obj.loss_scale
    return DTYPE_MAP[flags_obj.dtype][1]


def get_num_gpus(flags_obj):
    """Treat num_gpus=-1 as 'use all'."""
    if flags_obj.num_gpus != -1:
        return flags_obj.num_gpus

    from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
    local_device_protos = device_lib.list_local_devices()
    return sum([1 for d in local_device_protos if d.device_type == "GPU"])


def register_key_flags_in_core(f):
    """Defines a function in core.py, and registers its key flags.

    absl uses the location of a flags.declare_key_flag() to determine the context
    in which a flag is key. By making all declares in core, this allows model
    main functions to call flags.adopt_module_key_flags() on core and correctly
    chain key flags.

    Args:
        f:  The function to be wrapped

    Returns:
        The "core-defined" version of the input function.
    """

    def core_fn(*args, **kwargs):
        key_flags = f(*args, **kwargs)
        [flags.declare_key_flag(fl) for fl in key_flags]  # pylint: disable=expression-not-assigned
    return core_fn


def set_defaults(**kwargs):
    for key, value in kwargs.items():
        flags.FLAGS.set_default(name=key, value=value)


def parse_flags(argv=None):
    """Reset flags and reparse. Currently only used in testing."""
    flags.FLAGS.unparse_flags()
    absl_app.parse_flags_with_usage(argv or sys.argv)


define_base = register_key_flags_in_core(define_base)
# Remove options not relevant for Eager from define_base().
define_base_eager = register_key_flags_in_core(functools.partial(
    define_base, epochs_between_evals=False, stop_threshold=False,
    hooks=False))

define_image = register_key_flags_in_core(define_image)
define_performance = register_key_flags_in_core(define_performance)
