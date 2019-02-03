from keras.callbacks import TensorBoard
import numpy as np
import os


class TensorBoardGrouped(TensorBoard):
    """TensorBoard basic visualizations.

    [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.

    This callback is a subclass of `keras.callbacks.TensorBoard`.
    The only difference is that the training and validation logs are
    grouped and written to the same plot.

    It's a drop-in replacement for the keras callback.
    The arguments are the same.
    """

    def __init__(self, log_dir='./logs', *args, **kwargs):
        self.base_log_dir = log_dir
        self.train_log_dir = os.path.join(log_dir, 'train')
        self.val_log_dir = os.path.join(log_dir, 'val')
        super(TensorBoardGrouped, self).__init__(self.train_log_dir,
                                                 *args,
                                                 **kwargs)

    def set_model(self, model):
        super(TensorBoardGrouped, self).set_model(model)
        import tensorflow as tf
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)

    def _write_logs(self, logs, index):
        import tensorflow as tf
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            if name.startswith('val_'):
                writer = self.val_writer
                name = name[4:]  # remove val_
            else:
                writer = self.writer
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            writer.add_summary(summary, index)
        self.writer.flush()
        self.val_writer.flush()

    def on_train_end(self, _):
        self.writer.close()
        self.val_writer.flush()
