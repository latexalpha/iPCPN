import keras
import tensorflow as tf

from keras.layers import CuDNNLSTM
from tensorflow.signal import rfft, irfft
from tensorflow.keras.layers import Conv1D, Dense, Activation, Dropout


class Network(tf.keras.Model):
    def __init__(self, cfg, velocity_flag=False):
        super(Network, self).__init__()
        self.input_dim = cfg.model.input_dim
        self.lifting_channel = cfg.model.lifting_dim
        self.output_dim = cfg.model.output_dim
        self.filters = cfg.model.filters
        self.kernel_size = cfg.model.kernel_size
        if velocity_flag:
            self.modes = cfg.model.velocity_modes
        else:
            self.modes = cfg.model.modes
        self.fno_flag = cfg.model.fno_flag

        self.conv1d = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            activation="relu",
            input_shape=(None, self.input_dim),
        )
        self.lstm = CuDNNLSTM(
            units=128,
            return_sequences=True,
            stateful=False,
            # input_shape=(None, self.kernel_size),
            input_shape=(None, self.input_dim),
        )
        self.activation_tanh = Activation(keras.activations.tanh, name="Tanh")
        self.activation_swish = Activation(keras.activations.swish, name="Swish")
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dense1 = Dense(units=128)
        self.dense2 = Dense(units=32)
        self.dense3 = Dense(units=self.output_dim)

    def EncoderP(self, inputs):
        # CNN-LSTM model for EncoderP
        x = self.conv1d(inputs)
        x = self.lstm(x)
        x = self.activation_swish(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.activation_tanh(x)
        x = self.dense3(x)
        return x

    def SpectralConv1d(self, x, stage):
        """Second part of the FNO model: Defining F(x) --> U
        Args:
            x: input tensor [batch, sequence_length, num_channels]
            stage: stage of the model
        Returns:
            x: output tensor"""
        sequence_length = x.shape[1]
        xo_fft = rfft(tf.transpose(x, perm=[0, 2, 1]), name=f"FFT-Layer{stage}")
        xo_fft = xo_fft[:, :, : self.modes]
        x = irfft(xo_fft, fft_length=[sequence_length], name=f"IFFT-Layer{stage}")
        x = tf.transpose(x, perm=[0, 2, 1])
        return x

    def DecoderQ(self, x):
        """Third part of the FNO model: Defining Q --> u(x)"""
        x = Dense(units=self.output_dim, activation="relu")(x)
        return x

    def call(self, x):
        # in_channel --> out_channel
        x = self.EncoderP(x)
        if self.fno_flag:
            # out_channel --> out_channel
            x = self.SpectralConv1d(x, self.output_dim)
            # x = self.DecoderQ(x)
        y = tf.cast(x, tf.dtypes.float32)
        return y
