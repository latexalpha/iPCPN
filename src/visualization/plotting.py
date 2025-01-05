import matplotlib.pyplot as plt
import tensorflow as tf


def plot_training_data(clean_data, noisy_data, title, training_figure_path):
    clean_instance = clean_data[0]
    noisy_instance = noisy_data[0]
    rms_1 = tf.sqrt(tf.reduce_mean(tf.square(clean_instance - noisy_instance)))
    rms_2 = tf.sqrt(tf.reduce_mean(tf.square(clean_instance)))
    rms_ratio = rms_1 / rms_2
    plt.figure()
    plt.plot(clean_instance, label="Clean x")
    plt.plot(noisy_instance, label="Noisy x")
    plt.plot(clean_instance - noisy_instance, label="Error x")
    plt.title(f"Training {title}, {rms_ratio:.4f}")
    plt.legend()
    plt.savefig(training_figure_path)


def plot_loss_curve(loss_training_log, loss_validation_log, title, loss_figure_path):
    plt.figure()
    plt.semilogy(loss_training_log, label="training loss")
    plt.semilogy(loss_validation_log, label="validation loss")
    plt.legend()
    plt.title(f"Loss curve of {title}")
    plt.savefig(loss_figure_path)


def plot_prediction(prediction, clean_data, title, prediction_figure_path):
    # assure prediction and clean_data have same dtype
    prediction = tf.cast(prediction, clean_data.dtype)
    prediction_instance = prediction[0]
    clean_instance = clean_data[0]
    rms_1 = tf.sqrt(tf.reduce_mean(tf.square(clean_instance - prediction_instance)))
    rms_2 = tf.sqrt(tf.reduce_mean(tf.square(clean_instance)))
    rms_ratio = rms_1 / rms_2
    plt.figure()
    plt.plot(prediction_instance, label="Predicted")
    plt.plot(clean_instance, label="Clean")
    plt.plot(prediction_instance - clean_instance, label="Error")
    plt.title(f"{title} prediction, {rms_ratio:.4f}")
    plt.legend()
    plt.savefig(prediction_figure_path)
