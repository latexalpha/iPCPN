import os
import scipy
import tensorflow as tf
from tensorflow.data import Dataset

from src.data import return_data
from src.data.numerical_operator import integration_operator, differentiation_operator
from src.models.physics_ao_trainer import Trainer
from src.visualization.plotting import plot_training_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any 0 1 2
gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def system_training(cfg, output_dir, logger):
    """Load the data and set up the directories for saving the results"""
    data_path = (
        f"./data/{cfg.data.type}_{cfg.data.system}_"
        + (
            f"{cfg.data.noise_ratio:.2f}"
            if cfg.data.noise_ratio < 1
            else f"{cfg.data.noise_ratio}"
        )
        + ".mat"
    )
    data = return_data.return_data(data_path, cfg.data.split_ratio)
    fs = data["fs"]
    dt = 1 / fs
    excitation_max = data["excitation_max"]
    excitation_train = data["excitation_train"]
    excitation_validation = data["excitation_test"]
    # displacement
    displacement_noisy_max = data["displacement_noisy_max"]
    displacement_train = data["displacement_train"]
    displacement_train_clean = data["displacement_train_label"]
    displacement_validation = data["displacement_test"]
    displacement_validation_clean = data["displacement_test_label"]
    # velocity
    velocity_noisy_max = data["velocity_noisy_max"]
    velocity_test_clean = data["velocity_test_label"]

    figure_dir = os.path.join(output_dir, cfg.dirs.figures)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    result_dir = os.path.join(output_dir, cfg.dirs.results)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    plot_training_data(
        displacement_train_clean,
        displacement_train,
        "displacement",
        f"{figure_dir}/displacement_train.png",
    )

    batch_size = cfg.training.batch_size
    seq = excitation_train.shape[1]
    dt = tf.constant(dt, dtype=tf.float32)
    # prepare the training and validation dataset for parallel networks
    training_dataset = Dataset.from_tensor_slices(
        (
            excitation_train,
            displacement_train,
        )
    ).batch(batch_size, drop_remainder=True)
    validation_dataset = Dataset.from_tensor_slices(
        (
            excitation_validation,
            displacement_validation,
        )
    ).batch(batch_size, drop_remainder=True)

    trainer = Trainer(cfg, logger, output_dir)
    batch_integration_operator = integration_operator(batch_size, seq, dt)
    batch_differentiation_operator = differentiation_operator(batch_size, seq, dt)

    max_values = {
        "excitation_max": excitation_max,
        "displacement_max": displacement_noisy_max,
        "velocity_max": velocity_noisy_max,
    }
    clean_data = {
        "displacement_clean": displacement_validation_clean,
        "velocity_clean": velocity_test_clean,
    }
    trained_results = trainer.training(
        training_dataset,
        validation_dataset,
        batch_integration_operator,
        batch_differentiation_operator,
        clean_data,
        max_values,
    )
    displacement_all = trained_results["displacement_all"]
    velocity_hat_all = trained_results["velocity_hat_all"]
    result_path = (
        f"{result_dir}{cfg.data.system}_{cfg.data.type}_{cfg.data.noise_ratio}.mat"
    )
    scipy.io.savemat(
        result_path,
        {
            "fs": fs,
            "dt": dt,
            # excitation
            "excitation_max": excitation_max,
            "excitation": excitation_validation,
            # displacement
            "displacement_noisy_max": displacement_noisy_max,
            "displacement_noisy": displacement_validation,
            "displacement_clean": displacement_validation_clean,
            "displacement_pred": displacement_all,
            # velocity
            "velocity_noisy_max": velocity_noisy_max,
            "velocity_clean": velocity_test_clean,
            "velocity_pred": velocity_hat_all,
        },
    )
