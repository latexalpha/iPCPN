import scipy.io
from einops import repeat


def return_data(data_path, split_ratio):
    dataset = scipy.io.loadmat(data_path)

    fs = dataset["fs"]
    fs = fs[0, 0]
    excitation_max = dataset["excitation_max"]
    excitation_max = excitation_max[0, 0]
    displacement_noisy_max = dataset["displacement_noisy_max"]
    velocity_noisy_max = dataset["velocity_noisy_max"]
    displacement_noisy_max = displacement_noisy_max[0, 0]
    velocity_noisy_max = velocity_noisy_max[0, 0]

    excitation_matrix = dataset["excitation_matrix"]
    displacement_matrix = dataset["displacement_matrix"]
    displacement_noisy_matrix = dataset["displacement_noisy_matrix"]
    velocity_matrix = dataset["velocity_matrix"]
    velocity_noisy_matrix = dataset["velocity_noisy_matrix"]

    excitation_matrix = repeat(excitation_matrix, "batch seq -> batch seq c", c=1)
    displacement_matrix = repeat(displacement_matrix, "batch seq -> batch seq c", c=1)
    displacement_noisy_matrix = repeat(
        displacement_noisy_matrix, "batch seq -> batch seq c", c=1
    )
    velocity_matrix = repeat(velocity_matrix, "batch seq -> batch seq c", c=1)
    velocity_noisy_matrix = repeat(
        velocity_noisy_matrix, "batch seq -> batch seq c", c=1
    )

    sample_number = displacement_matrix.shape[0]
    train_number = int(sample_number * split_ratio)

    excitation_train = excitation_matrix[0:train_number, :, :]
    excitation_test = excitation_matrix[train_number:, :, :]
    displacement_train = displacement_noisy_matrix[0:train_number, :, :]
    displacement_train_label = displacement_matrix[0:train_number, :, :]
    displacement_test = displacement_noisy_matrix[train_number:, :, :]
    displacement_test_label = displacement_matrix[train_number:, :, :]
    velocity_train = velocity_noisy_matrix[0:train_number, :, :]
    velocity_test = velocity_noisy_matrix[train_number:, :, :]
    velocity_train_label = velocity_matrix[0:train_number, :, :]
    velocity_test_label = velocity_matrix[train_number:, :, :]

    return {
        "fs": fs,
        "excitation_max": excitation_max,
        "displacement_noisy_max": displacement_noisy_max,
        "velocity_noisy_max": velocity_noisy_max,
        "excitation_train": excitation_train,
        "excitation_test": excitation_test,
        "displacement_train": displacement_train,
        "displacement_train_label": displacement_train_label,
        "displacement_test": displacement_test,
        "displacement_test_label": displacement_test_label,
        "velocity_train": velocity_train,
        "velocity_train_label": velocity_train_label,
        "velocity_test": velocity_test,
        "velocity_test_label": velocity_test_label,
    }
