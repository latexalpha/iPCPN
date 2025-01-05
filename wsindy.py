import numpy as np
import scipy.io as sio
import pysindy as ps

if __name__ == "__main__":
    # Load the data
    dataset_path = "./data/nonautonomous_duffing_20.mat"
    dataset = sio.loadmat(dataset_path)

    # Extract the data
    fs = dataset["fs"][0, 0]
    dt = 1 / fs
    excitation_matrix = dataset["excitation_matrix"]
    displacement_noisy_matrix = dataset["displacement_noisy_matrix"]
    excitation_max = dataset["excitation_max"][0, 0]
    displacement_max = dataset["displacement_noisy_max"][0, 0]
    velocity_max = dataset["velocity_noisy_max"][0, 0]

    excitation = excitation_matrix[0:1, :].T
    displacement_noisy = displacement_noisy_matrix[0:1, :].T * displacement_max

    excitation = excitation * excitation_max
    t_train = np.arange(0, excitation.shape[0], 1) * dt

    sfd = ps.differentiation.SmoothedFiniteDifference(
        smoother_kws={"window_length": 50}
    )
    velocity_fd = sfd._differentiate(displacement_noisy, t=dt)
    u_train = np.concatenate((excitation, displacement_noisy, velocity_fd), axis=1)

    feats = [
        lambda f, x, y: f,
        lambda f, x, y: x,
        lambda f, x, y: y,
        lambda f, x, y: x**2,
        lambda f, x, y: x * y,
        lambda f, x, y: y**2,
        lambda f, x, y: x**3,
        lambda f, x, y: x**2 * y,
        lambda f, x, y: x * y**2,
        lambda f, x, y: y**3,
        lambda f, x, y: x**4,
        lambda f, x, y: x**3 * y,
        lambda f, x, y: x**2 * y**2,
        lambda f, x, y: x * y**3,
        lambda f, x, y: y**4,
        lambda f, x, y: x**5,
        lambda f, x, y: x**4 * y,
        lambda f, x, y: x**3 * y**2,
        lambda f, x, y: x**2 * y**3,
        lambda f, x, y: x * y**4,
        lambda f, x, y: y**5,
    ]
    feature_names = [
        lambda f, x, y: f,
        lambda f, x, y: x,
        lambda f, x, y: y,
        lambda f, x, y: x + x,
        lambda f, x, y: x + y,
        lambda f, x, y: y + y,
        lambda f, x, y: x + x + x,
        lambda f, x, y: x + x + y,
        lambda f, x, y: x + y + y,
        lambda f, x, y: y + y + y,
        lambda f, x, y: x + x + x + x,
        lambda f, x, y: x + x + x + y,
        lambda f, x, y: x + x + y + y,
        lambda f, x, y: x + y + y + y,
        lambda f, x, y: y + y + y + y,
        lambda f, x, y: x + x + x + x + x,
        lambda f, x, y: x + x + x + x + y,
        lambda f, x, y: x + x + x + y + y,
        lambda f, x, y: x + x + y + y + y,
        lambda f, x, y: x + y + y + y + y,
        lambda f, x, y: y + y + y + y + y,
    ]
    opt = ps.STLSQ(threshold=0.05, normalize_columns=True)
    lib = ps.WeakPDELibrary(
        library_functions=feats,
        function_names=feature_names,
        spatiotemporal_grid=t_train,
        is_uniform=True,
        K=100,
    )
    model = ps.SINDy(optimizer=opt, feature_library=lib, feature_names=["f", "x", "y"])
    model.fit(u_train)
    model.print()
