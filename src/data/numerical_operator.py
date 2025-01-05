import tensorflow as tf
from tensorflow import linalg
from einops import repeat, pack


@tf.function
def integration_operator(batch, seq_len, dt):
    phi_1 = linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    phi_1 = tf.tensor_scatter_nd_update(
        phi_1, [[0, 0]], tf.constant([0.0], dtype=tf.float32)
    )

    phi_2 = linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    # update the first column of phi_2 to zeros
    indices = [[i, 0] for i in range(seq_len)]
    updates = tf.zeros((seq_len,), dtype=tf.float32)
    phi_2 = tf.tensor_scatter_nd_update(phi_2, indices, updates)

    indices = [[i, i] for i in range(seq_len)]
    updates = tf.zeros((seq_len,), dtype=tf.float32)
    phi_2 = tf.tensor_scatter_nd_update(phi_2, indices, updates)

    dt = tf.cast(dt, dtype=tf.float32)
    phi = (phi_1 + phi_2) * dt / 2
    Phi = repeat(phi, "row col -> batch row col", batch=batch)
    return Phi


@tf.function
def differentiation_operator(batch, seq, dt):
    phi_1, ps = pack(
        [
            tf.constant([[-3 / 2, 2, -1 / 2]], dtype=tf.float32),
            tf.zeros([1, seq - 3], dtype=tf.float32),
        ],
        "r *",
    )

    temp_1, ps = pack(
        [
            -1 / 2 * tf.eye(seq - 2, dtype=tf.float32),
            tf.zeros([seq - 2, 2], dtype=tf.float32),
        ],
        "r *",
    )
    temp_2, ps = pack(
        [
            tf.zeros([seq - 2, 2], dtype=tf.float32),
            1 / 2 * tf.eye(seq - 2, dtype=tf.float32),
        ],
        "r *",
    )
    phi_2 = temp_1 + temp_2

    phi_3, ps = pack(
        [
            tf.zeros([1, seq - 3], dtype=tf.float32),
            tf.constant([[1 / 2, -2, 3 / 2]], dtype=tf.float32),
        ],
        "r *",
    )

    phi, ps = pack([phi_1, phi_2, phi_3], "* c")
    phi = 1 / dt * phi
    Phi = repeat(phi, "row col -> batch row col", batch=batch)

    return Phi
