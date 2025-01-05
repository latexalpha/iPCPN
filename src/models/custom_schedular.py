import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Define a custom schedule for the learning rate with minimum value.
    Args:
        initial_learning_rate (float): Initial learning rate.
        decay_steps (int): Number of steps to decay the learning rate.
        decay_rate (float): Rate of decay.
        minimum_learning_rate (float): Minimum learning rate.
    Returns:
        float: Learning rate."""

    def __init__(
        self, initial_learning_rate, decay_steps, decay_rate, minimum_learning_rate
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.minimum_learning_rate = minimum_learning_rate

    def __call__(self, step):
        learning_rate = self.initial_learning_rate * self.decay_rate ** (
            step / self.decay_steps
        )
        return tf.math.maximum(learning_rate, self.minimum_learning_rate)
