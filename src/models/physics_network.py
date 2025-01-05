import tensorflow as tf
from src.models.network import Network


class PhysicsNetwork(tf.keras.Model):
    def __init__(
        self,
        cfg,
        Phi_int,
        Phi_diff,
        number_library_terms,
        function_acceleration,
        max_values,
    ):
        super(PhysicsNetwork, self).__init__()
        self.Phi_int = Phi_int
        self.Phi_diff = Phi_diff
        self.number_library_terms = number_library_terms
        self.function_acceleration = function_acceleration
        self.excitation_max = tf.cast(max_values["excitation_max"], tf.float32)
        self.displacement_max = tf.cast(max_values["displacement_max"], tf.float32)
        self.velocity_max = tf.cast(max_values["velocity_max"], tf.float32)
        self.displacement_model = Network(cfg)

        def set_coefficients(prefix):
            setattr(
                self,
                f"{prefix}0",
                tf.Variable(
                    tf.constant(1.0),
                    trainable=True,
                    # trainable=False,
                    dtype=tf.float32,
                    name=f"{prefix}0",
                ),
            )
            for index in range(self.number_library_terms - 1):
                setattr(
                    self,
                    f"{prefix}{index+1}",
                    tf.Variable(
                        tf.constant(1.0),
                        trainable=True,
                        dtype=tf.float32,
                        name=f"{prefix}{index+1}",
                    ),
                )

        set_coefficients("cx")
        self.group_variables_called = False

    def update_function(self, function):
        self.function_acceleration = function

    def clip_variables(self, threshold=0.10):
        reserve_terms = 3
        for index in range(self.number_library_terms - reserve_terms):
            cx = getattr(self, f"cx{index+reserve_terms}")
            if tf.abs(cx) < threshold:
                cx.assign(tf.zeros_like(cx))

    def group_variables(self):
        if not self.group_variables_called:
            self.network_variables = self.displacement_model.trainable_variables
            self.physics_variables = [
                getattr(self, f"cx{index}")
                for index in range(self.number_library_terms)
            ]
            self.group_variables_called = True

    def predict(self, input):
        normalized_displacement = self.displacement_model(input)
        velocity = (
            tf.matmul(self.Phi_diff, normalized_displacement) * self.displacement_max
        )
        normalized_velocity = velocity / self.velocity_max
        return normalized_displacement, normalized_velocity

    def step_forward(self, input):
        def get_coefficients(prefix):
            for index in range(self.number_library_terms):
                globals()[f"{prefix}{index}"] = getattr(self, f"{prefix}{index}")

        get_coefficients("cx")
        normalized_displacement, normalized_velocity = self.predict(input)
        f = tf.cast(input, dtype=tf.float32) * self.excitation_max
        x = tf.cast(normalized_displacement, tf.dtypes.float32)
        y = tf.cast(normalized_velocity, tf.dtypes.float32)
        acceleration_fit = eval(self.function_acceleration)

        integrated_normalized_velocity = (
            tf.matmul(self.Phi_int, acceleration_fit) / self.velocity_max
            + normalized_velocity[:, 0:1, :]
        )
        normalized_velocity_error = integrated_normalized_velocity - normalized_velocity
        self.group_variables()

        return (
            normalized_displacement,
            normalized_velocity_error,
        )
