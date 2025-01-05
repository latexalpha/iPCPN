import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.models.physics_network import PhysicsNetwork as PNetwork
from src.models.custom_schedular import CustomSchedule as Schedule
from src.models.function_library import FunctionLibrary as Library
from src.visualization.plotting import plot_prediction, plot_loss_curve


class Trainer:
    def __init__(self, cfg, logger, output_dir):
        self.cfg = cfg
        self.logger = logger
        self.library = Library()
        checkpoint_dir = os.path.join(output_dir, cfg.dirs.checkpoints)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = checkpoint_dir + "model"
        self.figure_dir = os.path.join(output_dir, cfg.dirs.figures)
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)
        # training
        self.mse = tf.keras.losses.MeanSquaredError()
        network_scheduler = Schedule(
            cfg.training.network_initial_lr,
            cfg.training.network_decay_steps,
            cfg.training.network_decay_rate,
            cfg.training.network_minimum_lr,
        )
        physics_scheduler = Schedule(
            cfg.training.physics_initial_lr,
            cfg.training.physics_decay_steps,
            cfg.training.physics_decay_rate,
            cfg.training.physics_minimum_lr,
        )
        self.network_opt = tf.keras.optimizers.Adam(
            learning_rate=network_scheduler, beta_1=0.9, beta_2=0.999
        )
        self.physics_opt = tf.keras.optimizers.Adam(
            learning_rate=physics_scheduler, beta_1=0.9, beta_2=0.999
        )
        self.pretrain_opt = tf.keras.optimizers.Adam(
            learning_rate=cfg.training.pretrain_lr, beta_1=0.9, beta_2=0.999
        )
        self.pretrain_loss_minimum = 1e10
        self.physics_loss_minimum = 1e10
        self.network_loss_minimum = 1e10
        self.lambda_velocity_int = cfg.training.lambda_velocity_int
        self.lambda_l1 = cfg.training.lambda_l1

    def compute_losses(self, model, input, output):
        (
            normalized_displacement_hat,
            normalized_velocity_error,
        ) = model.step_forward(input)
        displacement_loss = self.mse(output, normalized_displacement_hat)
        velocity_loss = self.mse(
            normalized_velocity_error, tf.zeros_like(normalized_velocity_error)
        )
        l1_loss = tf.cast(tf.reduce_sum(tf.abs(model.physics_variables)), tf.float32)
        return (
            displacement_loss,
            velocity_loss,
            l1_loss,
        )

    def network_train_step(self, model, training_dataset, physics_flag=True):
        if training_dataset is None or not isinstance(
            training_dataset, tf.data.Dataset
        ):
            raise ValueError("Invalid training dataset")

        training_batch_num = 0
        average_training_loss = 0.0
        for input, displacement in training_dataset:
            training_batch_num += 1
            with tf.GradientTape() as tape:
                (
                    displacement_loss,
                    velocity_loss,
                    l1_loss,
                ) = self.compute_losses(model, input, displacement)
                if not physics_flag:
                    total_loss = displacement_loss
                    grads = tape.gradient(total_loss, model.network_variables)
                    self.pretrain_opt.apply_gradients(
                        zip(grads, model.network_variables)
                    )
                else:
                    total_loss = (
                        displacement_loss + self.lambda_velocity_int * velocity_loss
                    )
                    grads = tape.gradient(total_loss, model.network_variables)
                    self.network_opt.apply_gradients(
                        zip(grads, model.network_variables)
                    )
            average_training_loss += total_loss
        training_batch_num = tf.cast(training_batch_num, tf.float32)
        average_training_loss /= training_batch_num
        return average_training_loss

    def network_validate_step(self, model, validation_dataset, physics_flag=True):
        validation_batch_num = 0
        total_displacement_loss = 0.0
        total_velocity_loss = 0.0
        total_l1_loss = 0.0
        total_loss = 0.0

        for input, displacement in validation_dataset:
            validation_batch_num += 1
            (
                displacement_loss,
                velocity_loss,
                l1_loss,
            ) = self.compute_losses(model, input, displacement)
            total_displacement_loss += displacement_loss
            total_velocity_loss += velocity_loss
            total_l1_loss += l1_loss
            total_loss += displacement_loss + self.lambda_velocity_int * velocity_loss
        average_displacement_loss = total_displacement_loss / validation_batch_num
        average_velocity_loss = total_velocity_loss / validation_batch_num
        average_total_loss = total_loss / validation_batch_num

        if not physics_flag:
            return average_displacement_loss
        else:
            return (
                average_total_loss,
                average_displacement_loss,
                average_velocity_loss,
            )

    def physics_train_step(self, model, training_dataset):
        if training_dataset is None or not isinstance(
            training_dataset, tf.data.Dataset
        ):
            raise ValueError("Invalid training dataset")
        training_batch_num = 0
        average_training_loss = 0.0
        for input, displacement in training_dataset:
            training_batch_num += 1
            with tf.GradientTape() as tape:
                (
                    displacement_loss,
                    velocity_loss,
                    l1_loss,
                ) = self.compute_losses(model, input, displacement)
                total_loss = (
                    self.lambda_velocity_int * velocity_loss + self.lambda_l1 * l1_loss
                )
                grads = tape.gradient(total_loss, model.physics_variables)
                self.physics_opt.apply_gradients(zip(grads, model.physics_variables))
            average_training_loss += total_loss
        average_training_loss /= training_batch_num
        return average_training_loss

    def physics_validate_step(self, model, validation_dataset):
        validation_batch_num = 0
        total_velocity_loss = 0.0
        total_l1_loss = 0.0
        total_loss = 0.0

        for input, displacement in validation_dataset:
            validation_batch_num += 1
            (
                displacement_loss,
                velocity_loss,
                l1_loss,
            ) = self.compute_losses(model, input, displacement)
            total_velocity_loss += velocity_loss
            total_l1_loss += l1_loss
            total_loss += (
                self.lambda_velocity_int * velocity_loss + self.lambda_l1 * l1_loss
            )
        average_total_loss = total_loss / validation_batch_num
        average_velocity_loss = total_velocity_loss / validation_batch_num
        average_l1_loss = total_l1_loss / validation_batch_num

        return (
            average_total_loss,
            average_velocity_loss,
            average_l1_loss,
        )

    def load_and_visualize_model(
        self,
        model,
        validation_dataset,
        clean_data,
        figure_suffix,
    ):
        """Load best model and visualize the predictions"""
        model.load_weights(self.checkpoint_path)
        predictions = [model.predict(batch[0]) for batch in validation_dataset]
        excitation_all = tf.concat([batch[0] for batch in validation_dataset], axis=0)
        predicted_displacement_all = tf.concat(
            [pred[0] for pred in predictions], axis=0
        )
        predicted_velocity_all = tf.concat([pred[1] for pred in predictions], axis=0)

        clean_displacement_validation = clean_data["displacement_clean"]
        clean_velocity_validation = clean_data["velocity_clean"]
        plot_prediction(
            predicted_displacement_all,
            clean_displacement_validation,
            "Predicted displacement",
            f"{self.figure_dir}/displacement_hat_{figure_suffix}.png",
        )
        plot_prediction(
            predicted_velocity_all,
            clean_velocity_validation,
            "Predicted velocity",
            f"{self.figure_dir}/velocity_hat_{figure_suffix}.png",
        )
        plt.close("all")
        return (
            model,
            excitation_all,
            predicted_displacement_all,
            predicted_velocity_all,
        )

    def train_iteration(
        self,
        iteration,
        model,
        function_acceleration,
        train_dataset,
        validation_dataset,
        clean_data,
    ):
        physics_epochs = self.cfg.training.physics_epochs
        network_epochs = self.cfg.training.network_epochs
        with tf.device("/device:GPU:0"):
            physics_training_loss_log = []
            physics_validation_loss_log = []
            network_training_loss_log = []
            network_validation_loss_log = []
            """Physics training in each iteration"""
            self.logger.info(3 * "----------------------------------")
            self.logger.info(f"Iteration {iteration+1}, Physics training")
            for epoch in range(physics_epochs):
                training_loss = self.physics_train_step(model, train_dataset)
                (
                    validation_loss,
                    velocity_loss,
                    l1_loss,
                ) = self.physics_validate_step(model, validation_dataset)
                physics_training_loss_log.append(training_loss)
                physics_validation_loss_log.append(validation_loss)
                if validation_loss < self.physics_loss_minimum:
                    self.physics_loss_minimum = validation_loss
                    model.save_weights(self.checkpoint_path)
                if epoch % 100 == 0:
                    self.logger.info(
                        f"Epoch: {epoch+1}/{physics_epochs}, "
                        f"learning_rate: {self.physics_opt._decayed_lr(tf.float32):.6f}, "
                        f"Training: {training_loss:.4e}, \n"
                        f"Validation: {validation_loss:.4e}, "
                        f"velocity: {velocity_loss:.4e}, "
                        f"l1: {l1_loss:.4e}"
                    )
            plot_loss_curve(
                physics_training_loss_log,
                physics_validation_loss_log,
                f"Physics",
                f"{self.figure_dir}/physics_loss_{iteration+1}.png",
            )
            model.clip_variables()
            all_trainable_variables = model.trainable_variables
            lambda_acceleration = [
                weight for weight in all_trainable_variables if "cx" in weight.name
            ]
            function_acceleration = self.library.get_functions(lambda_acceleration)
            self.logger.info(f"Updated acceleration function: {function_acceleration}")
            function_acceleration = self.library.build_functions(lambda_acceleration)
            model.update_function(function_acceleration)

            """Network training in each iteration"""
            self.logger.info(3 * "----------------------------------")
            self.logger.info(f"Iteration {iteration+1}, Network training")
            for epoch in range(network_epochs):
                training_loss = self.network_train_step(model, train_dataset)
                (
                    validation_loss,
                    displacement_loss,
                    velocity_loss,
                ) = self.network_validate_step(model, validation_dataset)
                network_training_loss_log.append(training_loss)
                network_validation_loss_log.append(validation_loss)
                if validation_loss < self.network_loss_minimum:
                    self.network_loss_minimum = validation_loss
                    self.pretrain_loss_minimum = displacement_loss
                    model.save_weights(self.checkpoint_path)
                if epoch % 100 == 0:
                    self.logger.info(
                        f"Epoch: {epoch+1}/{network_epochs}, "
                        f"learning_rate: {self.network_opt._decayed_lr(tf.float32):.6f}, "
                        f"Training: {training_loss:.4e}, \n"
                        f"Validation: {validation_loss:.4e}, "
                        f"displacement: {displacement_loss:.4e}, "
                        f"velocity: {velocity_loss:.4e}, "
                    )
            plot_loss_curve(
                network_training_loss_log,
                network_validation_loss_log,
                f"Network",
                f"{self.figure_dir}/network_loss_{iteration+1}.png",
            )
            (
                model,
                excitation_all,
                predicted_displacement_all,
                predicted_velocity_all,
            ) = self.load_and_visualize_model(
                model,
                validation_dataset,
                clean_data,
                f"AO_{iteration+1}",
            )
        return (
            model,
            excitation_all,
            predicted_displacement_all,
            predicted_velocity_all,
        )

    def training(
        self,
        train_dataset,
        validation_dataset,
        Phi_int,
        Phi_diff,
        clean_data,
        max_values,
    ):
        lambda_acceleration = np.ones(self.library.terms_number)
        function_acceleration = self.library.build_functions(lambda_acceleration)
        model = PNetwork(
            self.cfg,
            Phi_int=Phi_int,
            Phi_diff=Phi_diff,
            number_library_terms=self.library.terms_number,
            function_acceleration=function_acceleration,
            max_values=max_values,
        )
        """Pretrain the displacement model"""
        self.logger.info("Pretrain the displacement network")
        model.update_function(function_acceleration)
        pretrain_epochs = self.cfg.training.pretrain_epochs
        with tf.device("/device:GPU:0"):
            pretrain_training_loss_log = []
            pretrain_validation_loss_log = []
            for epoch in range(pretrain_epochs):
                training_loss = self.network_train_step(model, train_dataset, False)
                validation_loss = self.network_validate_step(
                    model, validation_dataset, False
                )
                pretrain_training_loss_log.append(training_loss)
                pretrain_validation_loss_log.append(validation_loss)
                if validation_loss < self.pretrain_loss_minimum:
                    self.pretrain_loss_minimum = validation_loss
                    model.save_weights(self.checkpoint_path)
                if epoch % 100 == 0:
                    self.logger.info(
                        f"Epoch: {epoch+1}/{pretrain_epochs}, "
                        f"learning_rate: {self.pretrain_opt._decayed_lr(tf.float32).numpy():.6f}, "
                        f"Training: {training_loss:.4e}, "
                        f"Validation: {validation_loss:.4e}, "
                    )
            plot_loss_curve(
                pretrain_training_loss_log,
                pretrain_validation_loss_log,
                f"Pretrain",
                f"{self.figure_dir}/pretrain_loss.png",
            )
            (model, excitation_all, displacement_all, velocity_hat_all) = (
                self.load_and_visualize_model(
                    model,
                    validation_dataset,
                    clean_data,
                    f"pretrain",
                )
            )

        """Alternate between physics constrained training and network training"""
        alternate_number = self.cfg.training.alternate_number
        for iteration in range(alternate_number):
            (
                model,
                excitation_all,
                displacement_all,
                velocity_hat_all,
            ) = self.train_iteration(
                iteration,
                model,
                function_acceleration,
                train_dataset,
                validation_dataset,
                clean_data,
            )

        return {
            "model": model,
            "excitation_all": excitation_all,
            "displacement_all": displacement_all,
            "velocity_hat_all": velocity_hat_all,
        }
