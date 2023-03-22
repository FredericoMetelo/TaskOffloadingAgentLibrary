from ray.rllib.models.tf import TFModelV2
import tensorflow as tf


class DQNModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DQNModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        # Process the data:
        # self.observation_space = Dict(
        # {
        #     "n_i": Discrete(number_nodes, start=1),
        #     "Q": MultiDiscrete(q_list),
        #     "w": Box(high=max_w, low=0, dtype=np.float)
        #    }
        original_space = obs_space

        self.input = tf.keras.layers.Input(shape=original_space)
