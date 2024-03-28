from src.ControlAlgorithms.ControlAlgorithm import ControlAlgorithm


class ManualSelection(ControlAlgorithm):
    """
    The RandomControlAlgorithm will offload data randomly. It will select both the node and the amount of data to be
    offloaded randomly.
    """

    def __init__(self, action_space, output_shape, input_shape, agents, clip_rewards=False, collect_data=False,
                 plot_name=None, file_name=None):
        super().__init__(action_space=action_space, output_shape=output_shape, input_shape=input_shape, agents=agents,
                         clip_rewards=clip_rewards,
                         collect_data=collect_data, plot_name=plot_name, file_name=file_name)
        self.control_type = "Manual"

    @property
    def control_type(self):
        return "Manual"

    @control_type.setter
    def control_type(self, value):
        self._control_type = value

    def select_action(self, observation, agents):
        if input("all?") == "y":
            a = int(input("Enter the node to offload to: "))
            action = {agent: a for agent in agents}
        else:
            action = {
                agent: int(input(f"{agent} | Enter the node to offload to: "))
                for agent in agents
            }
        action_type = self.control_type
        return action, action_type

    @control_type.setter
    def control_type(self, value):
        self._control_type = value
