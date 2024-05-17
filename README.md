# Introduction

This repository acts as an extension to the Peersim simulator, providing a reinforcement learning environment for the simulator. The environment is designed to be used with the PettingZoo API, allowing for easy integration with existing reinforcement learning libraries.
In here we provide a set of RL agents (`src/MARL`) and baseline algorithms (`src/ControlAlgorithms`) to be used as examples.

The simulator can be found in the [simulator repository](https://anonymous.4open.science/anonymize/peersim-environment-CE2A) (anonymized for now) 

For a full showcase of how to use the environment and the agents provided see the `src/DebugEnv.py`

# Configuring the Environment

The environment is highly customizable a full list with detailed descriptions of the possible configurations can be found in the simulator repository, to facilitate the configuration process we provide a configuration helper in the `src/EnvUtils/ConfigHelper.py`. The helper can be used to generate a configuration dictionary that can be used to create the environment.
The helper generates a configurationn dictionary that can then be passed to the environment creation function. The helper can be used as follows:
```python
import src.EnvUtils.ConfigHelper as ch

config_dict = ch.generate_config_dict(controllers="[0]",
                         size=10,
                         simulation_time=1000,
                         radius=50,
                         frequency_of_action=2,

                         has_cloud=1,
                         cloud_VM_processing_power=[1e8],

                         nodes_per_layer=[10],
                         cloud_access=[0],
                         freqs_per_layer=[1e7],
                         no_cores_per_layer=[4],
                         q_max_per_layer=[50],
                         variations_per_layer=[0],
                         layersThatGetTasks=[0],

                         task_probs=[1],
                         task_sizes=[150],
                         task_instr=[4e7],
                         task_CPI=[1],
                         task_deadlines=[100],
                         lambda_task_arrival_rate=0.5,
                         target_time_for_occupancy=0.5,

                         comm_B=2,
                         comm_Beta1=0.001,
                         comm_Beta2=4,
                         comm_Power=20,

                         weight_utility=10,
                         weight_delay=1,
                         weight_overload=150,
                         RANDOMIZETOPOLOGY=True,
                         RANDOMIZEPOSITIONS=True,
                         POSITIONS="18.55895350495783,17.02475796027715;47.56499372388999,57.28732691557995;5.366872150976409,43.28729893321355;17.488160666668694,29.422819514162434;81.56549175388358,53.14564532018814;85.15660881172089,74.47408014762478;18.438454887921974,44.310130148722195;72.04311826903107,62.06952644109185;25.60125368295145,15.54795598202745;17.543669122835837,70.7258178169151",
                         TOPOLOGY="0,1,2,3,6,8;1,0,2,3,4,5,6,7,8,9;2,0,1,3,6,8,9;3,0,1,2,6,8,9;4,1,5,7;5,1,4,7;6,0,1,2,3,8,9;7,1,4,5;8,0,1,2,3,6;9,1,2,3,6",
                         MANUAL_CONFIG=False,
                         MANUAL_CORES="1",
                         MANUAL_FREQS="1e7",
                         MANUAL_QMAX="10",
                         clientLayers="0",
                         defaultCPUWorkload="2.4e+9",
                         defaultMemoryWorkload="100",
                         workloadPath=None,
                         clientIsSelf=1
                         )
env = PeersimEnv(configs=config_dict, ...)

```
### Configuration Options
There are many ways of using the simulator tool, we will leave the details on what each variable does for the simulator repository, where we go in detail on each one. But we will list here the variables required to activate each of the modes, note that with this values we would be using the default values adn more information on how to use the plausible configurations can be found in the 
 following sections.

We separate the main configurations in the topology options and the task options, options for each category can be picked interchangeably (I.E. it's possible to select the ether topology and the random task generation). The main configurations are:
- **Topology Configuration**: The configuration of the topology of the network.
  - **Random Topology**: The configuration of the random topology that will be generated. The variables required to use this mode are:
      ```python
      RANDOMIZETOPOLOGY=True,
      RANDOMIZEPOSITIONS=True,
      size=<NUMBER OF NODES>,
    ```
  - **Manual Topology**: The configuration of the manual topology that will be used.
    ```python
    RANDOMIZETOPOLOGY=False,
    RANDOMIZEPOSITIONS=False,
    POSITIONS=[<POSITIONS>],
    TOPOLOGY=[<TOPOLOGY>],
    ```
  - **Ether Topology**: The configuration of the ether topology that will be used.
    ```python
    RANDOMIZETOPOLOGY=False,
    RANDOMIZEPOSITIONS=False,
    MANUAL_CONFIG=True,
    MANUAL_CORES=[<CORES>],
    MANUAL_FREQS=[<FREQUENCIES>],
    MANUAL_QMAX=[<QMAX>],
    POSITIONS=[<POSITIONS>],
    TOPOLOGY=[<TOPOLOGY>],
    
    ```
- **Task Configuration**: The configuration of the tasks that will be generated in the simulation. The task generation does not require special configurations, but for the simulaiton mode to be set, which we will show how to do in following sections.
    - **Random Task Generation**: The configuration of the random task generation that will be used.
    - **Trace Task Generation**: The configuration of the trace task generation that will be used. Requires a special simulaiton type to be set. 
    
  

## Selecting the task generation mode
There are two modes for task generation, one based on the trace-generation tool and another that uses a set of potential 
user defined tasks. To change between the different modes set the environment to be created with one of two options:
```python

config_dict = ch.generate_config_dict(..., 
                                      workloadPath="<PATH TO TRACE DATASET FILE>",
                                      ...)

simtype = "basic" # Uses the user defined workload generation
simtype = "basic-workload" # Uses the trace generation tool

...
env = PeersimEnv(configs=config_dict, ..., simtype=simtype, ...) 
```

Notice, that you must set the configuration indicating what is the file containing the required dataset. We recommend using the configuration helper to generate the configuration dictionary, for manually generating a file see the .
## Generating your own workload from the trace generation tool

> This instructions require the repository for the PeersimEnv Simulator, that can be found in the simulator repository.

If you wish to generate a dataset of a different size or characteristics than the one provided, you can use the trace generation tool to generate a new dataset.

Set up the trace generation tool following the instructions in the [Trace Generation repository](https://github.com/All-less/trace-generator.git). We only require the file generated by the tool, hence we recommend following the `pip3` installation.
After installig, follow the repo's instructions on generating the datasets, move the datasets generated to the `/Datasets/` directory in the simulator repository.

We then convert the tool to be usable with PeerSim by running the script from the `src/EnvUtils/AlibabaTraceGeneratorCleaner.py` in the simulator repository. This will create the file `Datasets/alibaba_trace_cleaned.json` that can be used in the environment.


## Using the Ether generated topology:
### Generating the topology
The process to generate teh ether topology requires using the `ether` tool. The tool is available in the [Ether repository](https://github.com/edgerun/ether.git).
To generate the topology, copy and past the contents of the folder `src/Utils/EtherTopologyGeneration` into the ether repository
root. Then, from the ether repository run the python script `examples/MyNet.py`. This will generate a file called `topology.json`
that can be used in the environment.

### Using the generated topology in the environment
Using the generated topology in the environment requires a special set of configuration values to be set. To help with reading this values from the configuration file to the environment we provide a helper function in the `src/EnvUtils/EtherTopologyReader.py`. The function reads the topology file and returns a dictionary with the values that can be used in the configuration helper. The function can be used as follows:
```python 
    import src.EnvUtils.EtherTopologyReader as etr

    topology_file = "./EetherTopologies/one_cluste_8rpi_manual.json"
    topology_dict = etr.get_topology_data(topology_file, project_coordinates=True, expected_task_size=32e7)
    
    manual_config = True
    manual_no_layers = topology_dict["number_of_layers"]
    manual_layers_that_get_tasks = topology_dict["layers_that_get_tasks"]
    manual_clientLayers = topology_dict["client_layers"]
    manual_no_nodes = topology_dict["number_of_nodes"]
    manual_nodes_per_layer = topology_dict["nodes_per_layer"]
    manual_freqs = topology_dict["processing_powers"]
    manual_freqs_array = topology_dict["freqs_per_layer_array"]
    manual_qmax = topology_dict["memories"]
    manual_qmax_array = topology_dict["q_max_per_layer_array"]
    manual_cores = topology_dict["cores"]
    manual_cores_array = topology_dict["no_cores_per_layer_array"]
    manual_variations = topology_dict["variations_per_layer_array"]
    manual_positions = topology_dict["positions"]
    manual_topology = topology_dict["topology"]
    controllers = topology_dict["controllers"] #  ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


    config_dict = ch.generate_config_dict(...,
                                          # This variables indicate to the simulation that a manual topology configuration is being used
                                          MANUAL_CONFIG=manual_config,
                                          RANDOMIZETOPOLOGY=False,
                                          RANDOMIZEPOSITIONS=False,
                                         
                                          # This are the values extracted from the file generated with the simulation topology data
                                          size=manual_no_nodes,
                                          nodes_per_layer=manual_nodes_per_layer,
                                          freqs_per_layer=manual_freqs_array,
                                          no_cores_per_layer=manual_cores_array,
                                          q_max_per_layer=manual_qmax_array,
                                          variations_per_layer=manual_variations,

                                          MANUAL_QMAX=manual_qmax,
                                          MANUAL_CORES=manual_cores,
                                          MANUAL_FREQS=manual_freqs,

                                          
                                          layersThatGetTasks=manual_layers_that_get_tasks,
                                          clientLayers=manual_clientLayers,
                                          
                                          POSITIONS=manual_positions,
                                          TOPOLOGY=manual_topology,
                                          ...
                                          )
    env = PeersimEnv(configs=config_dict, ...)
    
```
## Adding a reward shaping function
Reward Shaping is an useful tool to speed up the reinforcement learning process. We provide a mechanism to pass a reward shaping term to the environment.
A default function can be found in the `src/EnvUtils/RewardShaping.py` file. The function must take the state as an argument and return the reward. The function can be added to the environment by setting the phy_rs_term parameter as follows:
```python
import src.EnvUtils.RewardShaping as rs

def reward_shaping(state):
    # Add your reward shaping logic here, we assume output is stored in reward_shaping_term
    return reward_shaping_term

env = PeersimEnv(..., phy_rs_term=reward_shaping, ...)
```

## Visualizing the environment
The environment provides three visualization modes, that can be set with the render_mode parameter. The modes are:
- `None`: Minimum information is printed to stdout.
- `"ascii"`: We print extra information on the state of the simulation to stdout.
- `"human"`: We provide a visual representation of the simulation using a pygame canvas.

The visualization can be set as follows:
```python
render_mode = None # or
render_mode = "ascii" # or
render_mode = "human"

env = PeersimEnv(..., render_mode=render_mode, ...)
```

A video explaining the "human" rendering mode can be observed on the following video (also available in the repository as FinalVisualization.mp4):
[![Peersim Visualization: https://www.youtube.com/watch?v=sgDV2Ytavk4] (https://i9.ytimg.com/vi_webp/sgDV2Ytavk4/mq2.webp?sqp=CNy-nbIG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGE0gZShEMA8=&rs=AOn4CLCRFxna9ssJ_2B3UETpg4MUuG8P1Q)](https://www.youtube.com/watch?v=sgDV2Ytavk4 "PeersimGym Visualization")

# Bibtex
@article{metelo2024peersimgym,
  title={PeersimGym: An Environment for Solving the Task Offloading Problem with Reinforcement Learning}, 
  author={Frederico Metelo and Stevo Racković and Pedro Ákos Costa and Cláudia Soares},
  year={2024}, eprint={2403.17637},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
