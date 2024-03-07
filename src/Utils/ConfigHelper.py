import math

import numpy as np


def minimumTaskProcessingPower(taskArrivalRate, taskTypeProbabilities, taskSizes, expectedSessions, maxTimeSteps,
                               noCores=4):
    """
    This function helps compute the minimum number of processing power to clear all tasks arriving in the system
    at a time-step before a specified number of time-steps.

    :param taskArrivalRate: The rate at which tasks arrive in the system, the arrival rate of a Poisson Process.
    :param taskSizes: The size of the tasks arriving in the system. Can be an array of sizes if there are multiple task types.
    :param taskTypeProbabilities: The probabilities of each task type arriving whenever a task arrives in the system.
    :param : The expected number of neighbours each node has.
    :param maxTimeSteps: The maximum number of time-steps to consider.
    :param noCores: The Cores the CPU's will have.
    :return: The processing power per CPU required to clear all tasks arriving in the system at a time-step before a specified
    """

    # From the task rates get the mean number of tasks arriving per time-step. Because the picking of a task type is
    # independent of the task arrival we have P_taskType_arriving = P_taskType_arriving * P_task_arriving
    meanTaskTypesArriving = taskArrivalRate * taskTypeProbabilities
    meanNumberOfInstructionsPerClient = np.sum(meanTaskTypesArriving) / taskTypeProbabilities.shape[0]

    # To process the expected number of tasks arriving we have in the time specified we have to process:
    requiredInstructionsPerTimeStep = (meanNumberOfInstructionsPerClient * expectedSessions) / (noCores * maxTimeSteps)

    # This is me confirming I didn't make a blunder. If I have meanNumberOfInstructionsPerClient total expected
    # instructions to process and I have maxTimeSteps to do it, then I need to process
    # meanNumberOfInstructionsPerClient/maxTimeSteps per time-step. But if I have noCores cores per CPU to process
    # the tasks, then I need to process meanNumberOfInstructionsPerClient/(maxTimeSteps*noCores) per core per time-step.

    return requiredInstructionsPerTimeStep


def taskArrivalForOccupancy(occupancy, taskTypeProbabilities, taskSizes, numClients, targetTimeForOccupancy,
                            processingPower, maxQueueSize):
    """
    Want to compute the average task arrival for a node to be at a certain occupancy level at a certain time-step. Given
    the task sizes, number of clients and the worker's processing power.
    O - Occupancy
    lambda - Task arrival Rate
    P - Processing power
    c - Number of clients
    Qt - Queue size at time t
    Qm - Maximum queue size
    T - Average Task Size

    Occupancy = Qt/Qm - Percentage of the queue filled

    The Q size vaires from time-step to time step by:
    Qt = Qt-1 + lambda*c - T/P

    therefore:
    Qn = n(lambda*c - T/P) + Q0, but, because Q0 = 0 we have Qn = n(lambda*c - T/P)

    Therfore if we want to have an occupancy of O at time-step n we have:
    Qn = O*Qm = n(lambda*c - T/P) (=) lambda = (O*Qm/n + T/P)/c
    :param occupancy:
    :param taskTypeProbabilities:
    :param taskSizes:
    :param numClients:
    :param targetTimeForOccupancy:
    :param processingPower:
    :return: lambda - The task arrival rate required to have an occupancy of O at time-step n
    """
    # Get the mean task size
    meanTaskSize = np.sum(taskSizes * taskTypeProbabilities)
    lam = (occupancy * maxQueueSize / targetTimeForOccupancy + meanTaskSize / processingPower) / numClients

    if lam > 1:
        print("Warning: Lambda is 1, this means the specifications are not possible")

    return min(lam, 1)


def average_points_in_circle(radius, num_generated_points):
    """
    A uniform distribution has equal probability of creating a point for every coordinate in a circle. Therefore, the
     chance of it being inside the communications radius is the area of the circle divided by the area of the square.
     But this value is only for a node in the middle of the square, the other extreme would be a point that is in one of
     the corners, this circle would have 1/4 of the area of the circle in the middle inside the square. Therefore, to
     account for this we average the expected number of nodes on a node on each corner and in the middle.
    :param radius:
    :param num_generated_points:
    :return:
    """
    total_area = 10000
    circle_area = math.pi * radius**2
    probability_point_inside = circle_area / total_area
    average_points_from_middle = num_generated_points * probability_point_inside
    average_points_from_corner = average_points_from_middle/2
    average_points_inside = average_points_from_middle * 4 / 5
    return round(average_points_inside)

def generate_config_dict(controllers="[0]",
                         size=10,
                         simulation_time=1000,
                         radius=50,
                         frequency_of_action=5,

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
                         expected_occupancy=0.5,
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
                         clientLayers="0"
                        ):
    if size != sum(nodes_per_layer):
        raise Exception("Size and sum of nodes per layer must be equal")
    # Press the green button in the gutter to run the script.
    total_cpu_cycles = [a * b for a, b in zip(task_CPI, task_instr)]
    avg_neighbours = average_points_in_circle(radius, nodes_per_layer[0])
    configs = {
        "SIZE": str(size),
        "CYCLE": "1",
        "CYCLES": str(simulation_time),
        "random.seed": "1234567890",
        "MINDELAY": "0",
        "MAXDELAY": "0",
        "DROP": "0",
        "CONTROLLERS": make_ctr(controllers),


        "CLOUD_EXISTS": str(has_cloud),
        "NO_LAYERS": str(len(nodes_per_layer)),
        "NO_NODES_PER_LAYERS": to_string_array(nodes_per_layer),
        "CLOUD_ACCESS": to_string_array(cloud_access),

        "FREQS": to_string_array(freqs_per_layer),
        "NO_CORES": to_string_array(no_cores_per_layer),
        "Q_MAX": to_string_array(q_max_per_layer),
        "VARIATIONS": to_string_array(variations_per_layer),

        "protocol.cld.no_vms": str(len(cloud_VM_processing_power)),
        "protocol.cld.VMProcessingPower": to_string_array(cloud_VM_processing_power),

        "init.Net1.r": str(radius),

        "protocol.mng.r_u": str(weight_utility),
        "protocol.mng.X_d": str(weight_delay),

        "protocol.mng.X_o": str(weight_overload),
        "protocol.mng.cycle": str(frequency_of_action),

        "protocol.clt.numberOfTasks": str(len(task_probs)),
        "protocol.clt.weight": to_string_array(task_probs, print_as_int=False),
        "protocol.clt.CPI": to_string_array(task_CPI),
        "protocol.clt.T": to_string_array(task_sizes),
        "protocol.clt.I": to_string_array(task_instr),
        "protocol.clt.taskArrivalRate": str(expected_occupancy),

        "protocol.clt.numberOfDAG": "1",
        "protocol.clt.dagWeights": "1",
        "protocol.clt.edges": "",
        "protocol.clt.maxDeadline": to_string_array(task_deadlines),
        "protocol.clt.vertices": "1",
        "protocol.clt.layersThatGetTasks": to_string_array(layersThatGetTasks),


        "protocol.props.B": str(comm_B),
        "protocol.props.Beta1": str(comm_Beta1),
        "protocol.props.Beta2": str(comm_Beta2),
        "protocol.props.P_ti": str(comm_Power),
        "RANDOMIZEPOSITIONS": str(RANDOMIZEPOSITIONS),
        "init.Net0.POSITIONS": POSITIONS,

        "RANDOMIZETOPOLOGY": str(RANDOMIZETOPOLOGY),
        "init.Net1.TOPOLOGY": TOPOLOGY,

        "MANUAL_CONFIG": str(MANUAL_CONFIG),
        "MANUAL_CORES": MANUAL_CORES,
        "MANUAL_FREQS": MANUAL_FREQS,
        "MANUAL_QMAX": MANUAL_QMAX,
        "clientLayers": clientLayers,

        "protocol.clt.defaultCPUWorkload": "100000000",
        "protocol.clt.defaultMemoryWorkload": "100",
        "protocol.clt.workloadPath": "/home/fm/IdeaProjects/peersim-environment/Datasets/alibaba_trace_cleaned.json",
    }
    return configs

# taskArrivalForOccupancy(expected_occupancy, np.array(task_probs), np.array(total_cpu_cycles),
#                                                                 avg_neighbours, simulation_time*target_time_for_occupancy,
#                                                                 total_cpu_cycles[0], q_max_per_layer[0])

def make_ctr(ctrs_list):

    return to_string_array(ctrs_list, ";")


def to_string_array(arr, separator=",", print_as_int=True):
    s = ""
    for i in range(len(arr)):
        s += str(int(arr[i]) if print_as_int else arr[i])
        if i < len(arr) - 1:
            s += separator
    return s
