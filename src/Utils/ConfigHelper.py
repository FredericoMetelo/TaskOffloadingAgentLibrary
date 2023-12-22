import numpy as np


def minimumTaskProcessingPower(taskArrivalRate, taskTypeProbabilities, taskSizes, expectedNeighbours,  maxTimeSteps, noCores=4):
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

    # From the task rates get the mean number of tasks arriving per time-step. Because the picking of a task type is independent
    # of the task arrival we have P_taskType_arriving = P_taskType_arriving * P_task_arriving
    meanTaskTypesArriving = taskArrivalRate * taskTypeProbabilities
    meanNumberOfInstructions = np.sum(taskSizes * taskTypeProbabilities)/taskTypeProbabilities.shape[0]

    # To process the expected number of tasks arriving we have in the time specified we have:
    requiredTimeToProcess = meanNumberOfInstructions / (expectedNeighbours * maxTimeSteps)

    # This is me confirming I didn't make a blunder. If I have meanNumberOfInstructions total expected instructions to
    # process and I have maxTimeSteps to do it, then I need to process meanNumberOfInstructions/maxTimeSteps per time-step.
    # But if I have noCores cores per CPU to process the tasks, then I need to process
    # meanNumberOfInstructions/(maxTimeSteps*noCores) per core per time-step.

    return requiredTimeToProcess