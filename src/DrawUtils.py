'''
This file has a set uf utilities to help interactively draw plots over the results and data.
'''
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import ast
import re

import matplotlib.pyplot as plt


def read_cvs(filename):
    # Open the CSV file
    with open(f'{filename}', 'r') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Convert the CSV data to a list of lists
        data = list(csv_reader)
    return data


def process_string_array_entry(array):
    return re.sub(',+', ',', re.sub(' +', ',', array.replace('\n', ' ')).replace('[,', '[').replace(',]', ']'))


def plot_lines(x_values, y_values, y_labels, plot_title, x_axis_label, y_axis_label, convert_to_float=False, plot_dots=False, prefix_plot_png=""):
    # Convert the y values to float arrays
    if convert_to_float:
        for idx in range(len(y_values)):
            y_values[idx] = np.array(y_values[idx], dtype=float)

    # Create a plot
    if len(y_values) != len(y_labels):
        raise Exception('The number of labels and values must be the same')

    for idx in range(len(y_values)):
        if plot_dots:
            plt.scatter(x_values, y_values[idx])
        plt.plot(x_values, y_values[idx], label=y_labels[idx])

    # Add a title and labels for the axes
    plt.title(plot_title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    # Show a legend to label the lines
    plt.legend()
    plt.savefig(f'./Plots/Output/{plot_title + prefix_plot_png}.png')
    # Display the plot
    plt.show()


def plot_lines_fill_between(x_values, y_values, y_labels, plot_title, x_axis_label, y_axis_label,
                            convert_to_float=False):
    # Convert the y values to float arrays
    if convert_to_float:
        for idx in range(len(y_values)):
            y_values[idx] = np.array(y_values[idx], dtype=float)

    # Create a plot
    if len(y_values) != len(y_labels):
        raise Exception('The number of labels and values must be the same')

    for idx in range(len(y_values)):
        plt.fill_between(x_values, y_values[idx][1], -y_values[idx][1], alpha=.5, label=y_labels[idx])
        plt.plot(x_values, y_values[idx][0], label=y_labels[idx])

    # Add a title and labels for the axes
    plt.title(plot_title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    # Show a legend to label the lines
    plt.legend()
    # Display the plot
    plt.show()


def plot_all_complete_percentage_in_dir(dir, cores, lambdas):
    csv_files = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]
    plt.figure()
    percentages = {
        str(c): {} for c in cores
    }
    for file in csv_files:
        path = os.path.join(dir, file)
        data_results = read_cvs(path)
        data = data_results[1:]
        x = range(len(data))
        total_tasks = [np.sum(ast.literal_eval(process_string_array_entry(row[5]))) for row in data]
        finished_tasks = [np.sum(ast.literal_eval(process_string_array_entry(row[4]))) for row in data]

        percentage_finished_tasks = np.average(
            [finished / total for finished, total in zip(finished_tasks, total_tasks)])
        info = file.replace(".csv", "").replace("_result_metrics", "").replace("least_queue_", "").split("_")
        c = info[0].replace("c", "")
        oc = info[1].replace("oc", "")
        percentages[c][oc] = percentage_finished_tasks

    x = lambdas
    for c in cores:
        # Plot the data
        data = []
        for lamb in lambdas:
            data.append(percentages[str(c)][str(lamb)])
        plt.scatter(x, data, marker='x')
        plt.plot(x, data, label=c)

    # Add labels and legend to the plot
    plt.xlabel('Episode')
    plt.ylabel('Completion Percentage')
    plt.legend()
    # for file in csv_files:
    #     path = os.path.join(dir, file)
    #     data_results = read_cvs(path)
    #     data = data_results[1:]
    #     x = range(len(data))
    #     total_tasks = [np.sum(ast.literal_eval(process_string_array_entry(row[5]))) for row in data]
    #     # Plot the data
    #     plt.plot(x, total_tasks, label=file.replace(".csv", "").replace("_result_metrics", "").replace("least_queue_", " "))
    # plt.xlabel('Episode')
    # plt.ylabel('Total Tasks')
    # plt.legend()
    # plt.show()
    #
    # for file in csv_files:
    #     path = os.path.join(dir, file)
    #     data_results = read_cvs(path)
    #     data = data_results[1:]
    #     x = range(len(data))
    #     finished_tasks = [np.sum(ast.literal_eval(process_string_array_entry(row[4]))) for row in data]
    #
    #
    #     # Plot the data
    #     plt.plot(x, finished_tasks,
    #              label=file.replace(".csv", "").replace("_result_metrics", "").replace("least_queue_", " "))
    # plt.xlabel('Episode')
    # plt.ylabel('Completed Tasks')
    # plt.legend()
    # plt.show()

    # Display the plot
    plt.show()
    return


def plot_rewards(ddqn):
    # Read the data from the CSV file
    ddqn_result = read_cvs(ddqn)
    # Extract the headers and the data
    headers = ddqn_result[0]
    ddqn_data = ddqn_result[1:]
    agent_list = [header for header in headers if 'reward' in header]
    agent_rewards = {agent: [] for agent in agent_list}
    # Separate the data and pre-process it

    for i in range(len(ddqn_data)):
        for data, header in zip(ddqn_data[i], headers):
            if 'reward' in header:
                agent_rewards[header].append(np.mean(ast.literal_eval(process_string_array_entry(data))))
    # Generate X axis
    x = range(len(ddqn_data))
    # Add all the dataon the same plot to a matrix
    y_labels = agent_list
    rewards_matrix = [agent_rewards[agent] for agent in agent_list]
    # Convert the data to numpy arrays
    plot_lines(x_values=x, y_values=rewards_matrix, y_labels=y_labels, plot_title='Rewards',
               x_axis_label='Episodes', y_axis_label='Rewards', convert_to_float=True)


def plot_results(least_queues, random, always_local, ddqn):
    # Read the data from the CSV file
    least_queues_result = read_cvs(least_queues)
    random_result = read_cvs(random)
    always_local_result = read_cvs(always_local)
    ddqn_result = read_cvs(ddqn)

    # Extract the headers and the data
    headers = least_queues_result[0]
    least_queues_data = least_queues_result[1:]
    random_data = random_result[1:]
    always_local_data = always_local_result[1:]
    ddqn_data = ddqn_result[1:]

    least_queues_occupancy = []
    random_occupancy = []
    always_local_occupancy = []
    ddqn_occupancy = []

    least_queues_response_time = []
    random_response_time = []
    always_local_response_time = []
    ddqn_response_time = []

    least_queues_overloaded = []
    random_overloaded = []
    always_local_overloaded = []
    ddqn_overloaded = []

    least_queues_dropped_Tasks = []
    random_dropped_Tasks = []
    always_local_dropped_Tasks = []
    ddqn_dropped_Tasks = []

    least_queues_finished_Tasks = []
    random_finished_Tasks = []
    always_local_finished_Tasks = []
    ddqn_finished_Tasks = []

    least_queues_total_Tasks = []
    random_total_Tasks = []
    always_local_total_Tasks = []
    ddqn_total_Tasks = []

    least_queues_percentage_dropped_tasks = []
    random_percentage_dropped_tasks = []
    always_local_percentage_dropped_tasks = []
    ddqn_percentage_dropped_tasks = []

    least_queues_percentage_finished_tasks = []
    random_percentage_finished_tasks = []
    always_local_percentage_finished_tasks = []
    ddqn_percentage_finished_tasks = []

    # Separate the data and pre-process it
    for i in range(len(least_queues_data)):
        try:
            # Pre-process the into three arrays, occupancy, overloads and response time for each type of agent
            least_queues_overloaded.append(
                np.mean(ast.literal_eval(process_string_array_entry(least_queues_data[i][0]))))
            random_overloaded.append(np.mean(ast.literal_eval(process_string_array_entry(random_data[i][0]))))
            always_local_overloaded.append(
                np.mean(ast.literal_eval(process_string_array_entry(always_local_data[i][0]))))
            ddqn_overloaded.append(np.mean(ast.literal_eval(process_string_array_entry(ddqn_data[i][0]))))

            least_queues_occupancy.append(
                np.mean(ast.literal_eval(process_string_array_entry(least_queues_data[i][1]))))
            random_occupancy.append(np.mean(ast.literal_eval(process_string_array_entry(random_data[i][1]))))
            always_local_occupancy.append(
                np.mean(ast.literal_eval(process_string_array_entry(always_local_data[i][1]))))
            ddqn_occupancy.append(np.mean(ast.literal_eval(process_string_array_entry(ddqn_data[i][1]))))

            least_queues_response_time.append(
                np.mean(ast.literal_eval(process_string_array_entry(least_queues_data[i][2]))))
            random_response_time.append(np.mean(ast.literal_eval(process_string_array_entry(random_data[i][2]))))
            always_local_response_time.append(
                np.mean(ast.literal_eval(process_string_array_entry(always_local_data[i][2]))))
            ddqn_response_time.append(np.mean(ast.literal_eval(process_string_array_entry(ddqn_data[i][2]))))

            least_queues_dropped_Tasks.append(
                np.sum(ast.literal_eval(process_string_array_entry(least_queues_data[i][3]))))
            random_dropped_Tasks.append(np.sum(ast.literal_eval(process_string_array_entry(random_data[i][3]))))
            always_local_dropped_Tasks.append(
                np.sum(ast.literal_eval(process_string_array_entry(always_local_data[i][3]))))
            ddqn_dropped_Tasks.append(np.sum(ast.literal_eval(process_string_array_entry(ddqn_data[i][3]))))

            least_queues_finished_Tasks.append(
                np.sum(ast.literal_eval(process_string_array_entry(least_queues_data[i][4]))))
            random_finished_Tasks.append(np.sum(ast.literal_eval(process_string_array_entry(random_data[i][4]))))
            always_local_finished_Tasks.append(
                np.sum(ast.literal_eval(process_string_array_entry(always_local_data[i][4]))))
            ddqn_finished_Tasks.append(np.sum(ast.literal_eval(process_string_array_entry(ddqn_data[i][4]))))

            least_queues_total_Tasks.append(
                np.sum(ast.literal_eval(process_string_array_entry(least_queues_data[i][5]))))
            random_total_Tasks.append(np.sum(ast.literal_eval(process_string_array_entry(random_data[i][5]))))
            always_local_total_Tasks.append(
                np.sum(ast.literal_eval(process_string_array_entry(always_local_data[i][5]))))
            ddqn_total_Tasks.append(np.sum(ast.literal_eval(process_string_array_entry(ddqn_data[i][5]))))

            # Compute the percentage of tasks dropped and completed
            least_queues_percentage_dropped_tasks.append(least_queues_dropped_Tasks[i] / least_queues_total_Tasks[i])
            random_percentage_dropped_tasks.append(random_dropped_Tasks[i] / random_total_Tasks[i])
            always_local_percentage_dropped_tasks.append(always_local_dropped_Tasks[i] / always_local_total_Tasks[i])
            ddqn_percentage_dropped_tasks.append(ddqn_dropped_Tasks[i] / ddqn_total_Tasks[i])

            least_queues_percentage_finished_tasks.append(least_queues_finished_Tasks[i] / least_queues_total_Tasks[i])
            random_percentage_finished_tasks.append(random_finished_Tasks[i] / random_total_Tasks[i])
            always_local_percentage_finished_tasks.append(always_local_finished_Tasks[i] / always_local_total_Tasks[i])
            ddqn_percentage_finished_tasks.append(ddqn_finished_Tasks[i] / ddqn_total_Tasks[i])


        except Exception as e:
            print(f"Error on line {i}")
            print(f'Least Queues: {least_queues_data[i]}')
            print(f'Random Data : {random_data[i]}')
            print(f'Alwasy local: {always_local_data[i]}')
            print(f'DDQN: {ddqn_data[i]}')
            input('Failed here, continue?')

        # ddqn_data[i] = list(map(ast.literal_eval, ddqn_data[i]))
    # Generate X axis
    x = range(len(least_queues_data))
    # Add all the dataon the same plot to a matrix
    y_labels = ['Least Queue', 'Random Offloading', 'Local Processing', 'DDQN']
    occupancy_matrix = [least_queues_occupancy, random_occupancy, always_local_occupancy, ddqn_occupancy]
    response_time_matrix = [least_queues_response_time, random_response_time, always_local_response_time,
                            ddqn_response_time]
    overloaded_matrix = [least_queues_overloaded, random_overloaded, always_local_overloaded, ddqn_overloaded]

    dropped_tasks_matrix = [least_queues_dropped_Tasks, random_dropped_Tasks, always_local_dropped_Tasks,
                            ddqn_dropped_Tasks]
    finished_tasks_matrix = [least_queues_finished_Tasks, random_finished_Tasks, always_local_finished_Tasks,
                             ddqn_finished_Tasks]
    total_tasks_matrix = [least_queues_total_Tasks, random_total_Tasks, always_local_total_Tasks, ddqn_total_Tasks]

    percentage_dropped_tasks_matrix = [least_queues_percentage_dropped_tasks, random_percentage_dropped_tasks,
                                       always_local_percentage_dropped_tasks, ddqn_percentage_dropped_tasks]
    percentage_finished_tasks_matrix = [least_queues_percentage_finished_tasks, random_percentage_finished_tasks,
                                        always_local_percentage_finished_tasks, ddqn_percentage_finished_tasks]

    # Convert the data to numpy arrays
    plot_lines(x_values=x, y_values=occupancy_matrix, y_labels=y_labels, plot_title='Occupancy',
               x_axis_label='Episodes', y_axis_label='Occupancy', convert_to_float=True)
    plot_lines(x_values=x, y_values=response_time_matrix, y_labels=y_labels, plot_title='Response Time',
               x_axis_label='Episodes', y_axis_label='Response Time', convert_to_float=True)
    plot_lines(x_values=x, y_values=overloaded_matrix, y_labels=y_labels, plot_title='Overloaded',
               x_axis_label='Episodes', y_axis_label='Overloaded', convert_to_float=True)
    plot_lines(x_values=x, y_values=dropped_tasks_matrix, y_labels=y_labels, plot_title='Dropped Tasks',
               x_axis_label='Episodes', y_axis_label='Dropped Tasks', convert_to_float=True)
    plot_lines(x_values=x, y_values=finished_tasks_matrix, y_labels=y_labels, plot_title='Finished Tasks',
               x_axis_label='Episodes', y_axis_label='Finished Tasks', convert_to_float=True)
    plot_lines(x_values=x, y_values=total_tasks_matrix, y_labels=y_labels, plot_title='Total Tasks',
               x_axis_label='Episodes', y_axis_label='Total Tasks', convert_to_float=True)

    plot_lines(x_values=x, y_values=percentage_dropped_tasks_matrix, y_labels=y_labels,
               plot_title='Percentage Dropped Tasks',
               x_axis_label='Episodes', y_axis_label='Percentage Dropped Tasks', convert_to_float=True)
    plot_lines(x_values=x, y_values=percentage_finished_tasks_matrix, y_labels=y_labels,
               plot_title='Percentage Finished Tasks',
               x_axis_label='Episodes', y_axis_label='Percentage Finished Tasks', convert_to_float=True)


def plot_pe_10_episodes(
        least_queues='./OutputData/least_queue_ether_result_metrics',
        random='./OutputData/random_ether_result_metrics',
        always_local='./OutputData/always_local_ether_result_metrics',
        ddqn='./OutputData/DDQN_result_ether_metrics'
):
    # Read the data from the CSV file
    least_queues_result = read_cvs(least_queues)
    random_result = read_cvs(random)
    always_local_result = read_cvs(always_local)
    ddqn_result = read_cvs(ddqn)

    # Extract the headers and the data
    headers = least_queues_result[0]
    least_queues_data = least_queues_result[1:]
    random_data = random_result[1:]
    always_local_data = always_local_result[1:]
    ddqn_data = ddqn_result[1:]

    least_queues_occupancy, least_queues_overloaded, least_queues_response_time, least_queues_dropped, least_queues_finished, least_queues_total = process_for_each_10(
        least_queues_data)
    random_occupancy, random_overloaded, random_response_time, random_dropped, random_finished, random_total = process_for_each_10(
        random_data)
    always_local_occupancy, always_local_overloaded, always_local_response_time, always_local_dropped, always_local_finished, always_local_total = process_for_each_10(
        always_local_data)
    ddqn_occupancy, ddqn_overloaded, ddqn_response_time, ddqn_dropped, ddqn_finished, ddqn_total = process_for_each_10(
        ddqn_data)

    x = range(len(least_queues_occupancy[0]))
    # Add all the dataon the same plot to a matrix
    y_labels = ['Least Queues', 'Random', 'Always Local', 'DDQN']
    occupancy_matrix = [least_queues_occupancy, random_occupancy, always_local_occupancy, ddqn_occupancy]
    response_time_matrix = [least_queues_response_time, random_response_time, always_local_response_time,
                            ddqn_response_time]
    overloaded_matrix = [least_queues_overloaded, random_overloaded, always_local_overloaded, ddqn_overloaded]

    dropped_tasks_matrix = [always_local_dropped, random_dropped, always_local_dropped,
                            ddqn_dropped]
    finished_tasks_matrix = [least_queues_finished, random_finished, always_local_finished,
                             ddqn_finished]
    total_tasks_matrix = [least_queues_total, random_total, always_local_total, ddqn_total]

    # Convert the data to numpy arrays
    # plot_lines(x_values=x, y_values=occupancy_matrix, y_labels=y_labels, plot_title='Occupancy',
    #            x_axis_label='Episodes', y_axis_label='Occupancy', convert_to_float=True)
    plot_lines_fill_between(x_values=x, y_values=occupancy_matrix, y_labels=y_labels, plot_title='Occupancy',
                            x_axis_label='Episodes', y_axis_label='Occupancy', convert_to_float=True)
    plot_lines_fill_between(x_values=x, y_values=response_time_matrix, y_labels=y_labels, plot_title='Response Time',
                            x_axis_label='Episodes', y_axis_label='Response Time', convert_to_float=True)
    plot_lines_fill_between(x_values=x, y_values=overloaded_matrix, y_labels=y_labels, plot_title='Overloaded',
                            x_axis_label='Episodes', y_axis_label='Overloaded', convert_to_float=True)
    plot_lines(x_values=x, y_values=dropped_tasks_matrix, y_labels=y_labels, plot_title='Dropped Tasks',
               x_axis_label='Episodes', y_axis_label='Dropped Tasks', convert_to_float=True)
    plot_lines(x_values=x, y_values=finished_tasks_matrix, y_labels=y_labels, plot_title='Finished Tasks',
               x_axis_label='Episodes', y_axis_label='Finished Tasks', convert_to_float=True)
    plot_lines(x_values=x, y_values=total_tasks_matrix, y_labels=y_labels, plot_title='Total Tasks',
               x_axis_label='Episodes', y_axis_label='Total Tasks', convert_to_float=True)


def process_for_each_10(array):
    # Separate the data and pre-process it

    overloaded_l = []
    occupancy_l = []
    response_time_l = []
    dropped_l = []
    finished_l = []
    total_l = []

    j = 0
    acc_overloaded = []
    acc_occupancy = []
    acc_response_time = []
    acc_dropped = []
    acc_finished = []
    acc_total = []
    for i in range(len(array)):
        next = j % 10
        overloaded = ast.literal_eval(process_string_array_entry(array[i][0]))
        occupancy = ast.literal_eval(process_string_array_entry(array[i][1]))
        response_time = ast.literal_eval(process_string_array_entry(array[i][2]))
        dropped = ast.literal_eval(process_string_array_entry(array[i][3]))
        finished = ast.literal_eval(process_string_array_entry(array[i][4]))
        total = ast.literal_eval(process_string_array_entry(array[i][5]))

        acc_overloaded.append(np.array([np.mean(overloaded), np.std(overloaded)]))
        acc_occupancy.append(np.array([np.mean(occupancy), np.std(occupancy)]))
        acc_response_time.append(np.array([np.mean(response_time), np.std(response_time)]))
        acc_dropped.append(np.sum(dropped))
        acc_finished.append(np.sum(finished))
        acc_total.append(np.sum(total))
        j += 1
        if next != 0:
            j = 0
            acc_overloaded = []
            acc_occupancy = []
            acc_response_time = []
            acc_dropped = []
            acc_finished = []
            acc_total = []
        else:
            i += 1
            overloaded_l.append(np.mean(acc_overloaded, axis=0))
            occupancy_l.append(np.mean(acc_occupancy, axis=0))
            response_time_l.append(np.mean(acc_response_time, axis=0))
            dropped_l.append(np.mean(acc_dropped))
            finished_l.append(np.mean(acc_finished))
            total_l.append(np.mean(acc_total))
    return overloaded_l, occupancy_l, response_time_l, dropped_l, finished_l, total_l


def get_average_episode_data(array):
    overloaded_l = []
    occupancy_l = []
    response_time_l = []
    dropped_l = []
    finished_l = []
    total_l = []

    acc_overloaded = []
    acc_occupancy = []
    acc_response_time = []
    acc_dropped = []
    acc_finished = []
    acc_total = []
    for i in range(len(array)):
        overloaded = ast.literal_eval(process_string_array_entry(array[i][0]))
        occupancy = ast.literal_eval(process_string_array_entry(array[i][1]))
        response_time = ast.literal_eval(process_string_array_entry(array[i][2]))
        dropped = ast.literal_eval(process_string_array_entry(array[i][3]))
        finished = ast.literal_eval(process_string_array_entry(array[i][4]))
        total = ast.literal_eval(process_string_array_entry(array[i][5]))

        acc_overloaded.append(np.array([np.mean(overloaded), np.std(overloaded)]))
        acc_occupancy.append(np.array([np.mean(occupancy), np.std(occupancy)]))
        acc_response_time.append(np.array([np.mean(response_time), np.std(response_time)]))
        acc_dropped.append(np.sum(dropped))
        acc_finished.append(np.sum(finished))
        acc_total.append(np.sum(total))

        overloaded_l.append(np.mean(acc_overloaded, axis=0))
        occupancy_l.append(np.mean(acc_occupancy, axis=0))
        response_time_l.append(np.mean(acc_response_time, axis=0))
        dropped_l.append(np.mean(acc_dropped))
        finished_l.append(np.mean(acc_finished))
        total_l.append(np.mean(acc_total))

    return overloaded_l, occupancy_l, response_time_l, dropped_l, finished_l, total_l


def plot_average_state_space_exploration(least_queues_base, random_base, always_local_base, ddqn_base, prefix_format,
                                         x_values, x_label="", plot_dots=False, prefix_plot_png=""):
    """
    Make chart of the evolution of the average test across all the episodes. Therefore, gathering the average of the averages
    might be interesting. Aka a plot with an entry with 4 lines (DDQN, LQ, AL, RN) with axis x lambda, axis y whatever
    data is being measured.
    :param least_queues_base:
    :param random_base:
    :param always_local_base:
    :param ddqn_base:
    :param prefix_format:
    :param x_values:
    :return:
    """
    least_queues_avg_for_x = {
        'occupancy': [],
        'overloaded': [],
        'response_time': [],
        'dropped': [],
        'finished': [],
        'total': []
    }
    random_avg_for_x = {
        'occupancy': [],
        'overloaded': [],
        'response_time': [],
        'dropped': [],
        'finished': [],
        'total': []
    }

    always_local_avg_for_x = {
        'occupancy': [],
        'overloaded': [],
        'response_time': [],
        'dropped': [],
        'finished': [],
        'total': []
    }

    ddqn_avg_for_x = {
        'occupancy': [],
        'overloaded': [],
        'response_time': [],
        'dropped': [],
        'finished': [],
        'total': []

    }

    for x_value in x_values:
        least_queues_result = read_cvs(least_queues_base + prefix_format % str(x_value))
        random_result = read_cvs(random_base + prefix_format %  str(x_value))
        always_local_result = read_cvs(always_local_base + prefix_format %  str(x_value))
        ddqn_result = read_cvs(ddqn_base + prefix_format %  str(x_value))

        # Extract the headers and the data
        headers = least_queues_result[0]
        least_queues_data = least_queues_result[1:]
        random_data = random_result[1:]
        always_local_data = always_local_result[1:]
        ddqn_data = ddqn_result[1:]

        least_queues_overloaded,  least_queues_occupancy, least_queues_response_time, least_queues_dropped, least_queues_finished, least_queues_total = get_average_episode_data(
            least_queues_data)
        # add the average of each to the dictionary:
        least_queues_avg_for_x['occupancy'].append(np.mean(least_queues_occupancy))
        least_queues_avg_for_x['overloaded'].append(np.mean(least_queues_overloaded))
        least_queues_avg_for_x['response_time'].append(np.mean(least_queues_response_time))
        least_queues_avg_for_x['dropped'].append(np.mean(least_queues_dropped))
        least_queues_avg_for_x['finished'].append(np.mean(least_queues_finished))
        least_queues_avg_for_x['total'].append(np.mean(least_queues_total))

        random_overloaded, random_occupancy, random_response_time, random_dropped, random_finished, random_total = get_average_episode_data(
            random_data)
        # add the average of each to the dictionary:
        random_avg_for_x['occupancy'].append(np.mean(random_occupancy))
        random_avg_for_x['overloaded'].append(np.mean(random_overloaded))
        random_avg_for_x['response_time'].append(np.mean(random_response_time))
        random_avg_for_x['dropped'].append(np.mean(random_dropped))
        random_avg_for_x['finished'].append(np.mean(random_finished))
        random_avg_for_x['total'].append(np.mean(random_total))

        always_local_overloaded, always_local_occupancy, always_local_response_time, always_local_dropped, always_local_finished, always_local_total = get_average_episode_data(
            always_local_data)
        # add the average of each to the dictionary:
        always_local_avg_for_x['occupancy'].append(np.mean(always_local_occupancy))
        always_local_avg_for_x['overloaded'].append(np.mean(always_local_overloaded))
        always_local_avg_for_x['response_time'].append(np.mean(always_local_response_time))
        always_local_avg_for_x['dropped'].append(np.mean(always_local_dropped))
        always_local_avg_for_x['finished'].append(np.mean(always_local_finished))
        always_local_avg_for_x['total'].append(np.mean(always_local_total))

        ddqn_overloaded, ddqn_occupancy, ddqn_response_time, ddqn_dropped, ddqn_finished, ddqn_total = get_average_episode_data(
            ddqn_data)
        # add the average of each to the dictionary:
        ddqn_avg_for_x['occupancy'].append(np.mean(ddqn_occupancy))
        ddqn_avg_for_x['overloaded'].append(np.mean(ddqn_overloaded))
        ddqn_avg_for_x['response_time'].append(np.mean(ddqn_response_time))
        ddqn_avg_for_x['dropped'].append(np.mean(ddqn_dropped))
        ddqn_avg_for_x['finished'].append(np.mean(ddqn_finished))
        ddqn_avg_for_x['total'].append(np.mean(ddqn_total))
    # plot the averages
    plot_lines(x_values,
               [least_queues_avg_for_x['occupancy'], random_avg_for_x['occupancy'], always_local_avg_for_x['occupancy'],
                ddqn_avg_for_x['occupancy']], ['Least Queues', 'Random', 'Always Local', 'DDQN'], 'Occupancy', x_label,
               'Occupancy', convert_to_float=True, plot_dots=plot_dots, prefix_plot_png=prefix_plot_png)
    plot_lines(x_values, [least_queues_avg_for_x['overloaded'], random_avg_for_x['overloaded'],
                          always_local_avg_for_x['overloaded'], ddqn_avg_for_x['overloaded']],
               ['Least Queues', 'Random', 'Always Local', 'DDQN'], 'Overloaded',  x_label,'Overloaded',
               convert_to_float=True, plot_dots=plot_dots, prefix_plot_png=prefix_plot_png)
    plot_lines(x_values, [least_queues_avg_for_x['response_time'], random_avg_for_x['response_time'],
                          always_local_avg_for_x['response_time'], ddqn_avg_for_x['response_time']],
               ['Least Queues', 'Random', 'Always Local', 'DDQN'], 'Response Time',  x_label, 'Response Time',
               convert_to_float=True, plot_dots=plot_dots, prefix_plot_png=prefix_plot_png)
    plot_lines(x_values,
               [least_queues_avg_for_x['dropped'], random_avg_for_x['dropped'], always_local_avg_for_x['dropped'],
                ddqn_avg_for_x['dropped']], ['Least Queues', 'Random', 'Always Local', 'DDQN'], 'Dropped',  x_label,
               'Dropped', convert_to_float=True, plot_dots=plot_dots, prefix_plot_png=prefix_plot_png)
    plot_lines(x_values,
               [least_queues_avg_for_x['finished'], random_avg_for_x['finished'], always_local_avg_for_x['finished'],
                ddqn_avg_for_x['finished']], ['Least Queues', 'Random', 'Always Local', 'DDQN'], 'Finished',  x_label,
               'Finished', convert_to_float=True, plot_dots=plot_dots, prefix_plot_png=prefix_plot_png)
    plot_lines(x_values, [least_queues_avg_for_x['total'], random_avg_for_x['total'], always_local_avg_for_x['total'],
                          ddqn_avg_for_x['total']], ['Least Queues', 'Random', 'Always Local', 'DDQN'], 'Total',
               x_label, 'Total', convert_to_float=True, plot_dots=plot_dots, prefix_plot_png=prefix_plot_png)


def plot_per_episode(
        least_queues='./OutputData/least_queue_ether_result_metrics',
        random='./OutputData/random_ether_result_metrics',
        always_local='./OutputData/always_local_ether_result_metrics',
        ddqn='./OutputData/DDQN_result_ether_metrics'
):
    # Read the data from the CSV file
    least_queues_result = read_cvs(least_queues)
    random_result = read_cvs(random)
    always_local_result = read_cvs(always_local)
    ddqn_result = read_cvs(ddqn)

    # Extract the headers and the data
    headers = least_queues_result[0]
    least_queues_data = least_queues_result[1:]
    random_data = random_result[1:]
    always_local_data = always_local_result[1:]
    ddqn_data = ddqn_result[1:]

    least_queues_occupancy, least_queues_overloaded, least_queues_response_time, least_queues_dropped, least_queues_finished, least_queues_total = get_average_episode_data(
        least_queues_data)
    random_occupancy, random_overloaded, random_response_time, random_dropped, random_finished, random_total = get_average_episode_data(
        random_data)
    always_local_occupancy, always_local_overloaded, always_local_response_time, always_local_dropped, always_local_finished, always_local_total = get_average_episode_data(
        always_local_data)
    ddqn_occupancy, ddqn_overloaded, ddqn_response_time, ddqn_dropped, ddqn_finished, ddqn_total = get_average_episode_data(
        ddqn_data)

    x = range(len(least_queues_occupancy[0]))
    # Add all the dataon the same plot to a matrix
    y_labels = ['Least Queues', 'Random', 'Always Local', 'DDQN']
    occupancy_matrix = [least_queues_occupancy, random_occupancy, always_local_occupancy, ddqn_occupancy]
    response_time_matrix = [least_queues_response_time, random_response_time, always_local_response_time,
                            ddqn_response_time]
    overloaded_matrix = [least_queues_overloaded, random_overloaded, always_local_overloaded, ddqn_overloaded]

    dropped_tasks_matrix = [always_local_dropped, random_dropped, always_local_dropped,
                            ddqn_dropped]
    finished_tasks_matrix = [least_queues_finished, random_finished, always_local_finished,
                             ddqn_finished]
    total_tasks_matrix = [least_queues_total, random_total, always_local_total, ddqn_total]


if __name__ == '__main__':
    # plot_all_complete_percentage_in_dir('./OutputData/StateSpaceExploration/', [2, 4, 8, 16],
    #                                     [0.0005, 0.005, 0.01, 0.05, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plot_results(least_queues='./OutputData/least_queue_ether_result_metrics',
    #              random='./OutputData/random_ether_result_metrics',
    #              always_local='./OutputData/always_local_ether_result_metrics',
    #              ddqn='./OutputData/DDQN_result_ether_metrics')
    # plot_per_episode(least_queues='./OutputData/least_queue_ether_result_metrics', random='./OutputData/random_ether_result_metrics', always_local='./OutputData/always_local_ether_result_metrics', ddqn='./OutputData/DDQN_result_ether_metrics')
    # plot_rewards(ddqn='./OutputData/DDQN_result_ether_train_rewards')
    # plot_pe_10_episodes(least_queues='./OutputData/least_queue_ether_result_metrics', random='./OutputData/random_ether_result_metrics', always_local='./OutputData/always_local_ether_result_metrics', ddqn='./OutputData/DDQN_result_ether_metrics')
    plot_average_state_space_exploration(least_queues_base='./OutputData/LambdaExploration/least_queue_ether',
                                            random_base='./OutputData/LambdaExploration/random_ether',
                                            always_local_base='./OutputData/LambdaExploration/always_local_ether',
                                            ddqn_base='./OutputData/LambdaExploration/DDQN_result_ether',
                                            prefix_format='_%s_result_metrics',
                                            x_values=[ 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                            plot_dots=True, prefix_plot_png="_le_2")
    # plot_average_state_space_exploration(least_queues_base='./OutputData/clusters/least_queue_ether_no_clusters',
    #                                         random_base='./OutputData/clusters/random_ether_no_clusters',
    #                                         always_local_base='./OutputData/clusters/always_local_ether_no_clusters',
    #                                         ddqn_base='./OutputData/clusters/DDQN_result_ether_no_clusters',
    #                                         prefix_format='_%s_lambda_0.5_result_metrics',
    #                                         x_values=[1, 2, 3, 4],
    #                                         plot_dots=True, prefix_plot_png="_cluster") # always_local_ether_no_clusters_1_lambda_0.5_result_metrics