'''
This file has a set uf utilities to help interactively draw plots over the results and data.
'''
import csv

import matplotlib.pyplot as plt
import numpy as np
import ast
import re

def read_cvs(filename):
    # Open the CSV file
    with open(f'{filename}', 'r') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Convert the CSV data to a list of lists
        data = list(csv_reader)
    return data

def process_string_array_entry(array):
    return re.sub(',+', ',', re.sub( ' +', ',', array.replace('\n', ' ')).replace('[,','[').replace(',]', ']'))

import matplotlib.pyplot as plt

def plot_three_lines(x_values, y1_values, y2_values, y3_values, y1_label, y2_label, y3_label, plot_title, x_label, y_label, convert_to_float=False):

    # Convert the y values to float arrays
    if convert_to_float:
        y1_values = np.array(y1_values, dtype=float)
        y2_values = np.array(y2_values, dtype=float)
        y3_values = np.array(y3_values, dtype=float)

    # Create a plot
    plt.plot(x_values, y1_values, label=y1_label)
    plt.plot(x_values, y2_values, label=y2_label)
    plt.plot(x_values, y3_values, label=y3_label)
    # Add a title and labels for the axes
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Show a legend to label the lines
    plt.legend()
    # Display the plot
    plt.show()


if __name__ == '__main__':
    # Read the data from the CSV file
    least_qeueus_result= read_cvs('./OutputData/least_queue_result_metrics')
    random_result = read_cvs('./OutputData/random_result_metrics')
    always_local_result = read_cvs('./OutputData/always_local_result_metrics')
    # ddqn_result = read_cvs('./OutputData/ddqn_result_metrics')

    # Extract the headers and the data
    headers = least_qeueus_result[0]
    least_qeueus_data = least_qeueus_result[1:]
    random_data = random_result[1:]
    always_local_data = always_local_result[1:]
    # ddqn_data = ddqn_result[1:]
    
    least_queues_occupancy = []
    random_occupancy = []
    always_local_occupancy = []
    # ddqn_occupancy = []
    
    least_qeueus_response_time = []
    random_response_time = []
    always_local_response_time = []
    # ddqn_response_time = []
    
    least_qeueus_overloaded = []
    random_overloaded = []
    always_local_overloaded = []
    # ddqn_overloaded = []

    
    # Separate the data and pre-process it
    for i in range(len(least_qeueus_data)):
        try:
            # Pre-process the into three arrays, occupancy, overloads and response time for each type of agent
            least_qeueus_overloaded.append(np.mean(ast.literal_eval(process_string_array_entry(least_qeueus_data[i][0]))))
            random_overloaded.append(np.mean(ast.literal_eval(process_string_array_entry(random_data[i][0]))))
            always_local_overloaded.append(np.mean(ast.literal_eval(process_string_array_entry(always_local_data[i][0]))))
            # ddqn_overloaded.append(np.mean(ast.literal_eval(process(ddqn_data[i][0]))))

            least_queues_occupancy.append(np.mean(ast.literal_eval(process_string_array_entry(least_qeueus_data[i][1]))))
            random_occupancy.append(np.mean(ast.literal_eval(process_string_array_entry(random_data[i][1]))))
            always_local_occupancy.append(np.mean(ast.literal_eval(process_string_array_entry(always_local_data[i][1]))))
            # ddqn_occupancy.append(np.mean(ast.literal_eval(process_string_array_entry(ddqn_data[i][1]))))

            least_qeueus_response_time.append(np.mean(ast.literal_eval(process_string_array_entry(least_qeueus_data[i][2]))))
            random_response_time.append(np.mean(ast.literal_eval(process_string_array_entry(random_data[i][2]))))
            always_local_response_time.append(np.mean(ast.literal_eval(process_string_array_entry(always_local_data[i][2]))))
            # ddqn_response_time.append(np.mean(ast.literal_eval(process(ddqn_data[i][2]))))

        except Exception as e:
            print(f"Error on line {i}")
            print(f'Least Queues: {least_qeueus_data[i]}')
            print(f'Random Data : {random_data[i]}')
            print(f'Alwasy local: {always_local_data[i]}')
            # print(f'DDQN: {ddqn_data[i]}')
            input('Failed here, continue?')

        # ddqn_data[i] = list(map(ast.literal_eval, ddqn_data[i]))

    # Generate X axis
    x = range(len(least_qeueus_data))

    # Convert the data to numpy arrays
    plot_three_lines(x, least_queues_occupancy, random_occupancy, always_local_occupancy, 'Least Queues', 'Random', 'Always Local', 'Occupancy', 'Episodes', 'Occupancy', True)
    plot_three_lines(x, least_qeueus_response_time, random_response_time, always_local_response_time, 'Least Queues', 'Random', 'Always Local', 'Response Time', 'Episodes', 'Response Time', True)
    plot_three_lines(x, least_qeueus_overloaded, random_overloaded, always_local_overloaded, 'Least Queues', 'Random', 'Always Local', 'Overloaded', 'Episodes', 'Overloaded', True)
