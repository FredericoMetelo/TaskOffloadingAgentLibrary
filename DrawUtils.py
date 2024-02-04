'''
This file has a set uf utilities to help interactively draw plots over the results and data.
'''
import csv

import matplotlib.pyplot as plt
import numpy as np


def read_cvs(filename):
    # Open the CSV file
    with open(f'{filename}', 'r') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Convert the CSV data to a list of lists
        data = list(csv_reader)
    return data

if __name__ == '__main__':
    # Read the data from the CSV file
    least_qeueus_result= read_cvs('./OutputData/least_queue_result')
    random_result = read_cvs('./OutputData/random_result')
    always_local_result = read_cvs('./OutputData/always_local_result')

    # Extract the headers and the data
    headers = least_qeueus_result[0]
    least_qeueus_data = least_qeueus_result[1:]
    random_data = random_result[1:]
    always_local_data = always_local_result[1:]

    # Generate X axis
    x = range(len(least_qeueus_data))

    # Convert the data to numpy arrays
    least_qeueus_data = np.array(least_qeueus_data, dtype=np.float)
    random_data = np.array(random_data, dtype=np.float)
    always_local_data = np.array(always_local_data, dtype=np.float)

    # Plot the data
    # Plot Least Queues overloaded
    least_qeueus_overloaded = least_qeueus_data[:, 1]
    plt.plot(x, least_qeueus_data[:, 0], label="Least Queues")