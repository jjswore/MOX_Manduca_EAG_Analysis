import matplotlib.pyplot as plt
import numpy as np
from utils.EAG_DataProcessing_Library import Read_CSV_With_Col_Names
from config import TEST_CSV
def find_sol(data):
    """
    This function takes an array of data, calculates the difference between consecutive
    elements, and then finds the first indices of series where the difference is either
    less than -0.6 or greater than 0.6.

    Arguments:
    data -- a list or array-like object of numerical values

    Returns:
    sol -- a list of tuples, where the first element of each tuple is an index where
           the difference is greater than 0.6, and the second element is an index where
           the difference is less than -0.6.
    """
    # Calculate the difference between consecutive elements in the array
    SolFall = np.diff(data)

    # Initialize the lists to store the indices and the flags
    NI, PI = [], []
    flag_ni, flag_pi = False, False

    # Iterate through the SolFall array starting from the 6th element
    for i, v in enumerate(SolFall[5:]):
        # If the difference is less than -0.6 and the last data point did not meet this condition
        if v < -0.6 and not flag_ni:
            # Add the current index to the NI list
            NI.append(i)
            # Set the flag to True to indicate that the current data point meets the condition
            flag_ni = True
        elif v >= -0.6:
            # If the difference is not less than -0.6, reset the flag to False
            flag_ni = False

        # If the difference is more than 0.6 and the last data point did not meet this condition
        if v > 0.6 and not flag_pi:
            # Add the current index to the PI list
            PI.append(i)
            # Set the flag to True to indicate that the current data point meets the condition
            flag_pi = True
        elif v <= 0.6:
            # If the difference is not more than 0.6, reset the flag to False
            flag_pi = False

    # Create a list of tuples where each tuple contains an element from PI and an element from NI
    test = zip(PI, NI)
    # Convert the zip object to a list
    sol = list(test)

    # Return the list of tuples
    return sol
