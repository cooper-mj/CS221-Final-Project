###############################################################################
#                                                                             #
#                  CS221 Final Project Software Submission                    #
#                                                                             #
# ----------------------------------------------------------------------------#
#                                                                             #
#                Danny Takeuchi, Kaushal Alate, Michael Cooper                #
#                                                                             #
###############################################################################

from sklearn.linear_model import SGDClassifier
import csv
import random

'''
Takes in three floating point numbers representing the proportion of the dataset to be
devoted to the training set, evaluation set, and test set, respectively. Returns a 3-tuple
consisting of cleaned training set, evaluation set, and test set.

The format of each dataset is (X, Y), where X is a list of feature vectors, and Y is a 
list of buckets for multiclass classification.
'''
def read_data(frac_training_set, frac_evaluation_set, frac_test_set):

    # Error checking - can't have fractions not add up to 1.
    summation = frac_training_set + frac_evaluation_set + frac_test_set
    if summation != 1:
        # If they don't sum to 1, then we normalize
        frac_training_set = frac_training_set / summation
        frac_evaluation_set = frac_evaluation_set / summation
        frac_test_set = frac_test_set / summation

    # Load the data
    with open('LoanStats3a.csv', 'r') as file:
      reader = csv.reader(file)
      ret_list = list(reader)[2:] # This [2:] removes text header and individual column headers

    # Partition data
    random.shuffle(ret_list)
    ret_list = clean_data(ret_list)
    abs_training_set = int(len(ret_list[0]) * frac_training_set)
    abs_evaluation_set = int(len(ret_list[0]) * frac_evaluation_set)
    abs_test_set = int(len(ret_list[0]) * frac_test_set)

    training_set = (ret_list[0][0:abs_training_set], ret_list[1][0:abs_training_set])
    evaluation_set = (ret_list[0][abs_training_set+1:abs_training_set+1+abs_evaluation_set], ret_list[1][abs_training_set+1:abs_training_set+1+abs_evaluation_set])
    test_set = (ret_list[0][abs_training_set+1+abs_evaluation_set+1:], ret_list[1][abs_training_set+1+abs_evaluation_set+1:])
    return (training_set, evaluation_set, test_set)

'''
Takes in a list of lists representing the dataset - returns a 2-tuple
of the form (X, Y), where X is a list of feature vectors, and Y is a list of corresponding
classes for multiclass classification.
'''
def clean_data(data_list):
    # Features for extraction
    #   index: 2 - Loan Amount Requested
    #   index: 5 - Term of Loan (take [1:-7] to format the string to get the integer term of the loan in years)
    #   index: 11 - Employment Length (take [:-6] to format the string to get the integer term of employment in years)
    #   index: 12 - Home Ownership Status
    #   index: 13 - Annual Income
    #   index: 20 - Purpose of Loan (discrete set of options, including credit_card, car, small_business, debt_consolidation, other)
    #   index: 23 - Borrower’s Address State
    #   index: 24 - Debt to Income Ratio
    #   index: 25 - History of delinquency (binary value indicating borrower delinquency on a loan in the past two years).
    #   
    # Dependent variable
    #   index: 9 - Sub Grade (ranges from A1 to G5)
    X = []
    Y = []
    for data_point in data_list:
        updated_data_point_X = [data_point[2], data_point[5][1:-7], data_point[11][:-6], data_point[12], data_point[13], data_point[20], data_point[23], data_point[24], data_point[25]]
        updated_data_point_Y = data_point[9]
        X.append(updated_data_point_X)
        Y.append(updated_data_point_Y)
    return (X, Y)

'''
Indicates we are running submission.py as a script.
'''
if __name__ == "__main__":
    read_data(0.5, 0.25, 0.25)
