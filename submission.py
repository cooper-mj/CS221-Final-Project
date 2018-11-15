###############################################################################
#                                                                             #
#                  CS221 Final Project Software Submission                    #
#                                                                             #
# ----------------------------------------------------------------------------#
#                                                                             #
#                Danny Takeuchi, Kaushal Alate, Michael Cooper                #
#                                                                             #
###############################################################################

from sklearn import linear_model
import csv
import random
import numpy as np
import re


random.seed(100) # For consistency during development

'''
Takes in three floating point numbers representing the proportion of the dataset to be
devoted to the training set, evaluation set, and test set, respectively. Returns a 3-tuple
consisting of cleaned training set, evaluation set, and test set.

The format of each dataset is (X, Y), where X is a list of feature vectors, and Y is a 
list of buckets for multiclass classification.
'''
def read_data(frac_training_set, frac_evaluation_set, frac_test_set):

    print("Reading data . . .")
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

    print("Cleaning data . . .")
    ret_list = clean_data(ret_list)
    print("Data cleaned!")

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
    #   index: 23 - Borrower's Address State (REMOVED TEMPORARILY)
    #   index: 24 - Debt to Income Ratio
    #   index: 25 - History of delinquency (binary value indicating borrower delinquency on a loan in the past two years).
    #   
    # Dependent variable
    #   index: 9 - Sub Grade (ranges from A1 to G5)
    X = []
    Y = []
    home_ownership_status = {"":-1, "RENT":0, "MORTGAGE":1, "OWN":2, "OTHER":3, "NONE":4}
    purpose = {"":-1, "credit_card":0, "car":1, "small_business":2, "other":3, "wedding":4, "debt_consolidation":5, "home_improvement":6, "major_purchase":7, "medical":8, "moving":9, "vacation":10, "house":11, "renewable_energy":12, "educational":13}
    grade = {"A1":1, "A2":2, "A3":3, "A4":4, "A5":5, "B1":6, "B2":7, "B3":8, "B4":9, "B5":10, "C1":11, "C2":12, "C3":13, "C4":14, "C5":15, "D1":16, "D2":17, "D3":18, "D4":19, "D5":20, "E1":21, "E2":22, "E3":23, "E4":24, "E5":25, "F1":26, "F2":27, "F3":28, "F4":29, "F5":30, "G1":31, "G2":32, "G3":33, "G4":34, "G5":35, "":-1}
    
    def remove_invalid_values(st):
        if len(str(st))>0:
            return float(st)
        return -1


    for data_point in data_list:
        # Convert data into numeric
        updated_data_point_X = [data_point[2], data_point[5][1:-7], re.sub("[^0-9]", "", data_point[11]), home_ownership_status[data_point[12]], data_point[13], purpose[data_point[20]], data_point[24], data_point[25]]
        
        valid_updated_data_point_X = []
        for i, elem in enumerate(updated_data_point_X):
            valid_updated_data_point_X.append(remove_invalid_values(elem))

        updated_data_point_X = valid_updated_data_point_X[:]
        updated_data_point_X = [float(i) for i in updated_data_point_X]

        updated_data_point_Y = float(grade[data_point[9]])
        X.append(updated_data_point_X)
        Y.append(updated_data_point_Y)
    return (X, Y)

'''
Takes in a prediction and comparison, and checks whether or not they are in the same
general (A-G) grade.
'''
def test_categories(prediction, comparison, grades):
    if len(grades[prediction]) == 0 or len(grades[comparison]) == 0:
        return False
    return grades[prediction][0] == grades[comparison][0] 

def stochastic_gradient_descent(dataset_tuple):
    # Train
    print("Training . . .")
    training_set = dataset_tuple[0]
    evaluation_set = dataset_tuple[1]
    test_set = dataset_tuple[2]

    X = np.array(training_set[0])
    Y = np.array(training_set[1])
    clf = linear_model.SGDClassifier(max_iter=1000)
    clf.fit(X, Y)

    # Test - exact values
    print("Testing exact values . . .")
    correct = 0
    total = 0
    for i, X in enumerate(evaluation_set[0]):
        prediction = clf.predict(np.array([X]))[0]
        if prediction == evaluation_set[1][i]:
            correct += 1
        total += 1
    print("Exact values testing accuracy: " + str(round(float(correct)/float(total), 2)))


    # Test - approximate values
    print("Testing approximate (categorized) values . . .")
    correct = 0
    total = 0
    for i, X in enumerate(evaluation_set[0]):
        prediction = clf.predict(np.array([X]))[0]

        grades = {"A1":1, "A2":2, "A3":3, "A4":4, "A5":5, "B1":6, "B2":7, "B3":8, "B4":9, "B5":10, "C1":11, "C2":12, "C3":13, "C4":14, "C5":15, "D1":16, "D2":17, "D3":18, "D4":19, "D5":20, "E1":21, "E2":22, "E3":23, "E4":24, "E5":25, "F1":26, "F2":27, "F3":28, "F4":29, "F5":30, "G1":31, "G2":32, "G3":33, "G4":34, "G5":35, "":-1}
        inverse_grades = ivd = {v: k for k, v in grades.items()}

        if test_categories(prediction, evaluation_set[1][i], inverse_grades): # 5 is an arbitrary threshold value
            correct += 1
        total += 1
    print("Approximate values testing accuracy: " + str(round(float(correct)/float(total), 2)))





'''
Indicates we are running submission.py as a script.
'''
if __name__ == "__main__":
    dataset = read_data(0.5, 0.25, 0.25)
    stochastic_gradient_descent(dataset)
