from sklearn import linear_model
import csv
import random
import numpy as np
import re
import sys

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
    print("\nData cleaned!")

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
    #   index: 11 - Employment Length (regex it to take only the numerical values)
    #   index: 12 - Home Ownership Status
    #   index: 13 - Annual Income
    #   index: 20 - Purpose of Loan (discrete set of options, including credit_card, car, small_business, debt_consolidation, other)
    #   index: 23 - Borrower's Address State (REMOVED TEMPORARILY)
    #   index: 24 - Debt to Income Ratio
    #   index: 25 - History of delinquency (number of times borrower has been delinquent on a loan in the past two years).
    #   
    # Dependent variable
    #   index: 9 - Sub Grade (ranges from A1 to G5)
    X = []
    Y = []
    home_ownership_status = {"":-1, "NONE":-1, "RENT":0, "MORTGAGE":1, "OWN":2, "OTHER":3}
    purpose = {"":-1, "credit_card":0, "car":1, "small_business":2, "wedding":3, "debt_consolidation":4, "home_improvement":5, "major_purchase":6, "medical":7, "moving":8, "vacation":9, "house":10, "renewable_energy":11, "educational":12, "other":13}
    grade = {"A1":1, "A2":2, "A3":3, "A4":4, "A5":5, "B1":6, "B2":7, "B3":8, "B4":9, "B5":10, "C1":11, "C2":12, "C3":13, "C4":14, "C5":15, "D1":16, "D2":17, "D3":18, "D4":19, "D5":20, "E1":21, "E2":22, "E3":23, "E4":24, "E5":25, "F1":26, "F2":27, "F3":28, "F4":29, "F5":30, "G1":31, "G2":32, "G3":33, "G4":34, "G5":35, "":0}
    
    def remove_invalid_values(st):
        if len(str(st))>0:
            return float(st)
        return -1

    def one_hot_ownership_status(cell_text):
        # index -1 - "", or "NONE" receive no mapping (if block below against this).
        # index 0 - "RENT" maps to this one.
        # index 1 - "MORTGAGE" maps to this one.
        # index 2 - "OWN" maps to this one.
        # index 3 - "OTHER" maps to this one.
        ownership_vec = [0, 0, 0, 0]
        if home_ownership_status[cell_text] != -1:
            ownership_vec[home_ownership_status[cell_text]] = 1
        return ownership_vec

    def one_hot_purpose(cell_text):
        # index -1 = "" receives no mapping (if block below checks against this).
        # index 0 - credit card
        # index 1 - car
        # index 2 - small business
        # index 3 - wedding
        # index 4 - debt_consolidation
        # index 5 - home_improvement
        # index 6 - major purpose
        # index 7 - medical
        # index 8 - moving
        # index 9 - vacation
        # index 10 - house
        # index 11 - renewable energy
        # index 12 - educational
        # index 13 - other
        purpose_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if purpose[cell_text] != -1:
            purpose_vec[purpose[cell_text]] = 1
        return purpose_vec


    # Read in the data, clean each point, adds the cleaned data point to the X vector, and the
    # point's class to the Y vector.
    for i, data_point in enumerate(data_list):
        
        sys.stdout.write("\r%d%%" % int(100*float(i+1)/float(len(data_list))))
        sys.stdout.flush()

        # Convert data into numeric
        updated_data_point_X = [data_point[2], data_point[5][1:-7], re.sub("[^0-9]", "", data_point[11])] + one_hot_ownership_status(data_point[12]) +  [data_point[13]] + one_hot_purpose(data_point[20]) + [data_point[24], data_point[25]]
        
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
Takes in a list of lists representing the dataset - returns a 2-tuple
of the form (X, Y), where X is a list of feature vectors, and Y is a list of corresponding
classes for multiclass classification.

Unlike the above clean_data function, this one produces a list of feature vectors where each
feature vector contains few features.
'''
def clean_data_min_features(data_list):
    # Features for extraction
    #   index: 2 - Loan Amount Requested
    #   index: 13 - Annual Income
    #     
    # Dependent variable
    #   index: 9 - Sub Grade (ranges from A1 to G5)
    X = []
    Y = []
    grade = {"A1":1, "A2":2, "A3":3, "A4":4, "A5":5, "B1":6, "B2":7, "B3":8, "B4":9, "B5":10, "C1":11, "C2":12, "C3":13, "C4":14, "C5":15, "D1":16, "D2":17, "D3":18, "D4":19, "D5":20, "E1":21, "E2":22, "E3":23, "E4":24, "E5":25, "F1":26, "F2":27, "F3":28, "F4":29, "F5":30, "G1":31, "G2":32, "G3":33, "G4":34, "G5":35, "":0}
    
    def remove_invalid_values(st):
        if len(str(st))>0:
            return float(st)
        return -1

    for i, data_point in enumerate(data_list):
        
        sys.stdout.write("\r%d%%" % int(100*float(i+1)/float(len(data_list))))
        sys.stdout.flush()

        # Convert data into numeric
        updated_data_point_X = [data_point[2], data_point[13]]
        
        valid_updated_data_point_X = []
        for i, elem in enumerate(updated_data_point_X):
            valid_updated_data_point_X.append(remove_invalid_values(elem))

        updated_data_point_X = valid_updated_data_point_X[:]
        updated_data_point_X = [float(i) for i in updated_data_point_X]

        updated_data_point_Y = float(grade[data_point[9]])
        X.append(updated_data_point_X)
        Y.append(updated_data_point_Y)
    print(X[0])
    return (X, Y)


'''
Runs stochastic gradient descent on the dataset passed into the function. Trains
on the training set, evaluates accuracy on the evaluation set. Does not return anything,
but prints accuracy metrics to the console.

Evaluates the classification accuracy using two metrics:
    (1) Sub-grade classification accuracy (e.g. E5).
    (2) Broad-grade classification accuracy (e.g. B).

'''
def stochastic_gradient_descent(dataset_tuple, maxIters):
    # Train
    print("Training . . .")
    training_set = dataset_tuple[0]
    evaluation_set = dataset_tuple[1]
    test_set = dataset_tuple[2]

    X = np.array(training_set[0])
    Y = np.array(training_set[1])
    clf = linear_model.SGDClassifier(max_iter=maxIters, loss="log")
    clf.fit(X, Y)

    # Test - exact values (A1 through G5)
    print("\nTesting exact values . . .")
    exact_correct = 0
    exact_total = 0
    for i, X in enumerate(evaluation_set[0]):
        prediction = clf.predict(np.array([X]))[0]
        if prediction == evaluation_set[1][i]:
            exact_correct += 1
        exact_total += 1
    print("Exact values testing accuracy: " + str(round(float(exact_correct)/float(exact_total), 2)))
    
    '''
    Takes in a prediction and comparison, and checks whether or not they are in the same
    general (A-G) grade.
    '''
    def test_categories(prediction, comparison, grades):
        return grades[prediction][0] == grades[comparison][0]


    # Test - approximate values (A through G)
    print("\nTesting approximate (categorized) values . . .")

    grade_counter = {"A":0, "B":0, "C":0, "D":0, "E":0, "F":0, "G":0}
    grade_counter_correct = {"A":0, "B":0, "C":0, "D":0, "E":0, "F":0, "G":0}
    categorized_as = {"A":0, "B":0, "C":0, "D":0, "E":0, "F":0, "G":0}

    approximate_correct = 0
    approximate_total = 0
    for i, X in enumerate(evaluation_set[0]):

        prediction = clf.predict(np.array([X]))[0]
        grades = {"A1":1, "A2":2, "A3":3, "A4":4, "A5":5, "B1":6, "B2":7, "B3":8, "B4":9, "B5":10, "C1":11, "C2":12, "C3":13, "C4":14, "C5":15, "D1":16, "D2":17, "D3":18, "D4":19, "D5":20, "E1":21, "E2":22, "E3":23, "E4":24, "E5":25, "F1":26, "F2":27, "F3":28, "F4":29, "F5":30, "G1":31, "G2":32, "G3":33, "G4":34, "G5":35, "":0}
        inverse_grades = ivd = {v: k for k, v in grades.items()}

        if len(inverse_grades[prediction]) == 0 or len(inverse_grades[evaluation_set[1][i]]) == 0:
            # Pass if we have an invalid (empty) value
            continue

        if test_categories(prediction, evaluation_set[1][i], inverse_grades):
            # If we get it right, increment our correct counter
            approximate_correct += 1
            grade_counter_correct[inverse_grades[evaluation_set[1][i]][0]] += 1

        # Regardless of whether we got it right or wrong, increment our denominator counter
        approximate_total += 1
        grade_counter[inverse_grades[evaluation_set[1][i]][0]] += 1
        categorized_as[inverse_grades[prediction][0]] += 1

    print("Approximate values testing accuracy: " + str(round(float(approximate_correct)/float(approximate_total), 2)))

    print("")
    for key in grade_counter:
        if grade_counter[key] > 0:
            print("Accuracy for category " + str(key) + " : " + str(grade_counter_correct[key]/float(grade_counter[key])))
    
    print("")
    for data_point in sorted(categorized_as.keys()):
        print("Number of data points categorized as " + str(data_point) + " : " + str(categorized_as[data_point]))


'''
Indicates we are running submission.py as a script.
'''
if __name__ == "__main__":
    # Setup our datasets with 0.5 of the dataset going to the training set,
    # 0.25 going to the evaluation set, and 0.25 going to the test set.
    dataset = read_data(0.5, 0.25, 0.25)
    stochastic_gradient_descent(dataset, 1000)

