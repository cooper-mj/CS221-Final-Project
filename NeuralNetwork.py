import torch
import numpy as np
import submission
from sklearn import preprocessing
import torch.nn.functional as F
import csv
import random
import re

random.seed(100)  # For consistency during development

'''
Takes in three floating point numbers representing the proportion of the dataset to be
devoted to the training set, evaluation set, and test set, respectively. Returns a 3-tuple
consisting of cleaned training set, evaluation set, and test set.

The format of each dataset is (X, Y), where X is a list of feature vectors, and Y is a 
list of buckets for multiclass classification.
'''


def read_data(frac_training_set, frac_evaluation_set, frac_test_set, classify7 = False):
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
        ret_list = list(reader)[2:]  # This [2:] removes text header and individual column headers

    # Partition data
    random.shuffle(ret_list)

    print("Cleaning data . . .")
    ret_list = clean_data(ret_list, classify7)
    print("Data cleaned!")

    abs_training_set = int(len(ret_list[0]) * frac_training_set)
    abs_evaluation_set = int(len(ret_list[0]) * frac_evaluation_set)
    abs_test_set = int(len(ret_list[0]) * frac_test_set)

    training_set = (ret_list[0][0:abs_training_set], ret_list[1][0:abs_training_set])
    evaluation_set = (ret_list[0][abs_training_set + 1:abs_training_set + 1 + abs_evaluation_set],
                      ret_list[1][abs_training_set + 1:abs_training_set + 1 + abs_evaluation_set])
    test_set = (ret_list[0][abs_training_set + 1 + abs_evaluation_set + 1:],
                ret_list[1][abs_training_set + 1 + abs_evaluation_set + 1:])
    return (training_set, evaluation_set, test_set)


'''
Takes in a list of lists representing the dataset - returns a 2-tuple
of the form (X, Y), where X is a list of feature vectors, and Y is a list of corresponding
classes for multiclass classification.
'''


def clean_data(data_list, classify7):
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
    home_ownership_status = {"": -1, "RENT": 0, "MORTGAGE": 1, "OWN": 2, "OTHER": 3, "NONE": 4}
    purpose = {"": -1, "credit_card": 0, "car": 1, "small_business": 2, "other": 3, "wedding": 4,
               "debt_consolidation": 5, "home_improvement": 6, "major_purchase": 7, "medical": 8, "moving": 9,
               "vacation": 10, "house": 11, "renewable_energy": 12, "educational": 13}
    if classify7:
        grade = {"A1": 1, "A2": 1, "A3": 1, "A4": 1, "A5": 1, "B1": 2, "B2": 2, "B3": 2, "B4": 2, "B5": 2, "C1": 3,
                 "C2": 3, "C3": 3, "C4": 3, "C5": 3, "D1": 4, "D2": 4, "D3": 4, "D4": 4, "D5": 4, "E1": 5,
                 "E2": 5, "E3": 5, "E4": 5, "E5": 5, "F1": 6, "F2": 6, "F3": 6, "F4": 6, "F5": 6, "G1": 7,
                 "G2": 7, "G3": 7, "G4": 7, "G5": 7, "": 0}
    else:
        grade = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "B1": 6, "B2": 7, "B3": 8, "B4": 9, "B5": 10, "C1": 11,
            "C2": 12, "C3": 13, "C4": 14, "C5": 15, "D1": 16, "D2": 17, "D3": 18, "D4": 19, "D5": 20, "E1": 21,
            "E2": 22, "E3": 23, "E4": 24, "E5": 25, "F1": 26, "F2": 27, "F3": 28, "F4": 29, "F5": 30, "G1": 31,
            "G2": 32, "G3": 33, "G4": 34, "G5": 35, "": 0}

    verification = {"Verified": 0, "Source Verified": 1, "Not Verified": 2, "": -1}
    paid = {"Fully Paid": 0, "Charged Off": 1, "Does not meet the credit policy. Status:Fully Paid":2, "Does not meet the credit policy. Status:Charged Off":3, "": -1}
    def remove_invalid_values(st):
        if len(str(st)) > 0:
            return float(st)
        return -1

    for data_point in data_list:
        # Convert data into numeric
        # updated_data_point_X = [data_point[2], data_point[5][1:-7], re.sub("[^0-9]", "", data_point[11]),
        #                         home_ownership_status[data_point[12]], data_point[13], purpose[data_point[20]],
        #                         data_point[24], data_point[25]]
        updated_data_point_X = [data_point[2], data_point[3], data_point[4], data_point[5][1:-7],  data_point[7], re.sub("[^0-9]", "", data_point[11]),
                                home_ownership_status[data_point[12]], data_point[13], verification[data_point[14]], paid[data_point[16]], purpose[data_point[20]],
                                data_point[24], data_point[25]]
        valid_updated_data_point_X = []
        for i, elem in enumerate(updated_data_point_X):
            valid_updated_data_point_X.append(remove_invalid_values(elem))

        updated_data_point_X = valid_updated_data_point_X[:]
        updated_data_point_X = [float(i) for i in updated_data_point_X]

        updated_data_point_Y = float(grade[data_point[9]])
        X.append(updated_data_point_X)
        Y.append(updated_data_point_Y)

    scaler = preprocessing.MinMaxScaler((0,100))
    scaler.fit(X)
    newX = scaler.transform(X).tolist()
    return newX, Y

class NeuralNet(torch.nn.Module):
    def __init__(self, input, h1, h2, output):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(NeuralNet, self).__init__()
        self.linear1 = torch.nn.Linear(input, h1)
        self.linear2 = torch.nn.Linear(h1, h2)
        self.linear3 = torch.nn.Linear(h2, output)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        output = self.linear3(h2)
        return output

def getAccuracy(prediction, result):
    match = 0
    total = 0
    for i in range(len(prediction)):
        if prediction[i] == result[i]:
            match+=1
            total+=1
        else:
            total+=1
    return 1.0 * match / total

accuracyData7 = []
lossData7 = []
accuracyData35 = []
lossData35 = []
def test1(data, input, h1, h2, output, lr):
    # D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.

    train, dev, test = data
    xtrain = train[0]
    ytrain = train[1]
    xdev = dev[0]
    ydev = dev[1]
    xtest = test[0]
    ytest = test[1]

    # Create random Tensors to hold inputs and outputs
    x = torch.tensor(xtrain)
    y = torch.LongTensor(ytrain)


    # Construct our model by instantiating the class defined above
    model = NeuralNet(input, h1, h2, output)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    # torch.nn.CrossEntropyLoss() uses Softmax
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iter = 400
    for t in range(iter):
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        values = torch.argmax(y_pred, dim=1)
        # print(values.numpy())
        #print(getAccuracy(values.numpy(), ytrain))
        if len(accuracyData7) != iter:
            accuracyData7.append(getAccuracy(values.numpy(), ytrain))
        else:
            accuracyData35.append(getAccuracy(values.numpy(), ytrain))
        # Compute and print loss
        loss = criterion(y_pred, y)
        #print(t, loss.item())
        if len(lossData7) != iter:
            lossData7.append(loss.item())
        else:
            lossData35.append(loss.item())
        # Zero gradients, perform a backward pass, and update the weights.
        loss.backward()
        optimizer.step()

    trainingModel = model(torch.tensor(x))
    values = torch.argmax(trainingModel, dim = 1)
    print("Training Accuracy:")
    print(getAccuracy(values.numpy(), y))
    devModel = model(torch.tensor(xdev))
    values = torch.argmax(devModel, dim = 1)
    print("Dev Accuracy:")
    print(getAccuracy(values.numpy(), ydev))

    testModel = model(torch.tensor(xtest))
    values = torch.argmax(testModel, dim = 1)
    print("Test Accuracy:")
    print(getAccuracy(values.numpy(), ytest))



test1(read_data(.8,.1,.1, True),13,10,10,8,0.008)
test1(read_data(.8,.1,.1, False),13,20,20,36, 0.01)

import matplotlib.pyplot as plt

iteration = [i for i in range(len(accuracyData7))]
plt.plot(iteration, accuracyData7, 'r', iteration, accuracyData35, 'b')
plt.ylabel('% Accuracy')
plt.xlabel('Iteration')
plt.show()
plt.plot(iteration, lossData7, 'r', iteration, lossData35, 'b')
plt.ylabel('Training Loss')
plt.xlabel('Iteration')
plt.show()