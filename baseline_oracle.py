
# Implements baseline and oracle predictors for CS221-Final-Project
import csv
import random
from sgd import clean_data
from sgd import read_data

'''
Implements a baseline classifier, which classifies every loan as "B" if performing 7-class
classification, and classifies every loan as "B3" if performing 35-class classification. These
classes were chosen since "B" and "B3" are the most common grades and subgrades in the dataset,
respectively.
'''
def baseline(dataset_tuple, subgrades):
	correct = 0
	total = 0

	test_set = dataset_tuple[2]
	grades = {"A1":1, "A2":2, "A3":3, "A4":4, "A5":5, "B1":6, "B2":7, "B3":8, "B4":9, "B5":10, "C1":11, "C2":12, "C3":13, "C4":14, "C5":15, "D1":16, "D2":17, "D3":18, "D4":19, "D5":20, "E1":21, "E2":22, "E3":23, "E4":24, "E5":25, "F1":26, "F2":27, "F3":28, "F4":29, "F5":30, "G1":31, "G2":32, "G3":33, "G4":34, "G5":35, "":0}

	prediction_subgrade = 0
	prediction_grade = []

	if subgrades:
		prediction_subgrade = grades["B3"]
	else:
		prediction_grade = [grades["B1"], grades["B2"], grades["B3"], grades["B4"], grades["B5"]]

	for i in range(len(test_set[0])):
		prediction = grades["B3"]
		y = test_set[1][i]

		if subgrades:
			if y == prediction_subgrade:
				correct += 1
			total += 1
		else:
			if y in prediction_grade:
				correct += 1
			total += 1
	print("Baseline classification accuracy: " + str(float(correct)/float(total)))

'''
Implements an oracle classifier. Given that our dataset contained both independent and dependent variables, 
and given that we knew of an existing rating algorithm, an oracle could attain 100% accuracy by implementing 
the existing rating algorithm. Since we do not have access to that algorithm, we apply the oracle algorithm 
to our specific problem by reading off the classification from the dataset for a given feature vector.
'''
def oracle(dataset_tuple):
	correct = 0
	total = 0

	test_set = dataset_tuple[2]

	for i in range(len(test_set[0])):
		x = test_set[0][i]
		y = test_set[1][i]
		correct += 1
		total += 1

	print("Classification accuracy: " + str(float(correct)/float(total)))

if __name__ == "__main__":
	dataset = read_data(0.5, 0.25, 0.25)
	baseline(dataset, True)
	baseline(dataset, False)
	oracle(dataset)
