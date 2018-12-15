
# Implements baseline and oracle predictors for CS221-Final-Project
import csv
import random
from sgd import clean_data
from sgd import read_data

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
	oracle(dataset)
