###############################################################################
# 																			  #
#                  CS221 Final Project Software Submission					  #
# 																			  #
# ----------------------------------------------------------------------------#
# 																			  #
#                Danny Takeuchi, Kaushal Alate, Michael Cooper				  #
# 																			  #
###############################################################################

from sklearn.linear_model import SGDClassifier
import csv
import random



'''
Algorithms to Implement:
	- Stochastic Gradient Descent
	- K-nearest neighbour
	- SGD via neural network
	- Naive bayes classification
'''

'''
Reads in three pieces of data from the dataset - (1) Training set, (2) Evaluation
set, (3) Test set. Returns a 3-tuple consisting of the data in that order.
'''
def read_data(frac_training_set, frac_evaluation_set, frac_test_set):
	# Load the data
	with open('LoanStats3a.csv', 'r') as file:
	  reader = csv.reader(file)
	  ret_list = list(reader)[2:] # This [2:] removes text header and individual column headers

	# Partition data
	random.shuffle(ret_list)
	return ret_list

'''
Takes in a list of lists representing the dataset - returns a 2-tuple
of the form (feature_vector, y).
'''
def clean_data(data_list):
	# Features for extraction
	# 	index: 2 - Loan Amount Requested
	# 	index: 5[:1] - Term of Loan
	# 	index: 11 - Employment Length
	# 	index: 12 - Home Ownership Status
	# 	index: 13 - Annual Income
	# 	Purpose of Loan (discrete set of options, including credit_card, car, small_business, debt_consolidation, other)
	# 	index: 23 - Borrower’s Address State
	# 	index: 24 - Debt to Income Ratio
	# 	index: 25 - History of delinquency (binary value indicating borrower delinquency on a loan in the past two years).











read_data()
