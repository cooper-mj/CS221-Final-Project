# CS221-Final-Project

Collection of dataset and three classifiers - linear classifier, k-means classifier, neural net - which classify Lending Club loans into Lending Club risk grades.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing and grading purposes.

## Files in Repository

* **sgd.py** - code for linear classifier with SGD.
* **kmeans.py** - code to run k-means classifier.
* **neuralnet.py** - code for neural net classifier.
* **LoanStats3a.csv** - Lending Club dataset containing loan profiles and risk grade lables for 42,538 loans approved by Lending Club between 2007 and 2011.
* **smalldatasettest.csv** - Small portion of the above dataset, for early-stage testing purposes.

### Prerequisites

Our classifiers rely on the following Python packages. All are listed in requirements.txt and can be installed via the pip package manager.
* PyTorch
* NumPy
* SKLearn
* Pandas
* MatplotLib

### Installing

Install the above packages via pip, then you should be good to go!

To install the packages from the requirements.txt file, navigate into the project directory, then type:
```
$ pip install -r requirements.txt
```

## Running the classifiers

All of the files can be run as Python scripts from the command line.

## Built With

* [PyTorch](https://pytorch.org) - An open source deep learning platform that provides a seamless path from research prototyping to production deployment.
* [Scikit-Learn](https://scikit-learn.org/stable/) - Open source platform for machine learning in Python.

## Authors

* **Danny Takeuchi**
* **Michael Cooper**
* **Kaushal Alate**

## Acknowledgments

* Will Bakst, our project mentor.
* Percy and the entire course staff for a great quarter!



