from sklearn.cluster import KMeans
from sklearn import model_selection as ms
import pandas as pd
import matplotlib.pyplot as plt

def kmeans():
    ### Load and split data
    df = pd.read_csv("LoanStats3aKM.csv")
    trainDataFull, testDataFull = ms.train_test_split(df, train_size=0.25)
    trainData = trainDataFull.drop(axis=1, labels=["grade"])
    kmeans = KMeans(n_clusters=7, verbose=0, random_state=0).fit(trainData)
    testData = testDataFull.drop(axis=1, labels=["grade"])

    print("\n\n\nResults")
    
    ### Make predictions
    pred = list(kmeans.predict(testData))
   
    ### Assign clusters to credit grades
    freq = {}
    for i in range(7):
        freq[pred.count(i)] = i
    predToGradeMap = {}
    freqVal = freq.keys()
    freqVal.sort(reverse=True)
    i = 0
    for grade in ['B', 'A', 'C', 'D', 'E', 'F', 'G']:
        f = freqVal[i]
        predToGradeMap[freq[f]] = grade
        i += 1

    ### Compute accuracy and create plot
    accuracy = 0
    i = 0
    for actGrade in testDataFull['grade']:
        predGrade = predToGradeMap[pred[i]]
        if actGrade == predGrade:
            accuracy += 1
        i += 1
    accuracy /= float(len(testDataFull['grade']))
    print accuracy
    freqVal[0], freqVal[1] = freqVal[1], freqVal[0]
    plt.bar(['A', 'B', 'C', 'D', 'E', 'F', 'G'], freqVal)
    plt.show()

if __name__ == "__main__":
    main()

def main():
    kmeans()
