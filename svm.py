from sklearn import svm
from sklearn import metrics
import submission

train, dev, test = submission.read_data(.8,.1,.1)
xtrain = train[0]
ytrain = train[1]
xdev = dev[0]
ydev = dev[1]
xtest = test[0]
ytest = test[1]

#Create a svm Classifier
clf = svm.SVC(kernel='rbf')

#Train the model using the training sets
clf.fit(xtrain, ytrain)

#Predict the response for test dataset
y_pred_train = clf.predict(xtrain)
y_pred_test = clf.predict(xtest)

print("Train Accuracy:", metrics.accuracy_score(ytrain, y_pred_train))
print("Test Accuracy:", metrics.accuracy_score(ytest, y_pred_test))