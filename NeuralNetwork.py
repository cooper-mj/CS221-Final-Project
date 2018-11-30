import torch
import numpy as np
import submission
import torch.nn.functional as F

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

        self.softmax = torch.nn.Softmax()

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

def test1():
    # D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, input, h1, h2,  output = 36, 8, 10, 10, 36

    train, dev, test = submission.read_data(.8,.1,.1)
    xtrain = train[0]
    ytrain = train[1]
    xdev = dev[0]
    ydev = dev[1]
    xtest = test[0]
    ytest = test[1]

    # Create random Tensors to hold inputs and outputs
    x = torch.tensor(xtrain)
    y = torch.LongTensor(ytrain)

    # x = torch.tensor(xtrain[0:5])
    # y = torch.LongTensor(ytrain[0:5])

    # Construct our model by instantiating the class defined above
    model = NeuralNet(input, h1, h2, output)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    # torch.nn.CrossEntropyLoss() uses Softmax
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for t in range(100):
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        values = torch.argmax(y_pred, dim=1)
        print(getAccuracy(values.numpy(), ytrain))
        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        loss.backward()
        optimizer.step()


    trainingModel = model(torch.tensor(xtrain))
    values = torch.argmax(trainingModel, dim = 1)
    print("Training Accuracy:")
    print(getAccuracy(values.numpy(), ytrain))


    devModel = model(torch.tensor(xdev))
    values = torch.argmax(devModel, dim = 1)
    print("Dev Accuracy:")
    print(getAccuracy(values.numpy(), ydev))

    testModel = model(torch.tensor(xtest))
    values = torch.argmax(testModel, dim = 1)
    print("Test Accuracy:")
    print(getAccuracy(values.numpy(), ytest))

    # y_devpred = model(xdev)
    # devloss = criterion(y_devpred, ydev)
    # print(t, loss.item(), devloss.item())
    # want accuracy on training set = accuracy on dev set.
    # Print argmax prediction for test set.
    # values = torch.argmax(y_pred, dim=1)

    #test = model(torch.tensor(xdev))

test1()