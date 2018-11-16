import torch
import numpy as np
import submission
import random

class NeuralNet(torch.nn.Module):
    def __init__(self, input, h1, output):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(NeuralNet, self).__init__()
        self.linear1 = torch.nn.Linear(input, h1)
        self.linear2 = torch.nn.Linear(h1, output)

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1 = torch.nn.functional.relu((self.linear1(x)))
        output = self.linear2(h1)
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

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, input, h1,  output = 36, 8, 3, 36

train, dev, test = submission.read_data(.5,.25,.25)
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
model = NeuralNet(input, h1, output)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# torch.nn.CrossEntropyLoss() uses Softmax
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

for t in range(15):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # otherPrediction = values, indices = tensor.max(0)
    #want accuracy on training set = accuracy on dev set.
    #Print argmax prediction for test set.
    # values = torch.argmax(y_pred, dim=1)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
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


#test = model(torch.tensor(xdev))
