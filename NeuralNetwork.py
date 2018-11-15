import torch
import numpy as np
import random

# n_in, n_h, n_out, batch_size = 10, 5, 1, 10
#
# x = torch.randn(batch_size, n_in)
# y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
#
# model = nn.Sequential(nn.Linear(n_in, n_h),
#                      nn.ReLU(),
#                      nn.Linear(n_h, n_out),
#                      nn.Sigmoid())
#
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# for epoch in range(50):
#     # Forward Propagation
#     y_pred = model(x)
#     # Compute and print loss
#     loss = criterion(y_pred, y)
#     print('epoch: ', epoch, ' loss: ', loss.item())
#     # Zero the gradients
#     optimizer.zero_grad()
#
#     # perform a backward pass (backpropagation)
#     loss.backward()
#
#     # Update the parameters
#     optimizer.step()






class TwoLayerNet(torch.nn.Module):
    def __init__(self, input, h1, h2, output):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(input, h1)
        self.linear2 = torch.nn.Linear(h1, h2)
        self.linear3 = torch.nn.Linear(h2, output)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1 = self.relu(self.linear1(x))
        h2 = self.relu(self.linear2(h1))
        output = self.softmax(self.linear3(h2))
        return torch.argmax(output, dim=1)


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, input, h1, h2, output = 35, 10, 20, 20, 35

train, xdev, xtest
ytrain, ydev, ytest

# Create random Tensors to hold inputs and outputs
# x = torch.tensor( [][] )
# y = torch.tensor([])
x = [[random.randint(1,10) for i in range(0,35)] for j in range(N)]
y = [1 for i in range(35)]

# Construct our model by instantiating the class defined above
model = TwoLayerNet(10, 20, 20, 35)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # # Compute and print loss
    # print(y_pred)
    # print(y)
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.

f(train)
f(dev)
f(test)
