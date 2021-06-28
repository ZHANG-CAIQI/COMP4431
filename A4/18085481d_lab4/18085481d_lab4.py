import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets


from sklearn.metrics import accuracy_score


np.random.seed(0)
torch.manual_seed(0)

# read in Iris Data

iris = datasets.load_iris()

X = iris.data
Y = iris.target

# Split into test and training data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=73)

# Transfer to torch tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=12)
        self.output = nn.Linear(in_features=12, out_features=3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x


model = ANN()
# define criterion and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


epochs = 100
loss_arr = []
train_loss = []
train_accuracy = []

for i in range(epochs):
    Y_hat = model(torch.FloatTensor(X_train))
    loss = criterion(Y_hat, Y_train)
    accuracy = accuracy_score(Y_train, np.argmax(Y_hat.detach().numpy(), axis=1))
    train_accuracy.append(accuracy)
    train_loss.append(loss.item())

    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


Y_hat_test = model(torch.FloatTensor(X_test))

test_accuracy = accuracy_score(Y_test, np.argmax(Y_hat_test.detach().numpy(), axis=1))

print("Test Accuracy {:.2f}".format(test_accuracy))


# plot the training loss and training accuracy
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(train_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss')

ax[1].plot(train_accuracy)
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')

plt.tight_layout()
plt.show()



