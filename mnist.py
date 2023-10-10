import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('C:/Users/Asus/Downloads/nishant python/mnist.csv', delimiter=',')
X = dataset[:1000,0:784]
y = dataset[:1000,784]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#print(y_train)
s = np.zeros([1,10])
Y_train = np.zeros([len(y_train),10])
Y_test = np.zeros([len(y_test),10])
j = 0
for i in y_train:
    x = int(i)
    s[:,x] = 1
    Y_train[j] = s
    s = np.zeros([1,10])
    j = j+1
j = 0    
for i in y_test:
    x = int(i)
    s[:,x] = 1
    Y_test[j] = s
    s = np.zeros([1,10])
    j = j+1
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
# define the model
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 1000)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(1000, 1500)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(1500,1000)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(1000,700)
        self.act4 = nn.ReLU()
        self.output = nn.Linear(700, 10)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act_output(self.output(x))
        return x

model = PimaClassifier()
#print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = Y_train[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# compute accuracy
y_pred = model(X_test)
Y_pred = y_pred.round()
pre_label = [];
for j in range(len(y_test)):
    x = 0
    for i in Y_pred[j]:
        if i == 1:
            pre_label.append(x)
        x = x+1
    
    

print(Y_pred , ' and ',Y_test[:,:]  )    
accuracy = (Y_pred == Y_test).float().mean()
print(f"Accuracy {accuracy}")
print(pre_label ,' and ', y_test)
