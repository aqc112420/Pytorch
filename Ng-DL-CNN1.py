import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
from tf_utils import *

X_train_orig,Y_train_orig,X_test_orig,Y_test_orig ,classes= load_dataset()

# index = 15
# plt.imshow(X_train_orig[index])
# plt.show()
# print('y=',np.squeeze(Y_train_orig[:,index]))

X_train = (X_train_orig/255).astype(np.float32)
X_test = (X_test_orig/255).astype(np.float32)
Y_train = Y_train_orig.squeeze()
Y_test = Y_test_orig.squeeze()
# print(Y_train.shape)
# Y_train = convert_to_one_hot(Y_train_orig,6).T
# Y_test = convert_to_one_hot(Y_test_orig,6).T
print(Y_train.shape)
# print ("number of training examples = " + str(X_train.shape[0]))
# print ("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

n_iterations = 100
minibatch_size = 64
lr = 0.001
costs = []


class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.out = nn.Linear(8*8*16,6)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        # print(x.data.size())
        output = self.out(x)

        return output

cnn = CNN()
# # print(cnn)
#
optimizer = torch.optim.Adam(cnn.parameters(),lr)

loss_function = nn.CrossEntropyLoss()
def model(X_train,Y_train,minibatch_size=64,lr=0.001,n_iterations=150):

    seed = 3
    m = X_train.shape[0]
    for epoch in range(n_iterations):
        minibatch_cost = 0
        num_minibatches = int(m/minibatch_size)
        seed += 1
        minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)

        for minibatch in minibatches:
            (minibatch_X,minibatch_Y) = minibatch
            minibatch_X = np.array([i.T for i in minibatch_X])
            minibatch_Y = np.array([i.T for i in minibatch_Y])
            minibatch_X = (torch.from_numpy(minibatch_X))
            minibatch_Y = torch.from_numpy(minibatch_Y).long()
            minibatch_X = Variable(minibatch_X)
            minibatch_Y = Variable(minibatch_Y).long()
            # print(minibatch_X.data.size())
            output = cnn(minibatch_X)
            # print(output.data.size())
            # print(minibatch_Y.data.size())
            loss = loss_function(output,minibatch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            minibatch_cost += loss

        if  epoch % 5 == 0:
            print('Cost after epoch{}:{}'.format(epoch, minibatch_cost
                                                 ))
        if  epoch % 1 == 0:
            costs.append(minibatch_cost)

    plt.plot(np.squeeze(costs))
    plt.xlabel('iterations(per tens)')
    plt.ylabel('cost')
    plt.title('Learning rate=' + str(lr))
    plt.show()
    torch.save(cnn,'F:/cnn.pth')

# model(X_train,Y_train,minibatch_size=64,lr=0.001,n_iterations=150)

def test(X_test,Y_test):
    cnn.eval()
    test_loss = 0
    correct = 0
    data = np.array([i.T for i in X_test])
    target = np.array([i.T for i in Y_test])
    data = torch.from_numpy(data)
    target = torch.from_numpy(target).long()
    data, target = Variable(data),Variable(target).long()

    output = cnn(data)
    test_loss = loss_function(output,target).data[0]
    pred = output.data.max(1,keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(data)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                 len(data),
                                                                                 100. * correct / len(
                                                                                     data)))

if __name__ == '__main__':
    model(X_train, Y_train, minibatch_size=64, lr=0.001, n_iterations=150)
    test(X_test,Y_test)
