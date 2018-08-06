import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]

)
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,transform=transform,download=False)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,transform=transform)

testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','trunk')


device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def imshow(img):
    img = img /2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# dataiter = iter(trainloader)
# images,labels = dataiter.next()
# print(images.data.size())
# print(labels.data.size())
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s'%classes[labels[j]] for j in range(4)))
# print(labels[j] for j in range(4))
#

class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out

net = Net()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i,data in enumerate(trainloader):
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        output = net(images)
        loss = criterion(output,labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print('Epoch:{}|Step:{}|loss:{:.3}'.format(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0

print('Finished Training')


# dataiter = iter(testloader)
# images,labels = dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
#
# print(' '.join('%5s'%classes[labels[j]] for j in range(4)))

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images,labels = data
#         output = net(images)
#         # print(output.data.size()) [4,10]
#         _,predicted = torch.max(output.data,1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _,predicted = torch.max(outputs,1)
        c = (predicted == labels).squeeze()
        print(c)
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



