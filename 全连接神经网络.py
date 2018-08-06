import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 500
num_classes = 10
n_epoches = 5
batch_size = 100
learning_rate = 0.001
train_dataset = torchvision.datasets.MNIST(root='F:\data',
                                           train=True,transform=transforms.ToTensor(),
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='F:\data',
                                          train=False,transform=transforms.ToTensor(),
                                          )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)


class NerualNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NerualNet()
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

total_step = len(train_loader)

for epoch in range(n_epoches):
    for i ,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, n_epoches, i + 1, total_step, loss.item()))
            print('------------')
            print('Loss:',loss)


with torch.no_grad(): # 不需要求导
    correct = 0
    total = 0
    for images,labels in test_loader:
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        output = model(images)
        print('test output shape',output.data.size())
        _,predicted = torch.max(output.data,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images:{}%'.format(100*correct/total))
