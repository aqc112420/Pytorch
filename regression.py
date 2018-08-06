import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F

x = torch.unsqueeze(torch.linspace(-1, 1, 100), 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        torch.nn.Module.__init__(self)
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
criterion = torch.nn.MSELoss()

plt.ion()

for t in range(200):
    prediction = net(x)
    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()