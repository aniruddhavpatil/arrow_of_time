import torch
import torchvision
import torchvision.transforms as transforms
from net import *
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pickle

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=False, transform=transform)

print trainset.shape
# trainset = 

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

ip = torch.randn()

classes = ()

net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        print('[%d] loss: %.3f' % (epoch + 1, loss.data[0]))
print('Finished Training')

correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = Variable(images.cuda()),Variable(labels.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted,labels = predicted.cpu().numpy(),labels.data.cpu()
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

filehandler = open("net.pth","wb")

pickle.dump(net, filehandler)

filehandler.close()
