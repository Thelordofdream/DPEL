import torch
from torchvision import datasets, transforms
import torchvision
import numpy as np
from torch.autograd import Variable
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world-size', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=9000)
parser.add_argument('--backend', type=str, default='nccl')
args = parser.parse_args()
print('----Torch Config----')
print('world-size : {}'.format(args.world_size))
print('rank : {}'.format(args.rank))
print('local_rank : {}'.format(args.local_rank))
print('mini batch-size : {}'.format(args.batch_size))
print('backend : {}'.format(args.backend))
print('--------------------')

torch.distributed.init_process_group(backend='nccl', init_method="env://")
torch.cuda.set_device(args.local_rank)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

data_train = datasets.MNIST(root="./",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./",
                           transform=transform,
                           train=False)

data_train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                pin_memory=True,
                                                batch_size=args.batch_size,
                                                sampler=data_train_sampler)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               pin_memory=True,
                                               batch_size=2000,
                                               shuffle=False)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2),
                                         torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(3 * 3 * 128, 64),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(64, 10),
                                         torch.nn.Softmax(1))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 3 * 3 * 128)
        x = self.dense(x)
        return x


model = Model()
model = torch.nn.parallel.DistributedDataParallel(model.to(args.local_rank), device_ids=[args.local_rank])
cost = torch.nn.CrossEntropyLoss().to(args.local_rank)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(100):
    s = time.time()
    running_loss = 0.0
    running_correct = 0
    step = 0
    print("Epoch {}/{}".format(epoch, 100))
    for train_image, train_target in data_loader_train:
        train_image = Variable(train_image).cuda(non_blocking=True)
        train_target = Variable(train_target).cuda(non_blocking=True)
        outputs = model(train_image)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, train_target)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == train_target.data)
        step += 1
    e = time.time()

    testing_correct = 0
    for test_image, test_target in data_loader_test:
        test_image = Variable(test_image).cuda(non_blocking=True)
        test_target = Variable(test_target).cuda(non_blocking=True)
        outputs = model(test_image)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == test_target.data)

    print("Loss is:{:.4f}, Train Accuracy is:{:.2f}%, Test Accuracy is:{:.2f}%, Cost Time is:{:.3f}s".format(running_loss / step,
                                                                                  100 * running_correct / len(data_train) * 2,
                                                                                  100 * testing_correct / len(data_test),
                                                                                  e - s))
torch.save(model.module.state_dict(), "pytorch_distributed_mnist_model.pth")