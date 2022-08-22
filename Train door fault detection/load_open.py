from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from pm_loss import CenterLoss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*3*3, 50)
        self.ip2 = nn.Linear(50, 2)
        self.ip3 = nn.Linear(2, 4)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 128*3*3)
        x = F.relu(self.ip1(x))
        x = F.dropout(x, training=self.training)
        x = self.ip2(x)
        out = self.ip3(x)
        return x, F.log_softmax(out, dim=1)

def visualize(feat, labels, epoch, acc, unseen, mis):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#0000ff', '#000000']
    plt.clf()
    for i in range(5):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['normal', 'little', 'medium', 'serious', 'unseen'], loc = 'upper right')
    plt.xlim(xmin=-5.0, xmax=5.0)
    plt.ylim(ymin=-5.0, ymax=5.0)
    #plt.text(-4.5, 4.5,"The number of unseen samples=%d" % unseen)
    #plt.text(-4.5, 4.0, "The number of misclassifications=%d" % mis)

    plt.draw()
    plt.pause(0.001)

def dist(args, model, device, train_loader, fea_loader, tar_loader, centers):
    model.eval()

    d0 = []
    d1 = []
    d2 = []
    d3 = []

    mean_d0 = 0
    mean_d1 = 0
    mean_d2 = 0
    mean_d3 = 0

    var_d0 = 0
    var_d1 = 0
    var_d2 = 0
    var_d3 = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            sample = data[:, 0:784]
            target = data[:, 784:785]
            sample = sample.reshape((-1, 1, 28, 28))
            sample = Variable(torch.from_numpy(sample))
            sample = sample.cuda()
            sample = sample.type(torch.cuda.FloatTensor)

            target = Variable(torch.from_numpy(target))
            target = target.view(64)
            target = target.cuda()
            target = target.type(torch.cuda.LongTensor)

            fea, output = model(sample) #[64,2]
            fea_loader.append(fea)
            tar_loader.append(target)

            expanded_centers = centers.expand(64, -1, -1)  # [64,4,2]
            expanded_feature = fea.expand(4, -1, -1).transpose(1, 0)  # [64,4,2]
            distance_centers = (expanded_feature - expanded_centers).pow(2).sum(dim=-1)  # [64,4]

            for i in range(0, 64):
                if target[i]==0:
                    d0.append(distance_centers[i,0]**0.5)
                if target[i]==1:
                    d1.append(distance_centers[i,1]**0.5)
                if target[i]==2:
                    d2.append(distance_centers[i,2]**0.5)
                if target[i]==3:
                    d3.append(distance_centers[i,3]**0.5)
    #print(len(d0))
    
    mean_d0 = np.mean(d0)
    mean_d1 = np.mean(d1)
    mean_d2 = np.mean(d2)
    mean_d3 = np.mean(d3)
 
    var_d0 = np.std(d0)
    var_d1 = np.std(d1)
    var_d2 = np.std(d2)
    var_d3 = np.std(d3)

    return mean_d0,mean_d1,mean_d2,mean_d3,var_d0,var_d1,var_d2,var_d3

def test(args, model, device, test_loader, fea_loader, tar_loader, centers,m0,m1,m2,m3,v0,v1,v2,v3):
    model.eval()
    test_loss = 0
    correct_cnt = 0
    total_cnt = 0
    unseen_cnt = 0
    mis_cnt = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            sample = data[:, 0:784]
            target = data[:, 784:785]
            sample = sample.reshape((-1, 1, 28, 28))
            sample = Variable(torch.from_numpy(sample))
            sample = sample.cuda()
            sample = sample.type(torch.cuda.FloatTensor)

            target = Variable(torch.from_numpy(target))
            target = target.view(64)
            target = target.cuda()
            target = target.type(torch.cuda.LongTensor)

            fea, output = model(sample) #[64,2]
            fea_loader.append(fea)
            tar_loader.append(target)
            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            out_label = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            expanded_centers = centers.expand(64, -1, -1)  # [64,4,2]
            expanded_feature = fea.expand(4, -1, -1).transpose(1, 0)  # [64,4,2]
            distance_centers = (expanded_feature - expanded_centers).pow(2).sum(dim=-1)  # [64,4]

            for i in range(0,64):
                if distance_centers[i,0]**0.5 > (m0+2*v0):
                    if distance_centers[i, 1]**0.5 > (m1+2*v1):
                        if distance_centers[i, 2]**0.5 > (m2+2*v2):
                            if distance_centers[i, 3]**0.5 > (m3+2*v3):
                                # print(distance_centers[i])
                                # print(out_label[i])
                                # print(target[i])
                                cnt += 1
                                if target[i] == 4:
                                    unseen_cnt += 1
                                else:
                                    mis_cnt += 1
                                # print(unseen_cnt, mis_cnt, cnt)

            correct_cnt += out_label.eq(target.view_as(out_label)).sum().item()
            total_cnt += sample.data.size()[0]
            test_loss /= batch_idx+1
            if batch_idx == 398:
                test_acc = 100.0 * correct_cnt / (total_cnt-200*64)
                print('\nAccuracy: {}/{} ({:.2f}%, batch: {})\n'.format(
                correct_cnt, total_cnt-200*64,
                test_acc, batch_idx+1))

    feat = torch.cat(fea_loader, 0)
    labels = torch.cat(tar_loader, 0)
    return feat, labels, test_acc, unseen_cnt, mis_cnt

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #data_train = np.load('open_train.npy')
    all_open_test_2=r'C:\Users\alpka\OneDrive\Masaüstü\train_door_pressure_classification-masterr\source_code\all_open_test_2.npy'
    data_test = np.load(all_open_test_2)
    #print(data_test.shape)
    saved_model_open= r'C:\Users\alpka\OneDrive\Masaüstü\train_door_pressure_classification-masterr\source_code\saved_model_open.pt' 
    model = torch.load(saved_model_open).to(device)
    centers_open= r'C:\Users\alpka\OneDrive\Masaüstü\train_door_pressure_classification-masterr\source_code\centers_open.txt' 
    centers = torch.load(centers_open).to(device)
    open_train= r'C:\Users\alpka\OneDrive\Masaüstü\train_door_pressure_classification-masterr\source_code\open_train.npy'
    data_train = np.load(open_train)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    sheduler = lr_scheduler.StepLR(optimizer, 30, gamma=0.5)

    epoch = 1
    sheduler.step()
    fea_loader = []
    tar_loader = []
    #train_acc = train(args, model, device, data_train, optimizer, epoch, fea_loader, tar_loader)
    m0,m1,m2,m3,v0, v1, v2, v3 = dist(args, model, device, data_train, fea_loader, tar_loader, centers)
    feat, labels, test_acc, unseen, mis = test(args, model, device, data_test, fea_loader, tar_loader, centers,m0,m1,m2,m3,v0,v1,v2,v3)
    
    #visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch, test_acc, unseen, mis)
    
    print(m0,m1,m2,v3,v0,v1,v2,v3)
if __name__ == '__main__':
    main()
