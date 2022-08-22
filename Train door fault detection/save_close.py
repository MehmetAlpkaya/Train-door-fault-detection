from __future__ import print_function
import argparse
from collections import defaultdict
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
# Evrişimli Sinir Ağları modelinin oluşturulması:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) # Giriş kanalı: 1, Çıkış kanalı: 32, Filtre boyutu: 5x5
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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    history = defaultdict(list)

    loss_weight = 0.5
    centerloss = CenterLoss(4, 2).to(device) #feat dim Başarı Hem nn.Module hem de ScriptModule tarafından paylaşılan dahili Modül durumunu başlatır.

    correct_cnt = 0
    total_cnt = 0
    # optimzer4center
    optimzer4center = optim.SGD(centerloss.parameters(), lr=0.5)

    for batch_idx, data in enumerate(train_loader):
        sample = data[:, 0:784]
        target = data[:, 784:785]
        sample = sample.reshape((-1,1,28,28))
        sample = Variable(torch.from_numpy(sample))
        sample = sample.cuda()
        sample = sample.type(torch.cuda.FloatTensor)
        target = Variable(torch.from_numpy(target))
        target = target.view(64)
        target = target.cuda()
        target = target.type(torch.cuda.LongTensor)
        optimizer.zero_grad()
        optimzer4center.zero_grad()
        fea, output = model(sample)
        c_loss, centers = centerloss(target, fea)
        loss = F.nll_loss(output, target) + loss_weight * c_loss
        loss.backward()
        optimizer.step()
        optimzer4center.step()
        out_label = output.max(1, keepdim=True)[1]
        total_cnt += sample.data.size()[0]
        #correct_cnt += (out_label == target.data).sum()
        correct_cnt += out_label.eq(target.view_as(out_label)).sum().item()
        acc = 100.0 * correct_cnt/ total_cnt
        if batch_idx % args.log_interval == 0:
            history['acc'].append(acc)
            print('Train Epoch: {} batch: {}\tLoss: {:.6f}\tAcc:{:.2f}'.format(
                epoch, batch_idx+1, loss.item(), acc))

    # feat = torch.cat(fea_loader, 0)
    # labels = torch.cat(tar_loader, 0)
    plt.plot(history['acc'], label='train accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    return loss, centers

def main():
    # Training settings
    ## Terminal komutundan alınan bilginin işlenmesi:
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
    use_cuda = not args.no_cuda and torch.cuda.is_available()# Cuda var mı diye kontrol edilir.
    # Rastgele sayı üretmek için:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    close_train=r'C:\Users\alpka\OneDrive\Masaüstü\train_door_pressure_classification-masterr\source_code\close_train.npy'
    data_train = np.load(close_train)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    sheduler = lr_scheduler.StepLR(optimizer, 30, gamma=0.5)
    for epoch in range(1, args.epochs + 1):
        sheduler.step()
        loss, centers = train(args, model, device, data_train, optimizer, epoch)
        # if loss < 0.02:
        #     break
    torch.save(model, 'saved_model_close.pt')
    torch.save(centers, 'centers_close.txt')
if __name__ == '__main__':
    main()