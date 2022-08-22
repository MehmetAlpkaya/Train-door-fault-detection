from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.autograd.function import Function

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)) #[3,2]
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)

        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss,self.centers


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        expanded_centers = centers.expand(int(batch_size), -1, -1) #[128,10,2]
        expanded_feature = feature.expand(4, -1, -1).transpose(1, 0) #[64,4,2]
        distance_centers = (expanded_feature - expanded_centers).pow(2).sum(dim=-1) #[64,4]
        label=label.type(torch.int64)
        distances_same = distance_centers.gather(1, label.unsqueeze(1)) #[64,1]
        distance_centers.scatter_(1, label.unsqueeze(1), 0)
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum()
        return (intra_distances - 0.1/3.0 * inter_distances) / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long()) #[64,2]
        diff = centers_batch - feature #[64,2]
        counts = centers.new_ones(centers.size(0)) #[1,1,1,1]
        ones = centers.new_ones(label.size(0)) #[1]
        grad_centers = centers.new_zeros(centers.size()) #[4,2]

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_1 = grad_centers/counts.view(-1, 1)

        return - grad_output * diff / batch_size, None, grad_1 / batch_size, None


def main(test_cuda=False):
    print('-'*80)
    device = torch.device("cuda" if test_cuda else "cpu")
    ct = CenterLoss(10,2,size_average=True).to(device)
    y = torch.Tensor([0,0,2,1]).to(device)
    feat = torch.zeros(4,2).to(device).requires_grad_()
    print (list(ct.parameters()))
    print (ct.centers.grad)
    out = ct(y,feat)
    print(out.item())
    out.backward()
    print(ct.centers.grad)
    print(feat.grad)

if __name__ == '__main__':
    torch.manual_seed(999)
    main(test_cuda=False)
    if torch.cuda.is_available():
        main(test_cuda=True)
