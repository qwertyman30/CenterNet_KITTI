#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import os
from progress.bar import Bar
import pycocotools.coco as coco
import time
import cv2
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.ops

from dcn_v2 import DCN

from decode import ddd_decode
from debugger import Debugger
from image import flip, color_aug
from image import get_affine_transform, affine_transform
from image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from oracle_utils import gen_oracle_map


# In[2]:


print(torch.cuda.is_available())
print(torch.version.cuda)
torch.cuda.empty_cache()

seed = 317
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)
np.random.seed(seed)

BatchNorm = nn.BatchNorm2d
BN_MOMENTUM = 0.1


# In[3]:


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return os.path.join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


# In[4]:


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# In[5]:


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


# In[6]:


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


# In[7]:


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


# In[8]:


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


# In[9]:


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


# In[10]:


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)


# In[11]:


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


# In[12]:


def set_bn(bn):
    global BatchNorm
    BatchNorm = bn
    dla.BatchNorm = bn


# In[13]:


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# In[14]:


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# In[15]:


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] =                 (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


# In[16]:


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


# In[17]:


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


# In[18]:


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


# In[19]:


class DLASeg(nn.Module):
    def __init__(self, heads, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, pretrained=False):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = dla34(pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, 
                              kernel_size=final_kernel, stride=1, 
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes, 
                               kernel_size=final_kernel, stride=1, 
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]


# In[20]:


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    neg_weights = torch.pow(1 - gt, 4)
    
    loss = 0
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


# In[21]:


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


# In[22]:


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


# In[23]:


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='mean')
        return loss


# In[24]:


def compute_rot_loss(output, target_bin, target_res, mask):
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


# In[25]:


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='mean')


# In[26]:


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='mean')


# In[27]:


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()
  
    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


# In[28]:


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


# In[29]:


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


# In[30]:


class DddLoss(torch.nn.Module):
    def __init__(self, opt):
        super(DddLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = L1Loss()
        self.crit_rot = BinRotLoss()
        self.opt = opt
        
    def forward(self, outputs, batch):
        opt = self.opt
        
        hm_loss, dep_loss, rot_loss, dim_loss = 0, 0, 0, 0
        wh_loss, off_loss = 0, 0
        for s in range(opt["num_stacks"]):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
            
            if opt["eval_oracle_dep"]:
                output['dep'] = torch.from_numpy(gen_oracle_map(
                    batch['dep'].detach().cpu().numpy(), 
                    batch['ind'].detach().cpu().numpy(), 
                    opt["output_w"], opt["output_h"])).to(opt["device"])
            
            hm_loss += self.crit(output['hm'], batch['hm']) / opt["num_stacks"]
            if opt["dep_weight"] > 0:
                dep_loss += self.crit_reg(output['dep'], batch['reg_mask'],
                                          batch['ind'], batch['dep']) / opt["num_stacks"]
                
            if opt["dim_weight"] > 0:
                dim_loss += self.crit_reg(output['dim'], batch['reg_mask'],
                                          batch['ind'], batch['dim']) / opt["num_stacks"]
            
            if opt["rot_weight"] > 0:
                rot_loss += self.crit_rot(output['rot'], batch['rot_mask'],
                                          batch['ind'], batch['rotbin'],
                                          batch['rotres']) / opt["num_stacks"]

            if opt["reg_bbox"] and opt["wh_weight"] > 0:
                wh_loss += self.crit_reg(output['wh'], batch['rot_mask'],
                                         batch['ind'], batch['wh']) / opt["num_stacks"]

            if opt["reg_offset"] and opt["off_weight"] > 0:
                off_loss += self.crit_reg(output['reg'], batch['rot_mask'],
                                          batch['ind'], batch['reg']) / opt["num_stacks"]

        loss = opt["hm_weight"] * hm_loss + opt["dep_weight"] * dep_loss +                 opt["dim_weight"] * dim_loss + opt["rot_weight"] * rot_loss +                opt["wh_weight"] * wh_loss + opt["off_weight"] * off_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'dep_loss': dep_loss, 
                      'dim_loss': dim_loss, 'rot_loss': rot_loss, 
                      'wh_loss': wh_loss, 'off_loss': off_loss}

        return loss, loss_stats


# In[31]:


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
  
    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


# In[32]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


# In[33]:


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss).to(device=opt["device"])

    def set_device(self, device):
        self.model_with_loss = self.model_with_loss.to(device)
    
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt["num_iters"] < 0 else opt["num_iters"]
        bar = Bar('{}/{}'.format(opt["task"], opt["exp_id"]), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
        
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt["device"], non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()

        
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
        
            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt["hide_data_time"]:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) '                '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt["print_iter"] > 0:
                if iter_id % opt["print_iter"] == 0:
                    print('{}/{}| {}'.format(opt["task"], opt["exp_id"], Bar.suffix))
            else:
                bar.next()

#             if opt["debug"] > 0:
#                 self.debug(batch, output, iter_id)

#             if opt["test"]:
#                 self.save_result(output, batch, results)

            del output, loss, loss_stats
            
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results
    
    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError
        
    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)
    
    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)


# In[34]:


class DddTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(DddTrainer, self).__init__(opt, model, optimizer=optimizer)
        
    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'dep_loss', 'dim_loss',
                       'rot_loss', 'wh_loss', 'off_loss']
        loss = DddLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        wh = output['wh'] if opt["reg_bbox"] else None
        reg = output['reg'] if opt["reg_offset"] else None
        dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=opt["K"])
        
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        calib = batch['meta']['calib'].detach().numpy()
        
        dets_pred = ddd_post_process(dets.copy(), batch['meta']['c'].detach().numpy(), 
                                     batch['meta']['s'].detach().numpy(), calib, opt)
        dets_gt = ddd_post_process(batch['meta']['gt_det'].detach().numpy().copy(),
                                   batch['meta']['c'].detach().numpy(), 
                                   batch['meta']['s'].detach().numpy(), calib, opt)
        
        #for i in range(input.size(0)):
        for i in range(1):
            debugger = Debugger(dataset=opt["dataset"], ipynb=(opt["debug"]==3),
                                theme=opt["debugger_theme"])
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.opt["std"] + self.opt["mean"]) * 255.).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'hm_pred')
            debugger.add_blend_img(img, gt, 'hm_gt')
            debugger.add_ct_detection(img, dets[i], show_box=opt["reg_bbox"],
                                      center_thresh=opt["center_thresh"], img_id='det_pred')
            debugger.add_ct_detection(img, batch['meta']['gt_det'][i].cpu().numpy().copy(), 
                                      show_box=opt["reg_bbox"], img_id='det_gt')
            debugger.add_3d_detection(batch['meta']['image_path'][i], dets_pred[i], calib[i],
                                      center_thresh=opt["center_thresh"], img_id='add_pred')
            debugger.add_3d_detection(batch['meta']['image_path'][i], dets_gt[i], calib[i],
                                      center_thresh=opt["center_thresh"], img_id='add_gt')
            debugger.add_bird_views(dets_pred[i], dets_gt[i],
                                    center_thresh=opt["center_thresh"], img_id='bird_pred_gt')
            debugger.compose_vis_add(batch['meta']['image_path'][i], dets_pred[i], calib[i],
                                     opt["center_thresh"], pred, 'bird_pred_gt', img_id='out')
            if opt["debug"] == 4:
                debugger.save_all_imgs(opt["debug_dir"], prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)
    
    def save_result(self, output, batch, results):
        opt = self.opt
        wh = output['wh'] if opt["reg_bbox"] else None
        reg = output['reg'] if opt["reg_offset"] else None
        dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=opt["K"])
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        calib = batch['meta']['calib'].detach().numpy()
        dets_pred = ddd_post_process(dets.copy(), batch['meta']['c'].detach().numpy(),
                                     batch['meta']['s'].detach().numpy(), calib, opt)
        img_id = batch['meta']['img_id'].detach().numpy()[0]
        results[img_id] = dets_pred[0]
        for j in range(1, opt["num_classes"] + 1):
            keep_inds = (results[img_id][j][:, -1] > opt["center_thresh"])
            results[img_id][j] = results[img_id][j][keep_inds]


# In[35]:


class KITTI(data.Dataset):
    num_classes = 3
    default_resolution = [384, 1280]
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(KITTI, self).__init__()
        self.data_dir = os.path.join(opt["data_dir"], 'kitti')
        self.img_dir = os.path.join(self.data_dir, 'images', 'trainval')
        if opt["trainval"]:
            split = 'trainval' if split == 'train' else 'test'
            self.img_dir = os.path.join(self.data_dir, 'images', split)
            self.annot_path = os.path.join(self.data_dir, 'annotations',
                                           'kitti_{}_{}.json').format(opt["kitti_split"], split)
        else:
            self.annot_path = os.path.join(self.data_dir, 
                                           'annotations', 'kitti_{}_{}.json').format(opt["kitti_split"], split)
        self.max_objs = 50
        self.class_name = ['__background__', 'Pedestrian', 'Car', 'Cyclist']
        self.cat_ids = {1:0, 2:1, 3:2, 4:-3, 5:-3, 6:-2, 7:-99, 8:-99, 9:-1}

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt
        self.alpha_in_degree = False

        print('==> initializing kitti {}, {} data.'.format(opt["kitti_split"], split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        pass

    def save_results(self, results, save_dir):
        results_dir = os.path.join(save_dir, 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for img_id in results.keys():
            out_path = os.path.join(results_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for cls_ind in results[img_id]:
                for j in range(len(results[img_id][cls_ind])):
                    class_name = self.class_name[cls_ind]
                    f.write('{} 0.0 0'.format(class_name))
                    for i in range(len(results[img_id][cls_ind][j])):
                        f.write(' {:.2f}'.format(results[img_id][cls_ind][j][i]))
                    f.write('\n')
            f.close()

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        os.system('./tools/kitti_eval/evaluate_object_3d_offline ' +                   '/data/kitti/training/label_val ' +                   '{}/results/'.format(save_dir))


# In[36]:


class DddDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _convert_alpha(self, alpha):
        return math.radians(alpha + 45) if self.alpha_in_degree else alpha

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if 'calib' in img_info:
            calib = np.array(img_info['calib'], dtype=np.float32)
        else:
            calib = self.calib

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
        if self.opt["keep_res"]:
            s = np.array([self.opt["input_w"], self.opt["input_h"]], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)
    
        aug = False
        if self.split == 'train' and np.random.random() < self.opt["aug_ddd"]:
            aug = True
            sf = self.opt["scale"]
            cf = self.opt["shift"]
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)

        trans_input = get_affine_transform(
            c, s, 0, [self.opt["input_w"], self.opt["input_h"]])
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt["input_w"], self.opt["input_h"]),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        num_classes = self.opt["num_classes"]
        trans_output = get_affine_transform(
            c, s, 0, [self.opt["output_w"], self.opt["output_h"]])

        hm = np.zeros((num_classes, self.opt["output_h"], self.opt["output_w"]), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        dep = np.zeros((self.max_objs, 1), dtype=np.float32)
        rotbin = np.zeros((self.max_objs, 2), dtype=np.int64)
        rotres = np.zeros((self.max_objs, 2), dtype=np.float32)
        dim = np.zeros((self.max_objs, 3), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        rot_mask = np.zeros((self.max_objs), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        draw_gaussian = draw_msra_gaussian if self.opt["mse_loss"] else draw_umich_gaussian
    
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id <= -99:
                continue
      
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt["output_w"] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt["output_h"] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((h, w))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
        
                if cls_id < 0:
                    ignore_id = [_ for _ in range(num_classes)] if cls_id == - 1 else  [- cls_id - 2]
                    
                    if self.opt["rect_mask"]:
                        hm[ignore_id, int(bbox[1]): int(bbox[3]) + 1, 
                        int(bbox[0]): int(bbox[2]) + 1] = 0.9999
                    else:
                        for cc in ignore_id:
                            draw_gaussian(hm[cc], ct, radius)
                        hm[ignore_id, ct_int[1], ct_int[0]] = 0.9999
                    continue
                draw_gaussian(hm[cls_id], ct, radius)

                wh[k] = 1. * w, 1. * h
                gt_det.append([ct[0], ct[1], 1] + self._alpha_to_8(self._convert_alpha(ann['alpha'])) +                               [ann['depth']] + (np.array(ann['dim']) / 1).tolist() + [cls_id])
                if self.opt["reg_bbox"]:
                    gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]
                if 1:
                    alpha = self._convert_alpha(ann['alpha'])
                    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                        rotbin[k, 0] = 1
                        rotres[k, 0] = alpha - (-0.5 * np.pi)    
                    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                        rotbin[k, 1] = 1
                        rotres[k, 1] = alpha - (0.5 * np.pi)
                    dep[k] = ann['depth']
                    dim[k] = ann['dim']
                    ind[k] = ct_int[1] * self.opt["output_w"] + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1 if not aug else 0
                    rot_mask[k] = 1
        ret = {'input': inp, 'hm': hm, 'dep': dep, 'dim': dim, 'ind': ind, 
               'rotbin': rotbin, 'rotres': rotres, 'reg_mask': reg_mask,
               'rot_mask': rot_mask}
        if self.opt["reg_bbox"]:
            ret.update({'wh': wh})
        if self.opt["reg_offset"]:
            ret.update({'reg': reg})
#         if self.opt["debug"] > 0 or not ('train' in self.split):
#             gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, 18), dtype=np.float32)
#             meta = {'c': c, 's': s, 'gt_det': gt_det, 'calib': calib, 'image_path': img_path, 'img_id': img_id}
#             ret['meta'] = meta

        return ret

    def _alpha_to_8(self, alpha):
        ret = [0, 0, 0, 1, 0, 0, 0, 1]
        if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            r = alpha - (-0.5 * np.pi)
            ret[1] = 1
            ret[2], ret[3] = np.sin(r), np.cos(r)
        if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            r = alpha - (0.5 * np.pi)
            ret[5] = 1
            ret[6], ret[7] = np.sin(r), np.cos(r)
        return ret


# In[37]:


def update_dataset_info_and_set_heads(opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt["mean"], opt["std"] = dataset.mean, dataset.std
    opt["num_classes"] = dataset.num_classes

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    opt["input_h"] = input_h
    opt["input_w"] = input_w
    opt["output_h"] = opt["input_h"] // opt["down_ratio"]
    opt["output_w"] = opt["input_w"] // opt["down_ratio"]
    opt["input_res"] = max(opt["input_h"], opt["input_w"])
    opt["output_res"] = max(opt["output_h"], opt["output_w"])

    opt["heads"] = {'hm': opt["num_classes"], 'dep': 1, 'rot': 8, 'dim': 3}
    if opt["reg_bbox"]:
        opt["heads"].update({'wh': 2})
    if opt["reg_offset"]:
        opt["heads"].update({'reg': 2})
    print('heads', opt["heads"])
    return opt


# In[38]:


def get_dataset():
    class Dataset(KITTI, DddDataset):
        pass
    return Dataset


# In[39]:


opt = {}
opt["head_conv"] = 256
opt["num_stacks"] = 1
opt["center_thresh"] = 0.1
opt["K"] = 100
opt["reg_offset"] = True
opt["reg_bbox"] = True
opt["debug_dir"] = "Debug"
opt["dataset"] = "kitti"
opt["std"] = [0.229, 0.224, 0.225]
opt["mean"] = [0.485, 0.456, 0.406]
opt["num_classes"] = 3
opt["debugger_theme"] = "white"
opt["exp_id"] = "default"
opt["num_iters"] = -1
opt["task"] = 'ddd'
opt["device"] = "cuda"
opt["print_iter"] = 0
opt["hide_data_time"] = True
opt["lr"] = 1.25e-4
opt["lr_step"] = [45, 60]
opt["trainval"] = False
opt["data_dir"] = "data"
opt["kitti_split"] = "3dop"
opt["down_ratio"] = 4
opt["batch_size"] = 4
opt["num_epochs"] = 70
opt["keep_res"] = False
opt["nms"] = False
opt["no_color_aug"] = False
opt["norm_wh"] = False
opt["reg_loss"] = "l1"
opt["scores_thresh"] = 0.1
opt["aug_ddd"] = 0.5
opt["mse_loss"] = False
opt["scale"] = 0.4
opt["rect_mask"] = False
opt["shift"] = 0.1
opt["eval_oracle_dep"] = False
opt["dep_weight"] = 1
opt["dim_weight"] = 1
opt["rot_weight"] = 1
opt["wh_weight"] = 0.1
opt["off_weight"] = 1
opt["hm_weight"] = 1
opt["test"] = False
opt["val_intervals"] = 5
opt["metric"] = "loss"
opt["test_scales"] = [1.0]
opt["peak_thresh"] = 0.2
opt["vis_thresh"] = 0.3
opt["debug"] = 0
opt["save_dir"] = "results/"


# In[40]:


Dataset = get_dataset()
opt = update_dataset_info_and_set_heads(opt, Dataset)
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# In[41]:


model = DLASeg(opt["heads"],
               final_kernel=1,
               last_level=5,
               head_conv=opt["head_conv"],
               down_ratio=opt["down_ratio"],
               pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), opt["lr"])

trainer = DddTrainer(opt, model, optimizer)
trainer.set_device(opt["device"])


# In[42]:


train_loader = torch.utils.data.DataLoader(
      Dataset(opt, "train"),
      batch_size=opt["batch_size"],
      shuffle=True,
      num_workers=4,
      pin_memory=True,
      drop_last=True)

val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
)


# In[ ]:


best = 1e10
losses, hm_losses, dep_losses, dim_losses, rot_losses, wh_losses, off_losses = [], [], [], [], [], [], []
losses_val, hm_losses_val, dep_losses_val, dim_losses_val, rot_losses_val, wh_losses_val, off_losses_val = [], [], [], [], [], [], []

for epoch in tqdm(range(1, opt["num_epochs"] + 1)):
    log_dict_train, _ = trainer.train(epoch, train_loader)
    losses.append(log_dict_train["loss"])
    hm_losses.append(log_dict_train["hm_loss"])
    dep_losses.append(log_dict_train["dep_loss"])
    dim_losses.append(log_dict_train["dim_loss"])
    rot_losses.append(log_dict_train["rot_loss"])
    wh_losses.append(log_dict_train["wh_loss"])
    off_losses.append(log_dict_train["off_loss"])
    print("EPOCH: {}, LOSS: {}, HM_LOSS: {}, DEP_LOSS:{}, DIM_LOSS: {}, ROT_LOSS: {}, WH_LOSS: {}, OFF_LOSS: {}".format(
        epoch, log_dict_train["loss"], log_dict_train["hm_loss"], log_dict_train["dep_loss"],
        log_dict_train["dim_loss"], log_dict_train["rot_loss"], log_dict_train["wh_loss"],
        log_dict_train["off_loss"]))
    #torch.save(model.state_dict(), "centernet_{}.pth".format(epoch))

    if opt["val_intervals"] > 0 and epoch % opt["val_intervals"] == 0:
        torch.save(model.state_dict(), "centernet_val_{}.pth".format(epoch))
        with torch.no_grad():
            log_dict_val, preds = trainer.val(epoch, val_loader)
            losses_val.append(log_dict_val["loss"])
            hm_losses_val.append(log_dict_val["hm_loss"])
            dep_losses_val.append(log_dict_val["dep_loss"])
            dim_losses_val.append(log_dict_val["dim_loss"])
            rot_losses_val.append(log_dict_val["rot_loss"])
            wh_losses_val.append(log_dict_val["wh_loss"])
            off_losses_val.append(log_dict_val["off_loss"])

            print("VALIDATION EPOCH: {}, LOSS: {}, HM_LOSS: {}, DEP_LOSS:{}, DIM_LOSS: {}, ROT_LOSS: {}, WH_LOSS: {}, OFF_LOSS: {}".format(
                epoch, log_dict_val["loss"], log_dict_val["hm_loss"], log_dict_val["dep_loss"],
                log_dict_val["dim_loss"], log_dict_val["rot_loss"], log_dict_val["wh_loss"],
                log_dict_val["off_loss"]))
            if log_dict_val[opt["metric"]] < best:
                best = log_dict_val[opt["metric"]]
                torch.save(model.state_dict(), "centernet_best.pth")

    if epoch in opt["lr_step"]:
        lr = opt["lr"] * (0.1 ** (opt["lr_step"].index(epoch) + 1))
        print('Drop LR to', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# In[ ]:


torch.save(model.state_dict(), "centernet_last.pth")


# In[ ]:


plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Losses")
plt.savefig("Losses.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(hm_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("hm_losses")
plt.savefig("hm_losses.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(dep_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("dep_losses")
plt.savefig("dep_losses.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(dim_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("dim_losses")
plt.savefig("dim_losses.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(rot_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("rot_losses")
plt.savefig("rot_losses.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(wh_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("wh_losses")
plt.savefig("wh_losses.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(off_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("off_losses")
plt.savefig("off_losses.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(losses_val)
plt.xlabel("Epochs (x5)")
plt.ylabel("Loss")
plt.title("Losses val")
plt.savefig("Losses_val.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(hm_losses_val)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("hm_losses_val")
plt.savefig("hm_losses_val.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(dep_losses_val)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("dep_losses_val")
plt.savefig("dep_losses_val.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(dim_losses_val)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("dim_losses_val")
plt.savefig("dim_losses_val.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(rot_losses_val)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("rot_losses_val")
plt.savefig("rot_losses_val.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(wh_losses_val)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("wh_losses_val")
plt.savefig("wh_losses_val.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:


plt.plot(off_losses_val)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("off_losses_val")
plt.savefig("off_losses_val.png")
plt.show()
plt.clf()
plt.cla()
plt.close()


# In[ ]:




