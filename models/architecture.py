# Design differnent basic network architecture aginst MSA
import functools
from functools import partial, reduce
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import torch.nn as nn
from torch import Tensor
from torch.optim import lr_scheduler

try:
    from torch.nn import Flatten
except ImportError:
    class Flatten(nn.Module):
        r"""
        Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.

        Shape:
            - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)`,'
            where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
            number of dimensions including none.
            - Output: :math:`(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)`.

        Args:
            start_dim: first dim to flatten (default = 1).
            end_dim: last dim to flatten (default = -1).

        Examples::
            >>> input = torch.randn(32, 1, 5, 5)
            >>> m = nn.Sequential(
            >>>     nn.Conv2d(1, 32, 5, 1, 1),
            >>>     nn.Flatten()
            >>> )
            >>> output = m(input)
            >>> output.size()
            torch.Size([32, 288])
        """
        __constants__ = ['start_dim', 'end_dim']
        start_dim: int
        end_dim: int

        def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
            super(Flatten, self).__init__()
            self.start_dim = start_dim
            self.end_dim = end_dim

        def forward(self, input: Tensor) -> Tensor:
            return input.flatten(self.start_dim, self.end_dim)

        def extra_repr(self) -> str:
            return 'start_dim={}, end_dim={}'.format(
                self.start_dim, self.end_dim
            )
try:
    from torch.nn import Identity
except ImportError:
    class Identity(nn.Module):
        def forward(self, x):
            return x

import os
import sys

prj_dir = os.path.dirname(os.path.dirname(__file__))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

from math import ceil

def get_filter(filt_size:int = 3):
    if(filt_size == 1):
        a : Tensor = torch.tensor([1.])
    elif(filt_size == 2):
        a : Tensor = torch.tensor([1., 1.])
    elif(filt_size == 3):
        a : Tensor = torch.tensor([1., 2., 1.])
    elif(filt_size == 4):
        a : Tensor= torch.tensor([1., 3., 3., 1.])
    elif(filt_size == 5):
        a : Tensor = torch.tensor([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a : Tensor = torch.tensor([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a : Tensor = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
    else:
        raise NotImplementedError(f"filt_size = {filt_size} is not implemented")

    # filt = torch.Tensor(a[:, None] * a[None, :])
    # filt = a[:, None] * a[None, :]
    a = a.reshape((-1,1))
    filt = torch.matmul(a, torch.transpose(a, 0, 1))
    filt = filt / filt.sum()

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        # self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        r, c = filt.shape
        self.register_buffer('filt', filt.reshape((1, 1, r, c)).repeat(self.channels, 1, 1, 1))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp:Tensor):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        # self.filt_odd = np.mod(filt_size, 2) == 1
        self.filt_odd = (filt_size % 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        # self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        r, c = filt.shape
        self.register_buffer('filt', filt.reshape((1, 1, r, c)).repeat(self.channels, 1, 1, 1))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp : Tensor):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def get_norm_layer(norm_type="instance"):
    """
    """
    layer_dict = dict(
        batch=partial(nn.BatchNorm2d, affine=True, track_running_stats=True),
        instance=partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
        none=nn.Identity()
    )
    try:
        return layer_dict[norm_type]
    except Exception as _:
        raise NotImplementedError(f"normalization layer {norm_type} is not found")

def get_scheduler(optimizer, opt, **kArgs):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.epochs) / float(opt.epoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net : nn.Module, init_type : str='normal', init_gain: float=0.02, debug:bool=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m : nn.Module):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net : nn.Module, init_type : str='normal', init_gain : float=0.02, 
             gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def get_conv1d_dim(l_in, kernel_size, padding=0, stride=1, dilation=1):
    l_out = (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return l_out

def get_conv2d_dim(input_shape, kernel_size, padding=0, stride=1, dilation=1):
    h_in, w_in = input_shape

    if isinstance(kernel_size, int):
        k0 = k1 = kernel_size
    else:
        k0, k1 = kernel_size

    if isinstance(padding, int):
        p0 = p1 = padding
    else:
        p0, p1 = padding
    
    if isinstance(stride, int):
        s0 = s1 = stride
    else:
        s0, s1 = stride

    if isinstance(dilation, int):
        d0 = d1 = dilation
    else:
        d0, d1 = dilation

    h_out = (h_in + 2 * p0 - d0 * (k0 - 1) - 1) / s0 + 1
    w_out = (w_in + 2 * p1 - d1 * (k1 - 1) - 1) / s1 + 1

    return h_out, w_out

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=6, padding_type='reflect', 
                 no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input: Tensor):
        """Standard forward"""
        fake : Tensor = self.model(input)
        return fake


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim), 
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class UnetGenerator(nn.Module):
    """
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        """

        super(UnetGenerator, self).__init__()

        # innermost
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc = None, submodule=None, norm_layer=norm_layer, innermost=True)

        def add_intermediate_layers(ub, i):
            ub = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=ub, norm_layer=norm_layer, use_dropout=use_dropout)
            return ub
        
        unet_block = reduce(add_intermediate_layers, range(num_downs - 5), unet_block)

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # outermost
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        """
    
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if isinstance(norm_layer, partial):
            use_bias = isinstance(norm_layer.func, nn.InstanceNorm2d)
        else:
            use_bias = isinstance(norm_layer, nn.InstanceNorm2d)
        
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_channels, num_classes, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            in_channels (int)  -- the number of channels in input images
            num_classes     -- the number of classes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        self.num_classes = num_classes
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        x = self.model(x)
        return x

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

#### conv seq 2 seq
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, kernel_size, dropout,device,max_length = 2000):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size mulst be odd"

        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.max_length = max_length

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim, out_channels= 2 * hid_dim, 
                                    kernel_size=kernel_size,padding= (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # src = [batch size, src len, embding dim]

        batch_size, src_len, emb_dim = src.shape

        self.tok_embedding = src
        self.pos_embedding = nn.Embedding(self.max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, self.hid_dim)
        self.hid2emb = nn.Linear(self.hid_dim, emb_dim)

        # create position tensor
        # pos = [0, 1, 2, 3, ..., src len - 1]

        # pos = [batch size, src len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        tok_embedded = src
        pos_embedded = self.pos_embedding(pos)

        embedded = self.dropout(tok_embedded + pos_embedded)

        conv_input = self.emb2hid(embedded)

        conv_input = conv_input.permute(0, 2, 1)

        def conv_block(resu, p):
            conved, conv_input = resu
            conv = p
            conved = conv(self.dropout(conv_input))

            conved = F.glu(conved, dim=1)

            conved = (conved + conv_input) * self.scale

            conv_input = conved

            return conved, conv_input
        
        conved, conv_input = reduce(conv_block, self.convs, (None, conv_input))

        conved = self.hid2emb(conved.permute(0, 2, 1))

        combined = (conved + embedded) * self.scale

        return conved, combined
        

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, kernel_size, dropout, trg_pad_idx, device, max_length = 2000):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.output_dim = output_dim
        self.max_length = max_length
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.hid_dim = hid_dim
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim, out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, endcoder_conved, encoder_combined):
        conved_emb = self.attn_emb2hid(conved.permute(0, 2, 1))

        combined = (conved_emb + embedded) * self.scale

        energy = torch.matmul(combined, endcoder_conved.permute(0, 2, 1))

        attention = F.softmax(energy, dim = 2)

        attended_encoding = torch.matmul(attention, encoder_combined)

        attended_encoding = self.attn_emb2hid(attended_encoding)

        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        # trg [batch size, trg len, embedding dim]
        batch_size, trg_len, emb_dim = trg.shape
        
        self.pos_embedding = nn.Embedding(self.max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, self.hid_dim)
        self.hid2emb = nn.Linear(self.hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(self.hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, self.hid_dim)

        self.fc_out = nn.Linear(emb_dim, self.output_dim)

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        tok_embedded = trg
        pos_embedded = self.pos_embedding(pos)

        embedded = self.dropout(tok_embedded + pos_embedded)

        conv_input = self.emb2hid(embedded)

        batch_size, = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        def cal_attn(resu, conv):
            attention, conved, conv_input = resu

            conv_input = self.dropout(conv_input)

            padding = torch.zeros(batch_size, hid_dim, self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            padded_conv_input = torch.cat((padding, conv_input), dim = 2)

            conved = conv(padded_conv_input)

            conved = F.glu(conved, dim = 1)

            attention, conved = self.calculate_attention(embedded, conved, encoder_conved, encoder_combined)

            conved = (conved + conv_input) * self.scale

            conv_input = conved

            return attention, conved, conv_input

        attention, conved, conv_input = reduce(cal_attn, self.convs, (None, None, conv_input))

        conved = self.hid2emb(conved.permute(0, 2, 1))

        output = self.fc_out(self.dropout(conved))

        return output, attention

class Concatenate(nn.Module):
    def __init__(self, module_list: Iterable[nn.Module], axis: int):
        super(Concatenate, self).__init__()

        self.axis= axis
        self.module_list = module_list
    
    def forward(self, input):
        tensors = [m(input) for m in self.module_list]
        return torch.cat(tensors, axis=self.axis)


## DeepGOPlus convolutional strategy
class DGPConv(nn.Module):
    """
    """
    def __init__(self, input_nc, output_nc, max_kernel_size, kernel_sizes=None, 
                 **kArgs):
        """
        """
        self.input_nc = input_nc
        self.output_nc = output_nc

        if kernel_sizes is None:
            kernel_sizes = range(8, max_kernel_size, 8)

        self.kernel_sizes = kernel_sizes

        if kArgs.get("reverse_flag") is not None:
            if kArgs["reverse_flag"]:
                kernel_sizes = reversed(kernel_sizes)

        super(DGPConv, self).__init__()

        if kArgs.get("max_len"):
            self.max_len = kArgs["max_len"]
        else:
            self.max_len = 2000
        def defing_convs(convs, i):
            conv = nn.Conv1d(self.input_nc, self.output_nc,self.kernel_sizes[i],
                          padding=0)
            # pool = nn.MaxPool1d(self.max_len-self.kernel_sizes[i] + 1, stride=1,padding=0)
            pool = nn.AdaptiveMaxPool1d(1)
            flat = nn.Flatten()

            m = nn.Sequential(conv, pool, flat)
            convs.append(m)
            return convs

        convs = []
        convs = reduce(defing_convs, range(len(kernel_sizes)), convs)

        # net = Concatenate(convs, axis=1)
        net = nn.ModuleList(convs)
        self.model = net
    
    def get_output_dim(self):
        # conv1d or pool
        # def _get_conv1d_dim(kernel_size, padding=0, stride=1, dilation=1, l_in = self.max_len):
        #     l_out = (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        #     return l_out

        # conv_out_dims = [_get_conv1d_dim(x) for x in self.kernel_sizes]

        # pool_out_dims = [_get_conv1d_dim(self.max_len - x + 1, l_in=conv_out_dims[i]) 
        #                  for i, x in enumerate(self.kernel_sizes)]

        # flat_out_dims = [self.output_nc * pool_out_dims[i]
        #                  for i in range(len(self.kernel_sizes))]
        
        # return int(sum(flat_out_dims))

        n_kernels = len(self.kernel_sizes)
        return n_kernels * self.output_nc

        
    def forward(self, input):
        # print(f"intput: {input.shape}")
        tensors = [m(input) for m in self.model]
        # print("tensors: {}".format([x.shape for x in tensors]))
        # output = rearrange(tensors, "k b c -> b (k c)")
        output = tensors.transpose(0,1).flatten(-2,-1)
        # print(f"output: {output.shape}")
        return output
