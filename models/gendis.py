## generator and discriminator
from argparse import ArgumentParser, Namespace
import pickle
import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

import os
import sys

prj_dir = os.path.dirname(os.path.dirname(__file__))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

from experiments.go import MSAEncoder

from models.architecture import get_scheduler
from models.architecture import NLayerDiscriminator, PixelDiscriminator, get_norm_layer, init_net
from models.architecture import ResnetGenerator
from resnet import resnet18, resnet34, resnet50,resnet101, resnet152
from resnet import ResNet
from resnet import resnext50_32x4d, resnext101_32x8d
from resnet import wide_resnet50_2, wide_resnet101_2
from models.utils import parsing

cuda = True if torch.cuda.is_available() else False

class Arch(nn.Module):
    def __init__(self, opt : Namespace):
        """
        in_channels: int
        out_channels_G: int
        num_classes: int
        top_k: int
        max_len: int
        msa_cutoff: float
        msa_penalty: float
        msa_embedding_dim: int
        msa_encoding_strategy: str
        ngf: int
        netG: str
        no_antialias: bool
        no_antialias_up: bool
        load_gen: Optional[str]
        freeze_gen: Optional[str]
        ndf: int
        netD: str
        replace_stride_with_dilation: List[bool]
        no_dropout: bool
        normG: str
        normD: str
        init_type: str
        init_gain: float
        gpu_ids: List[int]
        no_jit: bool
        """
        super(Arch, self).__init__()

        self.in_channels = opt.in_channels
        self.out_channels_G = opt.out_channels_G
        self.num_classes = opt.num_classes
        self.gpu_ids = opt.gpu_ids
        self.no_jit = opt.no_jit

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        need_init = opt.load_gen is None

        pre_model = MSAEncoder(opt.msa_cutoff, opt.msa_penalty, opt.msa_embedding_dim, 
                                encoding_strategy=opt.msa_encoding_strategy)
        # pre_model = nn.Embedding(21, embedding_dim)
        if not self.no_jit:
            pre_model = torch.jit.script(pre_model)
        # the weight of pre_model should also be loaded
        self.pre_model = init_net(pre_model, init_type=opt.init_type, init_gain=opt.init_gain,
                                  gpu_ids=opt.gpu_ids, initialize_weights=need_init)

        # nets = []
        # input: b * 21 * seqLen * topK
        # update
        # input: b * 441 * seqLen(patched) * seqLen(patched)
        # defaults is b * 441 * 128 * 128
        
        self.gnet = define_G(self.in_channels, self.out_channels_G,ngf=opt.ngf, netG=opt.netG,
                             norm=opt.normG, use_dropout=not opt.no_dropout,
                             init_type=opt.init_type, init_gain=opt.init_gain,
                             no_antialias=opt.no_antialias, no_antialias_up=opt.no_antialias_up,
                             gpu_ids=opt.gpu_ids,
                             need_init_weights=need_init,
                             no_jit=self.no_jit)

        # load the pre_model and the gnet weight
        if not need_init:
            gen_model_path = opt.load_gen

            pnet_state_dict = self.pre_model.state_dict()
            gnet_state_dict = self.gnet.state_dict()

            try:
                trained_state_dict = torch.load(gen_model_path)
            except Exception as _:
                with open(gen_model_path, "rb") as h:
                    trained_state_dict = pickle.load(h)
            
            ploaded_state_dict = {k: v for k, v in trained_state_dict.items()
                                  if pnet_state_dict.get(k) is not None}
            gloaded_state_dict = {k: v for k, v in trained_state_dict.items()
                                  if gnet_state_dict.get(k) is not None}

            pnet_state_dict.update(ploaded_state_dict)
            gnet_state_dict.update(gloaded_state_dict)

            self.pre_model.load_state_dict(pnet_state_dict)
            self.gnet.load_state_dict(gnet_state_dict)

            print("loading")
            
        # freeze pre_mode and gnet
        if opt.__dict__.get("freeze_gen", False):
            for param in self.pre_model.parameters():
                param.requires_grad = False

            for param in self.gnet.parameters():
                param.requires_grad = False
            
            print("freezing")

        params = opt.__dict__
        params = {k: params[k] for k in get_init_keys(name2model[opt.netD])}

        self.rnet = define_D(self.out_channels_G, self.num_classes, netD=opt.netD,
                             init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids,
                             **params)
            
    def forward(self, x : Tensor):
        x = self.pre_model(x)
        x = x.permute(0, 3, 2, 1) # b c h w -> b w h c
        if self.gnet is not None:
            x = self.gnet(x)
        x = self.rnet(x)
        return x


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None,
             need_init_weights = True, no_jit = False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2.
        need_init_weights (bool) -- control whether to initialize the net weights
    """

    norm_layer = get_norm_layer(norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=9, opt=opt)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=4, opt=opt)
    elif netG == "resnet_2blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up,
                              n_blocks=2, opt=opt)
    elif netG == "resnet_oneblock":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up,
                              n_blocks=1, opt=opt)
    elif netG is None or netG == "none":
        net = None
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    if not no_jit and net is not None:
        net = torch.jit.script(net)

    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=need_init_weights) \
            if net is not None else net

name2net = {
    "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,"resnet101": resnet101, "resnet152" : resnet152,
    "resnext50_32x4d": resnext50_32x4d, "resnext101_32x8d": resnext101_32x8d,
    "wide_resnet50_2":wide_resnet50_2, "wide_resnet101_2": wide_resnet101_2,
    "n_layers": NLayerDiscriminator,
    "pix":PixelDiscriminator
}

name2model = {
    "resnet18": ResNet, "resnet34": ResNet, "resnet50": ResNet,"resnet101": ResNet, "resnet152" : ResNet,
    "resnext50_32x4d": ResNet, "resnext101_32x8d": ResNet,
    "wide_resnet50_2":ResNet, "wide_resnet101_2": ResNet,
    "n_layers": NLayerDiscriminator,
    "pix":PixelDiscriminator
}

def get_init_keys(net):
    if net == ResNet:
        init_keys = ["replace_stride_with_dilation", "normD", "no_jit"]
    elif net == NLayerDiscriminator:
        init_keys = ["n_layers_D", "normD", "ndf"]
    else:
        init_keys = []

    return init_keys

def define_D(in_channels : int, num_classes : int, netD : str,
             init_type='normal', init_gain=0.02, gpu_ids = [], need_init_weights = True,
             **params):
        assert  name2net.get(netD) is not None, \
            f"{netD} is not implemented"
        
        norm_type = params.get("normD", None)
        if norm_type is not None:
            del params["normD"]
            params["norm_layer"] = get_norm_layer(norm_type)
        
        n_layers = params.get("n_layers_D", None)
        if n_layers is not None:
            params["n_layers"] = n_layers
            del params["n_layers_D"]

        rnet =  name2net[netD](in_channels=in_channels, num_classes=num_classes, **params)
        if need_init_weights:
            rnet = init_net(rnet, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)

        return rnet
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def compute_gradient_penalty(netD, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = netD(interpolates)
    fake = Variable(Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def main():
    """
    """
    parser = ArgumentParser()

    parser.add_argument("--in-channels", dest="in_channels", type=int, default=21)
    parser.add_argument("--out-channels-G", dest="out_channels_G", type=int, default=42)
    parser.add_argument("--num-classes",dest="num_classes", type=int, default=19939)

    parser.add_argument("--top-k",dest="top_k",type=int,default=100,
                        help="select the top k sequence ib msa")
    parser.add_argument("--max-len",dest="max_len", type=int, default=1000,
                        help="the max lenght of sequence for using")

    parser.add_argument("--msa-cutoff", dest="msa_cutoff", type=float,default=0.8,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-penalty", dest="msa_penalty", type=float,default=4.5,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-embedding-dim", dest="msa_embedding_dim", type=int, default=21,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-encoding-strategy", dest="msa_encoding_strategy", type=str, 
                        choices=["one_hot", "emb", "emb_plus_one_hot", "emb_plus_pssm","fast_dca"], 
                        default="emb_plus_one_hot",
                        help="parameters for encoding the msa file")

    parser.add_argument('--ngf', type=int, default=64, 
                        help='# of gen filters in the last conv layer')
    parser.add_argument('--netG', type=str, default='resnet_4blocks', 
                        help="specify generator architecture " +
                        "[resnet_9blocks | resnet_6blocks | resnet_4blocks]")
    parser.add_argument("--no-antialias", dest="no_antialias", action="store_true",
                        help="use dilated convolution blocks in generator")
    parser.add_argument("--no-antialias-up", dest="no_antialias_up", action="store_true",
                        help="use dilated convolution_transposed blocks in generator")
    parser.add_argument("--load-gen", dest="load_gen", type=str,
                        help="the path of pre-trained generator model")
    parser.add_argument("--freeze-gen", dest="freeze_gen", action="store_true",
                        help="control whether to freeze the generator model when training")
    parser.add_argument('--ndf', type=int, default=64, 
                            help='# of dis filters in the last conv layer')
    parser.add_argument('--netD', type=str, default='resnet50v2', 
                        help='specify discriminator architecture '+
                        '[resnet18 | resnet34 | resnet50 | resnet101 | resnet152] or ' + 
                        '[resnext50_32x4d | resnext101_32x8d]' +
                        '[wide_resnet50_2 | wide_resnet101_2]'+
                        '[basic]')

    parser.add_argument("--dilation", nargs=3, dest="replace_stride_with_dilation", 
                        default=[False, False, False], type=bool,
                        help="using dilation to replace the stride in resnet")
    parser.add_argument('--no-dropout', dest="no_dropout",action='store_true', 
                        help='no dropout for the generator', default=False)
    parser.add_argument('--normG', type=str, default='instance', 
                        help='for generator, instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--normD', type=str, default='batch', 
                            help='for discriminator, instance normalization or batch normalization [instance | batch | none]')    
     
    parser.add_argument('--init-type', dest="init_type", type=str, default='xavier', 
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init-gain', dest="init_gain", type=float, default=0.02, 
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--gpu-ids', dest="gpu_ids",type=str, default='0,1', 
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument("--no-jit", dest="no_jit", action="store_true",
                        help="not use torch.jit.script")
    opt : Namespace
    opt = parsing(parser)

    print(opt)

    model_arch = Arch(opt).cuda()
    print(model_arch)
    x = torch.randint(0, 21,(12, 40, 2000)).cuda()
    x = model_arch(x)

if __name__ == "__main__":
    main()
