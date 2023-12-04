import torch
from typing import *
import torch.nn as nn   # Basic building blocks for graphs
import torch.optim as optim   # optimization algorithms (e.g. optimizer)
import torch.nn.functional as F   # Activation functions & loss functions


def autopad(k, p=None):  # kernel, padding
    ''' 
    Padding='same'
        - 입력 데이터의 w와 h를 보존하는 k와 p를 자동으로 세팅
        - e.g. k=3 + p=1 / k=2 + p=0 
    '''
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class CBS(nn.Module):
    '''
    Conv + Batch + SiLU
        - 1 x 1 conv               ::  mainly used to change the number of channels
        - 3 x 3 conv with stride 1 ::  mainly used for feature extraction
        - 3 x 3 conv with stride 2 ::  mainly used for downsampling
    '''
    def __init__(self, in_c:int, out_c:int, k:int, s:int, p:int, g=1):
        super(CBS, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=s, padding=p, groups=g, bias=False), # NOTE: bias=True as default 
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layers(x)


class CBM(nn.Module):
    '''
    Conv + Batch + Sigmoid
        - 1 x 1 conv               ::  mainly used to change the number of channels
        - Sigmoid                  ::  Output between 0 and 1
    '''
    def __init__(self, in_c:int, out_c:int, k:int, s:int, p:int, g=1):
        super(CBM, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=s, padding=p, groups=g, bias=False), # NOTE: bias=True as default 
            nn.BatchNorm2d(out_c),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)


class Concat(nn.Module):
    '''
    Args
        - Input :: (n, c, h, w), (n, c, h, w)...
        - Output :: (n, c*n, h, w)
    '''
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension
        
    def forward(self, x):
        # NOTE: [b, c+c.., h, w]
        return torch.cat(tensors=x, dim=self.d) 
    
    
class MP1(nn.Module):
    '''
    Max Pooling for BACKBONE
    MP1 : C -> C
        - Two branches
            1. passes through Maxpool for downsampling + 1x1 convolution to change number of channels
            2. 1x1 conv + 3x3 conv with stride 2 for downsampling
            3. Concatenation (cat) to obtain the result of super downsampling
    5 layers
    '''
    def __init__(self, in_c:int, k=2):
        super(MP1, self).__init__()
        
        '''
        Args
            - Input :: (n, in_c, h, w)
            - Output :: (n, in_c, h//2, w//2)
        '''
        
        # Maxpool 
        # NOTE:  Output :: (n, in_c, h//2, w//2)
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)    
        
        # 1x1 conv
        # NOTE: Output  :: (n, in_c//2, h//2, w//2)
        self.conv_1_1 = CBS(in_c=in_c, out_c=in_c//2, k=1, s=1, p=0) 
        
        # 1x1 conv
        # NOTE: Output  :: (n, in_c//2, h//2, w//2)
        self.conv_1_2 = CBS(in_c=in_c, out_c=in_c//2, k=1, s=1, p=0)
        
        # 3x3 conv with stride 2
        # NOTE: Output  :: (n, in_c//2, h//2, w//2)
        self.conv_3_1 = CBS(in_c=in_c//2, out_c=in_c//2, k=3, s=2, p=1)
        
        # concatenation
        self.concat = Concat()
        
    def forward(self, x):
        
        # branch 1
        # NOTE: Maxpool + 1x1 conv
        out = self.m(x)
        out_1 = self.conv_1_1(out)
        
        # branch 2
        # NOTE : 1x1 conv + 3x3 conv with s 2
        out_2 = self.conv_1_2(x)
        out_3 = self.conv_3_1(out_2)
        
        # self.conv_3_1 + self.conv_1_1
        # NOTE:  Output :: (n, in_c, h//2, w//2)
        result = self.concat([out_3, out_1])
        
        return result


class MP2(nn.Module):
    '''
    Max Pooling for BACKBONE
    MP2 : C -> 2C
        - Two branches
            1. passes through Maxpool for downsampling + 1x1 convolution 
            2. 1x1 conv + 3x3 conv with stride 2 for downsampling
            3. Concatenation (cat) to obtain the result of super downsampling
    
    5 layers
        - Input :: (n, in_c, h, w)
        - Output :: (n, in_c*2, h//2, w//2)
    '''
    def __init__(self, in_c:int, k=2):
        super(MP2, self).__init__()
        
        # Maxpool 
        # NOTE:  Output :: (n, in_c, h//2, w//2)
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)    
        
        # 1x1 conv
        # NOTE: Output  :: (n, in_c, h//2, w//2)
        self.conv_1_1 = CBS(in_c=in_c, out_c=in_c, k=1, s=1, p=0) 
        
        # 1x1 conv
        # NOTE: Output  :: (n, in_c, h//2, w//2)
        self.conv_1_2 = CBS(in_c=in_c, out_c=in_c, k=1, s=1, p=0)
        
        # 3x3 conv with stride 2
        # NOTE: Output  :: (n, in_c, h//2, w//2)
        self.conv_3_1 = CBS(in_c=in_c, out_c=in_c, k=3, s=2, p=1)
        
        # concatenation
        self.concat = Concat()
        
        
    def forward(self, x):
        
        # branch 1
        # NOTE: Maxpool + 1x1 conv
        out = self.m(x)
        out_1 = self.conv_1_1(out)
        
        # branch 2
        # NOTE : 1x1 conv + 3x3 conv with s 2
        out_2 = self.conv_1_2(x)
        out_3 = self.conv_3_1(out_2)
        
        # self.conv_3_1 + self.conv_1_1
        # NOTE:  Output :: (n, in_c*2, h//2, w//2)
        result = self.concat([out_3, out_1])
        
        return result


class Upsample(nn.Module):
    '''
    Upsampling은 신경망이 작은 물체를 탐지하는 데 중요한 세밀한 features를 학습하는데 도움을 줄 수 있습니다.
    '''
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
    
    def forward(self, x):
        
        return self.upsample(x)

class Route_Backbone(nn.Module):
    '''
    1x1 conv Route Backbone from ELAN Output of Backbone for concatenating in Head Section
        - Input  :: (n, c, h, w)
        - Output :: (n, c//4, h, w)
    '''
    def __init__(self, in_c:int):
        super(Route_Backbone, self).__init__()
        
        # 1x1 conv
        self.route_back = CBS(in_c=in_c, out_c=in_c//4, k=1, s=1, p=0)
        
    def forward(self, x):
        return self.route_back(x)
        
        
class ELAN(nn.Module):
    '''
    ELAN is an efficient network structure which enable the network to learn more features and 
         has stronger robustness by controlling the shortest and longest 'gradient paths'.
        - Two branches
            1. To change the number of channels through a [1 x 1 convolution]
            2-1. First passes through [1 x 1 conv] to change the 'number of channels'
            2-2. Then, go through [four 3 x 3 conv] for feature extraction 
            2-3. Finally the four features are suerimposed togetgher to obtain the final feature extraction result
        - ELAN is composed of multiple CBSs
            - Its input and output feature sizes remain unchanged
            - The number of channels will change in the first two CBSs (1x1 conv)
            - The next few input channels are kept the same as the output channels.
    
    Args
        - Input :: (n, c, h, w)
        - Output :: (n, c*2, h, w)
    8 layers
    '''
    def __init__(self, in_c:int, f=False):
        '''
        f : check whether final elan is 
        '''
        super(ELAN, self).__init__()
        
        # Cross Stage Connection
        # NOTE: in_c = c | out_c = c//2
        self.partialc_t = CBS(in_c=in_c, out_c=in_c//2, k=1, s=1, p=0) # NOTE: 1 x 1 conv
        self.partialc_d = CBS(in_c=in_c, out_c=in_c//2, k=1, s=1, p=0) # NOTE: 1 x 1 conv
        
        # Stack in Computational Block    
        # NOTE: c = c(128)/2
        self.block_1 = CBS(in_c=in_c//2, out_c=in_c//2, k=3, s=1, p=1)
        self.block_2 = CBS(in_c=in_c//2, out_c=in_c//2, k=3, s=1, p=1)
        
        self.block_3 = CBS(in_c=in_c//2, out_c=in_c//2, k=3, s=1, p=1)
        self.block_4 = CBS(in_c=in_c//2, out_c=in_c//2, k=3, s=1, p=1)
        
        # Concatenation
        # NOTE: Concat Output --> (n, in_c*2, h, w) 
        self.concat = Concat() 
        
        # Final CBS
        # NOTE:  1 x 1 conv 
        # - if final elan, output channel is in_c 
        self.final_cbs = CBS(in_c=in_c*2, out_c=in_c*2, k=1, s=1, p=0,g=1) if f is False \
            else CBS(in_c=in_c*2, out_c=in_c, k=1, s=1, p=0,g=1)
     
    def forward(self, x):
        
        # 'C'ross stage connection
        # Partial Transition Layer
        c_out1 = self.partialc_t(x)
        
        # 'C'ross stage connection
        # Partial Dense Layer
        c_out2 = self.partialc_d(x)

        # 'S'tack in computational block
        s_out1 = self.block_1(c_out2)
        s_out2 = self.block_2(s_out1)
        s_out3 = self.block_3(s_out2)
        s_out4 = self.block_4(s_out3)
        
        # concatenation
        # NOTE: self.block4 + sel.block2 + self.partialc2 + self.partialc1
        # NOTE: forward함수의 x에 [s_out4......]를 보내는 것
        out = self.concat([s_out4, s_out2, c_out2, c_out1])
        out = self.final_cbs(out)
        
        return out
 
 
class ELAN_W(nn.Module):
    '''
    Args
        - Input :: (n, c, h, w)
        - Output :: (n, c//2, h, w)
    
    ELAN과의 차이점
        - ELAN은 4개의 block을 concat         // ELAN_W는 6개의 block을 concat
        - ELAN의 Output은 (n, in_c*2, h, w)  //  ELAN_W는  (n, in_c//2, h, w) 
    '''
    def __init__(self, in_c:int):
        super(ELAN_W, self).__init__()
         
        ### Cross Stage Connection ###
        # NOTE: in_c = c | out_c = c//2
        ## Partial Transition Layer
        self.partialc_t = CBS(in_c=in_c, out_c=in_c//2, k=1, s=1, p=0) # NOTE: 1 x 1 conv
        ## Partial Dense Layer
        self.partialc_d = CBS(in_c=in_c, out_c=in_c//2, k=1, s=1, p=0) # NOTE: 1 x 1 conv
        
        # Stack in Computational Block    
        self.block_1 = CBS(in_c=in_c//2, out_c=in_c//4, k=3, s=1, p=1)
        self.block_2 = CBS(in_c=in_c//4, out_c=in_c//4, k=3, s=1, p=1)
        self.block_3 = CBS(in_c=in_c//4, out_c=in_c//4, k=3, s=1, p=1)
        self.block_4 = CBS(in_c=in_c//4, out_c=in_c//4, k=3, s=1, p=1)
        
        # Concatenation
        # NOTE: self.partialc_t + self.partialc_d + self.block1 + self.block2 + self.block3 + self.block4 
        # Concat Output :: (n, sum , h, w)
        self.concat = Concat() 
        
        # Final CBS
        # NOTE:  1 x 1 conv 
        # - if final elan, output channel is in_c 
        self.final_cbs = CBS(in_c=in_c*2, out_c=in_c//2, k=1, s=1, p=0,g=1)
            
    def forward(self, x):
         
        # 'C'ross stage connection
        # Partial Transition Layer
        c_out1 = self.partialc_t(x)
        
        # 'C'ross stage connection
        # Partial Dense Layer
        c_out2 = self.partialc_d(x)

        # 'S'tack in computational block
        s_out1 = self.block_1(c_out2)
        s_out2 = self.block_2(s_out1)
        s_out3 = self.block_3(s_out2)
        s_out4 = self.block_4(s_out3)
        
        # concatenation
        # NOTE: sum = self.partialc_t + self.partialc_d + self.block1 + self.block2 + self.block3 + self.block4
        # NOTE: Concat Output :: (n, sum , h, w)
        out = self.concat([s_out4, s_out3, s_out2, s_out1, c_out2, c_out1])
        out = self.final_cbs(out)
        
        return out
 
 
class SPPCSPC(nn.Module):
    '''
    Combination of [Spatial Pyramid Pooling] and [Cross Stage Partial Network]
    [SPP]
        - to "increase the receptive field", so that the algorithm can adapt to "different resolution images". 
        - It obtains different receptive fields through " ✅ Multi-Scale Max Pooling " .

    [CSP]
        1. ✅ Partial Dense Block
        :: base layer의 feature map을 two part로 분할
    
            - gradient path 증가시킴
                - feature map을 분할하고 병합하는 과정에서 gradient path가 2배가 됩니다. 
                - 이는 concatenation을 위해 feature map을 복사하는 과정에서 발생하는 duplicate gradient information 문제를 완화시킵니다
                
            - 각 layer의 연산량 균형을 이루게 함 
                - Partial Dense Block은 base layer의 feature map이 반으로 분할하므로 각 dense layer가 사용하는 채널 수를 감소시킵니다.
                
            - memory traffic 감소시킴
                - base layer의 feature map이 반으로 분할되므로 연산량도 반으로 감소하게 됩니다.
            
            
        2. ✅ Partial Transition Layer
        :: 분할된 feture map을 병합  (목적은 gradient 조합의 차이를 최대화하는 것입니다)
        
            - gradient flow를 절단하여 각 layer가 duplicate gradient 정보를 학습하는 것을 예방합니다.
            - concatenation 하는 과정에서 gradient information이 복사됩니다.
                - gradient flow를 절단함으로써 많은 양의 gradient information이 재사용 하는 것을 방지합니다
            - split and merge 과정을 통하여 정보 통합 과정동안 duplication의 가능성을 효과적으로 감소할 수 있습니다.
    Args
        - Input :: (n, c, h, w)
        - Output :: (n, c//2, h, w)
    '''
    def __init__(self, in_c:int, pool_size:Tuple):
        '''
        Args
            - pool_size(kernel_size) : (pool_size, pool_size....) --> multiple maxpooling
        '''
        super(SPPCSPC, self).__init__()
        
        ###  Partial Transition Layer  ###
        # NOTE: 1 x 1 conv
        # o = i/2
        self.partial_t = CBS(in_c=in_c, out_c=in_c//2, k=1, s=1, p=0)
        
        
        ###  Partial Dense Layer  ###
        # NOTE: 1 x 1 conv
        # o = i/2
        self.conv1 = CBS(in_c=in_c, out_c=in_c//2, k=1, s=1, p=0)
        
        # NOTE: 3 x 3 conv
        # o = i
        self.conv2 = CBS(in_c=in_c//2, out_c=in_c//2, k=3, s=1, p=1)
        
        # NOTE: 1 x 1 conv
        # o = i
        self.conv3 = CBS(in_c=in_c//2, out_c=in_c//2, k=1, s=1, p=0)
        
        ###  SPP Block  ###
        # NOTE: '3' Maxpooling
        # - kernel size :: 5, 9, 13  /  stirde = 1  / padding = kernel size // 2
        # - 원본 사이즈 유지됨
        '''
        for문으로 레이어들을 하나씩 꺼내서 사용해야 돼서 모듈화 X (forward 에러)
        Output :: List
            - [
                MaxPool2d(kernel_size=n1, stride=1, padding=n1//2, dilation=1, ceil_mode=False),
                MaxPool2d(kernel_size=n2, stride=1, padding=n2//2, dilation=1, ceil_mode=False),
                MaxPool2d(kernel_size=n3, stride=1, padding=n3//2, dilation=1, ceil_mode=False)
                ]
            --> Multi Maxpooling을 진행해도 Tensor Size가 동일하게 유지됨
        '''
        self.spp_m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x//2) for x in pool_size])
        
        # Concatenation
        # NOTE : self.m + self.conv3
        # Output :: (n, in_c*2, h, w)
        self.concat1 = Concat()
        
        # 1 x 1 conv
        # o = i/4
        self.conv4 = CBS(in_c=in_c*2, out_c=in_c//2, k=1, s=1, p=0)

        # 3 x 3 conv
        # o = i
        self.conv5 = CBS(in_c=in_c//2, out_c=in_c//2, k=3, s=1, p=1)
        
        # NOTE : self.conv6 + self.partial_t
        # Output :: (n, in_c, h, w)
        self.concat2 = Concat()        
        
        # Fianl 1x1 CBS
        self.conv6 = CBS(in_c=in_c, out_c=in_c//2, k=1, s=1, p=0)
        
    def forward(self, x):
        '''Two Bracnehs'''
        ## 1. Partial Transition Layer ##
        partial_t = self.partial_t(x)
        
        ## 2. Partial Dense Layer ##
        partial_1  = self.conv1(x)
        partial_2  = self.conv2(partial_1)
        partial_3  = self.conv3(partial_2)
        
        ## SPP
        # Multi Maxpooling
        # Output :: List consisting of 3 layers --> (n, c//2, h, w) *3
        spp_maxp = [spp_m(partial_3) for spp_m in self.spp_m]
        
        # self.conv3 + list (multi maxpooling results)
        concat1 = self.concat1([partial_3] + spp_maxp)

        partial_4 = self.conv4(concat1)
        partial_5 = self.conv5(partial_4)
        
        # self.conv5 + self.partial_t
        concat2 = self.concat2([partial_5, partial_t])
        
        # Final CBS
        partial_final = self.conv6(concat2)
        
        return partial_final
                
                
class RepConv(nn.Module):
    '''
    RepVGG: Making VGG-style ConvNets Great Again
    RepConv Module - Two modules
        ✅ 1. Training Module 
        ::                                    |      FUSION     |  
        :: [3x3 conv] + BN -> 3x3 conv + bias -> | 3x3 conv + bias |
        [1x1 conv] + BN -> 1x1 conv + bias -> | 3x3 conv + bias |  --> 3x3 conv * 1
        [Identity] + BN -> 1x1 conv + bias -> | 3x3 conv + bias |
        :: 3 branches
            - The top branch is a 3x3 convolution for feature extraction.
            - The middle branch is a 1x1 convolution for smoothing features.
            - The last branch is an Identity, which is moved directly without convolution operation.
            - Finally add them together.
        
        ✅ 2. Deploy Module (Inference)
        :: 3x3 conv with stride 1 + BN
        :: is converted from the reparameterization of the training module.
    
    Args
        - Input  :: (n, c,   h, w)
        - Output :: (n, c*2, h, w)
    '''
    def __init__(self, in_c:int, out_c:int, k=3, s=1, g=1, is_deploy=False):
        super(RepConv, self).__init__()
        
        self.act = nn.SiLU()
        self.is_deploy = is_deploy
        
        # training module
        # 1. Identity block (stride =1, in_c == out_c)
        # NOTE: identity는 Batchnorm만 수행하기 때문에 input channel과 output channel이 동일해야 연산이 수행됨.
        self.rep_identity = (nn.BatchNorm2d(num_features=in_c) if out_c == in_c and s == 1 else None)
        
        # 2. 3x3 conv
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=s, padding=1, groups=g, bias=False),
            nn.BatchNorm2d(num_features=out_c),
        )
        
        # 3. 1x1 conv 
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=s, padding=0, groups=g, bias=False),
            nn.BatchNorm2d(num_features=out_c)
        )
        
        # deploy module
        self.deploy_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k,
                                     stride=s, groups=g, bias=True)
        
    def forward(self, x):
        
        # 논문 github에 BN이 없음
        if self.is_deploy:
            self.act(self.deploy_conv)
        
        # NOTE:: in_c와 out_c가 다르면 BN 연산할 때 오류남
        #     :: 그래서 다르면 0으로 초기화하고 같으면 연산에 추가
        identity_result = 0 if self.rep_identity is None else self.rep_identity(x)
        
        # identity + 1x1 conv + 3x3 conv
        return self.act(identity_result + self.conv_1x1(x) + self.conv_3x3(x))
        
        
class IDetect(nn.Module):
    def __init__(self, nc:int, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        '''implicit'''
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        for i in range(self.nl):
            
            x[i] = self.m[i](self.ia[i](x[i]))  # conv + ✨implicit
            x[i] = self.im[i](x[i]) # ✨implicit
            
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        return x 
        

'''YOLOR'''
class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    
    
class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
    

class Backbone(nn.Module):
    '''
    Return : x, bbone_skip1, bbone_skip2
        - bbone_skip1 output :: (n, 512, 80, 80) 
        (Neck1의 2번째 skip connection)
        
        - bbone_skip2 output :: (n, 1024, 40, 40)
        (Neck1의 1번째 skip connection)
    '''
    def __init__(self):
        super(Backbone, self).__init__()
        
        self.cbs1 = CBS(in_c=3, out_c=32, k=3, s=1, p=autopad(k=3))                     # Stage 0           // Output feature map size :: (n x 32 x 640 x 640)
        self.cbs2 = CBS(in_c=32, out_c=64, k=3, s=2, p=autopad(k=3))                    # Stage 1-1         // Output feature map size :: (n x 64 x 320 x 320)
        self.cbs3 = CBS(in_c=64, out_c=64, k=3, s=1, p=autopad(k=3))                    # Stage 1-2         // Output feature map size :: (n x 64 x 320 x 320)
        self.cbs4 = CBS(in_c=64, out_c=128, k=3, s=2, p=autopad(k=3))                   # Stage 2-1         // Output feature map size :: (n x 128 x 160 x 160)
        
        self.elan1 = ELAN(in_c=128)                                                     # Stage 2-2   --11  // Output feature map size :: (n x 256 x 160 x 160)
        self.mp1 = MP1(in_c=256, k=2)                                                   # Stage 3-1   --16  // Output feature map size :: (n x 256 x 80 x 80)

        self.elan2 = ELAN(in_c=256)                                                     # Stage 3-2   --24  // Output feature map size :: (n x 512 x 80 x 80)
        self.mp2 = MP1(in_c=512, k=2)                                                   # Stage 4-1   --29  // Output feature map size :: (n x 512 x 40 x 40)

        self.elan3 = ELAN(in_c=512) # Stage 4-2   --37  // Output feature map size :: (n x 1024 x 40 x 40)
        self.mp3 = MP1(in_c=1024, k=2) # Stage 5-1   --42  // Output feature map size :: (n x 1024 x 20 x 20)

        # Fianl ELAN
        # Same Input C == Output C
        self.elan4 = ELAN(in_c=1024, f=True)                                            # Stage 5-2   --42  // Output feature map size :: (n x 1024 x 20 x 20)
        
    def forward(self, x):
        x = self.cbs4(self.cbs3(self.cbs2(self.cbs1(x))))
        x = self.mp1(self.elan1(x))
        
        # NOTE: For Route Backbone (Neck1에서 concat 활용) // Github Line 24
        # Skip Connection
        # bbone_skip1 :: (n, 512, 80, 80)
        bbone_skip1 = self.elan2(x)
        x = self.mp2(bbone_skip1)
        
        # NOTE: For Route Backbone (Neck1에서 concat 활용) // Github Line 37
        # Skip Connection
        # bbone_skip2 :: (n, 1024, 40, 40)
        bbone_skip2 = self.elan3(x)
        x = self.mp3(bbone_skip2)
        x = self.elan4(x)
        
        return x, bbone_skip1, bbone_skip2
        

class Neck1(nn.Module):
    '''
    FPN TOP-DOWN
        - Concatenation with skip connection layers from backbone
    '''
    def __init__(self):
        super(Neck1, self).__init__()
        
        # SPP + CSP Architecture
        self.sppcspc = SPPCSPC(in_c=1024, pool_size=(5, 9, 13))                         # Stage_ 5     --51  // Output feature map size :: (n x 512 x 20 x 20)
        # 1x1 conv
        self.cbs1 = CBS(in_c=512, out_c=256, k=1, s=1, p=autopad(k=1))                  #              --52  // Output feature map size :: (n x 256 x 20 x 20)
        # Upsampling
        self.upsample = Upsample()                                                      # Stage_ 4-1   --53  // Output feature map size :: (n x 256 x 40 x 40)

        # Route backbone layer (from line 37, bbone_skip2)
        # NOTE: elan42의 Output :: (n, 1024, 40, 40) --> upsampling한 h와 w 동일
        # Output :: (n, in_c//4, h, w)
        self.route1 = Route_Backbone(in_c=1024)                                         #              --54  // Output feature map size :: (n x 256 x 40 x 40)
        
        # Concat
        # NOTE: L53 + L54
        self.concat = Concat()                                                          #              --55  // Output feature map size :: (n x 512 x 40 x 40)

        # ELAN_W
        self.elan_w1 = ELAN_W(in_c=512)                                                 # Stage_ 4-2   --63  // Output feature map size :: (n x 256 x 40 x 40)
        self.cbs2 = CBS(in_c=256, out_c=128, k=1, s=1, p=autopad(k=1))                  #              --64  // Output feature map size :: (n x 128 x 40 x 40)
        '''self.upsamp31_ = Upsample()'''                                               # Stage_ 3-1   --65  // Output feature map size :: (n x 128 x 80 x 80)

        # Route backbone layer (from line 24, bbone_skip1)
        # NOTE: elan32의 Output :: (n, 512, 80, 80) --> upsampling한 h와 w 동일
        # Output :: (n, in_c//4, h, w)
        self.route2 = Route_Backbone(in_c=512)                                          #               --66  // Output feature map size :: (n x 128 x 80 x 80)
        
        # NOTE: L65 + L66
        '''self.concat2_ = Concat()'''                                                  #               --67  // Output feature map size :: (n x 256 x 80 x 80)

        self.elan_w2 = ELAN_W(in_c=256)                                                 # Stage_ 3-2    --75  // Output feature map size :: (n x 128 x 80 x 80)
        
    def forward(self, x, bbone_skip1, bbone_skip2):
        '''
        Args (forward function)
            - bbone_skip1 :: (n, 512, 80, 80) --> Neck1의 2번째 concat
            - bbone_skip2 :: (n, 1024, 40, 40) --> Neck1의 1번째 concat
        
        Return
            - x
            - neck1_skip1 :: (n, 512, 20, 20) --> Neck2의 2번째 concat
            - neck1_skip2 :: (n, 256, 40, 40) --> Neck2의 1번째 concat
        '''
        
        # NOTE: For Neck2에서 concat 활용 // Github Line 51
        # For Skip Connection
        # neck1_skip1 :: (n, 512, 20, 20)
        neck1_skip1 = self.sppcspc(x)
        bbone_skip2 = self.route1(bbone_skip2) # 1x1 conv skip connection layer from backbone 
        
        x = self.upsample(self.cbs1(neck1_skip1))
        x = self.concat([x, bbone_skip2]) # skip concection from backbone
        
        # NOTE: For Neck2에서 concat 활용 // Github Line 63
        # For Skip Connection
        # neck1_skip2 :: (n, 256, 40, 40)
        neck1_skip2 = self.elan_w1(x) 
        bbone_skip1 = self.route2(bbone_skip1)
        
        x = self.upsample(self.cbs2(neck1_skip2))
        x = self.concat([x, bbone_skip1]) # skip concection from backbone
        x = self.elan_w2(x)
        
        return x, neck1_skip1, neck1_skip2
    
    
class Neck2(nn.Module):
    '''
    FPN BOTTOM-UP
        - Concatenation with skip connection layers from neck1
    '''
    def __init__(self):
        super(Neck2, self).__init__()
        
        self.mp2_1 = MP2(in_c=128)                                                      # Stage__ 4-1   --79  // Output feature map size :: (n x 256 x 40 x 40)

        # NOTE: L63 + L79
        self.concat = Concat()                                                          #               --80  // Output feature map size :: (n x 512 x 40 x 40)
        self.elan_w1 = ELAN_W(in_c=512)                                                 # Stage__ 4-2   --88  // Output feature map size :: (n x 256 x 40 x 40)
        self.mp2_2 = MP2(in_c=256)                                                      # Stage__ 5-1   --92  // Output feature map size :: (n x 512 x 20 x 20)

        # NOTE: L51 + L92
        '''self.concat = Concat()'''                                                    #               --93  // Output feature map size :: (n x 1024 x 20 x 20)
        self.elan_w2 = ELAN_W(in_c=1024)                                                # Stage__ 5-2   --101 // Output feature map size :: (n x 512 x 20 x 20)
        
    def forward(self, x, neck1_skip1, neck1_skip2):
        '''
        Args (forward function)
            -  neck1_skip1 :: (n, 512, 20, 20) --> Neck2의 2번째 concat
            -  neck1_skip2 :: (n, 256, 40, 40) --> Neck2의 1번째 concat
        
        Return
            -  neck2_rep1 :: (n, 256, 40, 40) --> For RepConv output :: (n,3,40,40,85) 
            -  neck2_rep2 :: (n, 512, 20, 20) -->  For RepConv output :: (n,3,20,20,85)
        '''
        
        x = self.mp2_1(x)
        x = self.concat([x, neck1_skip2])
        
        neck2_rep1 = self.elan_w1(x) # for RepConv 
        x = self.mp2_2(neck2_rep1)
        
        x = self.concat([x, neck1_skip1])
        neck2_rep2 = self.elan_w2(x) # for RepConv 
    
        return neck2_rep1, neck2_rep2 

class Head(nn.Module):
    '''
    RepConv + YOLOR's IDetect
    '''
    def __init__(self, nc:int):
        super(Head, self).__init__()
        self.anchors = ([12,16, 19,36, 40,28],
                        [36,75, 76,55, 72,146],
                        [142,110, 192,243, 459,401])

        self.nc = nc
        
        self.repconv1 = RepConv(in_c=128, out_c=256)                                    #               --102 (from 75) // Output feature map size :: (n x 256 x 80 x 80)
        # NOTE: Sigmoid --> 중앙 좌표 predicton의 Output 0 ~ 1
        self.cbm1 = CBM(in_c=256, out_c=256, k=1, s=1, p=autopad(k=1)) 
        
        self.repconv2 = RepConv(in_c=256, out_c=512)                                    #               --103 (from 88) // Output feature map size :: (n x 512 x 40 x 40)
        self.cbm2 = CBM(in_c=512, out_c=512, k=1, s=1, p=autopad(k=1)) 
        
        self.repconv3 = RepConv(in_c=512, out_c=1024)                                   #               --104 (from 101) // Output feature map size :: (n x 1024 x 20 x 20)
        self.cbm3 = CBM(in_c=1024, out_c=1024, k=1, s=1, p=autopad(k=1))
        
        self.detector = IDetect(nc=self.nc, anchors=self.anchors, ch=[256, 512, 1024])  #               --105 (from 102,103,104) 
        
    def forward(self, neck1_out, neck2_rep1, neck2_rep2):
        '''
        Args (forward function)
            -  neck1_out  :: (n, 128, 80, 80) --> For RepConv output :: (n,3,80,80,85)
            -  neck2_rep1 :: (n, 256, 40, 40) --> For RepConv output :: (n,3,40,40,85)
            -  neck2_rep1 :: (n, 512, 20, 20) --> For RepConv output :: (n,3,20,20,85)
        
        Return
            -  a list consisting of 3 torch.Tensors
                - [
                    torch.size([1, 3, 80, 80, 85]),
                    torch.Size([1, 3, 40, 40, 85]), 
                    torch.Size([1, 3, 20, 20, 85]), 
                   ]
        '''
        
        '''YOLOv7 원본 github 기준에 의거해서 CBM 모듈 제거 (추후 분석 필요 100% 완료 X)'''
        # rep1, rep2, rep3 = self.cbm1(self.repconv1(neck1_out)), \
                        #    self.cbm2(self.repconv2(neck2_rep1)), \
                        #    self.cbm3(self.repconv3(neck2_rep2))
        
        rep1, rep2, rep3 = self.repconv1(neck1_out), \
                           self.repconv2(neck2_rep1), \
                           self.repconv3(neck2_rep2)
        
        
        return self.detector([rep1, rep2, rep3])