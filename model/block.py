import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t
from einops import rearrange
import pywt
import pywt.data
from functools import partial
from torch.nn import init



class MSConv(nn.Module):
    def __init__(self, dim, kernel_sizes=(1, 3, 5)) -> None:
        super(MSConv, self).__init__()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
            for kernel_size in kernel_sizes 
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum([conv(x) for conv in self.dw_convs])

class AdativeFourierNeuralOperator(nn.Module):
    def __init__(self, dim):
        super(AdativeFourierNeuralOperator, self).__init__()
        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()
        self.softshrink = 0.0

    def multiply(self, input, weights):
        return torch.einsum('...bd, bdk->...bk', input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, H, W, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0], inplace=True)
        x_imag = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1], inplace=True)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm='ortho')
        x = x.reshape(B, C, H, W)
        return x
    
class Res2NetBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Res2NetBlock, self).__init__()
 
        width = inplanes // scale
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
 
        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out
    

class MSAFB(nn.Module):
    def __init__(self,
                 dim,
                 norm_layer=nn.BatchNorm2d):
        super(MSAFB, self).__init__()
        self.normlayer1 = norm_layer(dim)
        self.filter = AdativeFourierNeuralOperator(dim)
        self.normlayer2 = norm_layer(dim)
        self.msconv = MSConv(dim)

    def forward(self, x):
        x = x + torch.relu(self.normlayer1(self.filter(x)))
        x = x + torch.relu(self.normlayer2(self.msconv(x)))
        return x


class MSAFB2(nn.Module):
    def __init__(self,
                 dim,
                 norm_layer=nn.BatchNorm2d):
        super(MSAFB, self).__init__()
        self.normlayer1 = norm_layer(dim)
        self.filter = AdativeFourierNeuralOperator(dim)
        self.normlayer2 = norm_layer(dim)
        self.res2netblock = Res2NetBlock(dim, dim)

    def forward(self, x):
        x = x + self.filter(self.normlayer1(x))
        x = self.normlayer2(self.res2netblock(x))
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
  
class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=16):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)
 
		self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
		self.sigmoid = nn.Sigmoid()
 
	def forward(self, x):
		avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		out = avg_out + max_out
		return self.sigmoid(out)


class SGA(nn.Module):
  def __init__(self):
      super().__init__()
      self.sam = SpatialAttention()
      

  def forward(self, x, skip):
      sa = self.sam(skip)
      out = x + sa * x
      return out, skip
  

class SGA2(nn.Module):
  def __init__(self, skip_channels):
      super().__init__()
      kernal_size_dict = {
          1024: 14,
          512: 28,
          256: 56,
          64: 112,
      }
      kernal_size = kernal_size_dict.get(skip_channels)//2 + 1
    #   kernal_size = 5
      self.sam = nn.Sequential(
          nn.Conv2d(in_channels=skip_channels, out_channels=1, kernel_size=kernal_size, padding=kernal_size//2, bias=False),
          nn.Sigmoid(),
      )
      

  def forward(self, x, skip):
      sa = self.sam(skip)
      out = sa * x
      return out, skip
  

class SCGA(nn.Module):
  def __init__(self, in_channels):
      super().__init__()
      self.in_channels = in_channels
      self.sam = SpatialAttention()
      self.cam = ChannelAttention(self.in_channels)
      
      
  def forward(self, x, skip):
      assert x.shape[2] == skip.shape[2] and x.shape[3] == skip.shape[3]
      sa = self.sam(skip)
      assert x.shape[1] == skip.shape[1]
      ca = self.cam(skip)
      scga = sa * ca
      out = x * scga
      return out, skip


class SCSA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
        self.head_num = head_num  # 注意力头数
        self.head_dim = dim // head_num  # 每个头的维度
        self.scaler = self.head_dim ** -0.5  # 缩放因子
        self.group_kernel_sizes = group_kernel_sizes  # 分组卷积核大小
        self.window_size = window_size  # 窗口大小
        self.qkv_bias = qkv_bias  # 是否使用偏置
        self.fuse_bn = fuse_bn  # 是否融合批归一化
        self.down_sample_mode = down_sample_mode  # 下采样模式

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'  # 确保维度可被4整除
        self.group_chans = group_chans = self.dim // 4  # 分组通道数

        # 定义局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # 水平方向的归一化
        self.norm_w = nn.GroupNorm(4, dim)  # 垂直方向的归一化

        self.conv_d = nn.Identity()  # 直接连接
        self.norm = nn.GroupNorm(1, dim)  # 通道归一化
        # 定义查询、键和值的卷积层
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力丢弃层
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()  # 通道注意力门控

        # 根据窗口大小和下采样模式选择下采样函数
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans  # 重组合下采样
                # 维度降低
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 平均池化
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 最大池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入张量 x 的维度为 (B, C, H, W)
        """
        # 计算空间注意力优先级
        b, c, h_, w_ = x.size()  # 获取输入的形状
        # (B, C, H)
        x_h = x.mean(dim=3)  # 沿着宽度维度求平均
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)  # 拆分通道
        # (B, C, W)
        x_w = x.mean(dim=2)  # 沿着高度维度求平均
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)  # 拆分通道

        # 计算水平注意力
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((  
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)  # 调整形状

        # 计算垂直注意力
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((  
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)  # 调整形状

        # 计算最终的注意力加权
        x = x * x_h_attn * x_w_attn

        # 基于自注意力的通道注意力
        # 减少计算量
        y = self.down_func(x)  # 下采样
        y = self.conv_d(y)  # 维度转换
        _, _, h_, w_ = y.size()  # 获取形状

        # 先归一化，然后重塑 -> (B, H, W, C) -> (B, C, H * W)，并生成 q, k 和 v
        y = self.norm(y)  # 归一化
        q = self.q(y)  # 计算查询
        k = self.k(y)  # 计算键
        v = self.v(y)  # 计算值
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # 计算注意力
        attn = q @ k.transpose(-2, -1) * self.scaler  # 点积注意力计算
        attn = self.attn_drop(attn.softmax(dim=-1))  # 应用注意力丢弃
        # (B, head_num, head_dim, N)
        attn = attn @ v  # 加权值
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)  # 求平均
        attn = self.ca_gate(attn)  # 应用通道注意力门控
        return attn * x  # 返回加权后的输入
    

class SCGA33(nn.Module):
  def __init__(self, in_channels, size):
      super().__init__()
      self.sa_conv = nn.Sequential(
          nn.Conv2d(in_channels, 1, 1, bias=False),
          nn.Sigmoid()
      )
      self.sc_conv = nn.Sequential(
          nn.Linear(size * size, 1, bias=False),
          nn.Sigmoid()
      )

      self.final_conv = nn.Sequential(
          nn.Conv2d(in_channels*2, in_channels, 1),
          nn.BatchNorm2d(in_channels),
          nn.ReLU()
      )

  def forward(self, x, skip):
      B, C, H, W = skip.shape
      identify = x
      sa = self.sa_conv(skip)
      new_skip = skip.reshape(B, C, -1)
      sc = self.sc_conv(new_skip)
      sc = sc.reshape(B, C, 1, 1)
      x_sa = x * sa
      x_sc = x * sc
      x_sac = torch.cat((x_sa, x_sc), dim=1)
      out = self.final_conv(x_sac)
      return out
  

class SCGA34(nn.Module):
  def __init__(self, in_channels, size):
      super().__init__()
      self.sa_conv = nn.Sequential(
          nn.Conv2d(in_channels, 1, 1, bias=False),
          nn.Sigmoid()
      )
      self.sc_conv = nn.Sequential(
          nn.Linear(size * size, 1, bias=False),
          nn.Sigmoid()
      )

  def forward(self, x, skip):
      B, C, H, W = skip.shape
      identify = x
      sa = self.sa_conv(skip)
      new_skip = skip.reshape(B, C, -1)
      sc = self.sc_conv(new_skip)
      sc = sc.reshape(B, C, 1, 1)
      x_sa = x * sa
      x_sc = x * sc
      x_sac = x_sa + x_sc
      return x_sac


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv2d, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

        # 逐点卷积：使用 1x1 卷积
        # self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        # x = self.pointwise(x)
        return x


class MSWTB(nn.Module):
    def __init__(self, dim, kernel_sizes=(1, 3, 5)) -> None:
        super(MSWTB, self).__init__()
        self.wtconvs = nn.ModuleList([
            DepthwiseSeparableConvWithWTConv2d(dim, dim, kernel_size) for kernel_size in kernel_sizes
        ])
        self.normal = nn.BatchNorm2d(dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(self.normal(sum([conv(x) for conv in self.wtconvs])))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (1, 3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        # x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(max_out)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30
    
class DWConv2d(nn.Module):
    def __init__(self,in_ch,out_ch, kernel_size=3):
        super(DWConv2d, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=kernel_size//2,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
    

class DWConv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        conv = DWConv2d(
            in_channels,
            out_channels,
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(DWConv2dReLU, self).__init__(conv, bn, relu)


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out
    

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], dim=1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


if __name__ == "__main__":
    mswtb = MSWTB(64)
    input = torch.randn((1, 64, 112, 112))
    out = mswtb(input)

