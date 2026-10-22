import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.GELU() if relu else nn.Identity()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.GELU() if relu else nn.Identity()
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class DFCAttention(nn.Module):
    def __init__(self, dim, window_size=11):
        super().__init__()
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(window_size, 1), padding=(window_size//2, 0), groups=dim, bias=False)
        self.conv_w = nn.Conv2d(dim, dim, kernel_size=(1, window_size), padding=(0, window_size//2), groups=dim, bias=False)
    def forward(self, x):
        return torch.sigmoid(self.conv_h(self.conv_w(x)))

class CleanGhostV2Block(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0., use_dfc: bool = False, use_grn: bool = True):
        super().__init__()
        hidden_dim = int(dim * 3)
        self.use_dfc = use_dfc
        self.norm = LayerNorm2d(dim)
        self.expand_ghost = GhostModule(dim, hidden_dim, relu=True)
        if self.use_dfc:
            self.dfc = DFCAttention(hidden_dim)
        self.reduce_ghost = GhostModule(hidden_dim, dim, relu=False)
        if self.use_dfc:
            self.se = SEModule(dim)
        self.grn = GRN(dim) if use_grn else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x_expand = self.expand_ghost(x)
        if self.use_dfc:
            x_att = self.dfc(x_expand)
            x = x_expand * x_att
        else:
            x = x_expand
        x = self.reduce_ghost(x)
        if self.use_dfc:
            x = self.se(x)
        x = self.grn(x)
        return shortcut + self.drop_path(x)

class XLLCNet_GhostV2_DA_SE_On_Dim_500k(nn.Module):
    def __init__(self, num_classes: int = 5, drop_path_rate: float = 0.1):
        super().__init__()
        depths = (1, 1, 3, 2)
        dims   = (24, 48, 96, 192)
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4, bias=False),
            LayerNorm2d(dims[0])
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=False)
            ))
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        stage1 = nn.Sequential(*[CleanGhostV2Block(dim=dims[0], drop_path=dp_rates[cur+j], use_dfc=False, use_grn=False) for j in range(depths[0])])
        self.stages.append(stage1)
        cur += depths[0]
        stage2 = nn.Sequential(*[CleanGhostV2Block(dim=dims[1], drop_path=dp_rates[cur+j], use_dfc=False, use_grn=True) for j in range(depths[1])])
        self.stages.append(stage2)
        cur += depths[1]
        stage3 = nn.Sequential(*[CleanGhostV2Block(dim=dims[2], drop_path=dp_rates[cur+j], use_dfc=False, use_grn=True) for j in range(depths[2])])
        self.stages.append(stage3)
        cur += depths[2]
        stage4 = nn.Sequential(*[CleanGhostV2Block(dim=dims[3], drop_path=dp_rates[cur+j], use_dfc=True, use_grn=False) for j in range(depths[3])])
        self.stages.append(stage4)
        self.aux_head1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(dims[1], num_classes))
        self.aux_head2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(dims[2], num_classes))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        x_stage2 = x
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        x_stage3 = x
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        main_out = x.mean(dim=[-2, -1])
        main_out = self.head(self.norm(main_out))
        if self.training:
            aux_out1 = self.aux_head1(x_stage2)
            aux_out2 = self.aux_head2(x_stage3)
            return main_out, aux_out2, aux_out1
        else:
            return main_out

def create_model(num_classes=5, drop_path_rate=0.1):
    return XLLCNet_GhostV2_DA_SE_On_Dim_500k(
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )