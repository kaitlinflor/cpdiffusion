from collections import OrderedDict
from typing import Tuple, Union, List
from transformers import PreTrainedModel, PretrainedConfig

import timm
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block="bottleneck", layers: list = (3, 4, 23, 3), input_shape=None, output_dim=None, regression=False):
        self.inplanes = 64
        self.input_resolution = input_shape

        super().__init__()

        if block == "bottleneck":
            block = Bottleneck
        elif block == "basic":
            block = BasicBlock
        #self.n_classes = num_classes
        if input_shape is not None:
            channels_in = input_shape
        else:
            channels_in = 3

        self.is_regression = regression
        self.conv1 = nn.Conv2d(channels_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if x.shape[-2:] != (1, 1):
            x = nn.AvgPool2d(x.shape[2:])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()

        self.layers = nn.ModuleList()

        for layer in range(n_layers):
            dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(nn.Sequential(
                               nn.Linear(dim, hidden_dim),
                               nn.BatchNorm1d(hidden_dim),
                               nn.ReLU())
                               )

        self.layers.append(nn.Sequential(
                           nn.Linear(hidden_dim, output_dim))
                           )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        self.initialize_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def initialize_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class PretrainedResNet(nn.Module):
    """docstring for PretrainedResNet."""

    def __init__(self, input_shape, output_dim, adapt=False):
        super().__init__()

        self.adapt = adapt

        if self.adapt:
            input_channels = 3
        else:
            input_channels = input_shape

        self.conv1 = nn.Conv2d(input_shape, input_channels, kernel_size=7, padding=1, bias=False)
        self.pretrained_resnet = timm.create_model('resnet50', pretrained=True, in_chans=input_channels)
        self.pretrained_resnet.fc = nn.Linear(2048, output_dim)

    def forward(self, x: torch.Tensor):
        if self.adapt:
            x = self.conv1(x)
        x = self.pretrained_resnet(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextTransformer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        torch.nn.init.normal_(self.token_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * (
                (2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(
                self.text_projection, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.text_projection.dtype

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


def get_backbone(architecture, **kwargs):
    print("KWARGS : ", kwargs)
    if 'seed' in kwargs.keys():
        torch.manual_seed(kwargs['seed'])
    if architecture == "ResNet-pre":
        print(PretrainedResNet(kwargs['input_channels'], kwargs['embed_dim'], adapt=kwargs['adapt']))
        return PretrainedResNet(
            input_shape=kwargs['input_channels'],
            output_dim=kwargs['embed_dim'],
            adapt=kwargs['adapt'])
    if architecture == 'ResNet':
        return ResNet(
            layers=kwargs['vision_layers'],
            output_dim=kwargs['embed_dim'],
            input_shape=kwargs['input_channels'])
    if architecture == 'MLP':
        return MLP(
            input_dim=kwargs['input_size'],
            n_layers=kwargs['molecule_layers'],
            hidden_dim=kwargs['hidden_dim'],
            output_dim=kwargs['embed_dim'])
    if architecture == 'ModifiedResNet':
        return ModifiedResNet(
            layers=kwargs['vision_layers'],
            output_dim=kwargs['embed_dim'],
            heads=kwargs['vision_width'] * 32 // 64,
            input_resolution=kwargs['image_resolution'],
            width=kwargs['vision_width'])
    elif architecture == 'VisualTransformer':
        return VisualTransformer(
            input_resolution=kwargs['image_resolution'],
            patch_size=kwargs['vision_patch_size'],
            width=kwargs['vision_width'],
            layers=kwargs['vision_layers'],
            heads=kwargs['vision_width'] // 64,
            output_dim=kwargs['embed_dim'])
    elif architecture == 'TextTransformer':
        return TextTransformer(
            embed_dim=kwargs['embed_dim'],
            context_length=kwargs['context_length'],
            vocab_size=kwargs['vocab_size'],
            transformer_width=kwargs['transformer_width'],
            transformer_heads=kwargs['transformer_heads'],
            transformer_layers=kwargs['transformer_layers'])

# class CLIPGeneralConfig(PretrainedConfig):
#     model_type = "clip_general"

#     def __init__(self, hidden_size=512, init_inv_tau=14.3, learnable_inv_tau=True, backbone_architecture=None, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden_size = hidden_size
#         self.init_inv_tau = init_inv_tau
#         self.learnable_inv_tau = learnable_inv_tau
#         self.backbone_architecture = backbone_architecture if backbone_architecture else ['ResNet', 'MLP']


class CLIPGeneralConfig(PretrainedConfig):
    model_type = "clip_general"

    def __init__(self, embed_dim=512, image_resolution=224, vision_layers=None, vision_width=64, vision_patch_size=16,
                 context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12,
                 init_inv_tau=14.3, learnable_inv_tau=True, backbone_architecture=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.vision_layers = vision_layers if vision_layers is not None else [3, 4, 6, 3]
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.init_inv_tau = init_inv_tau
        self.learnable_inv_tau = learnable_inv_tau
        self.backbone_architecture = backbone_architecture if backbone_architecture else ['ResNet', 'MLP']

class CLIPGeneral(PreTrainedModel):
    config_class = CLIPGeneralConfig

    def __init__(self, config: CLIPGeneralConfig):
        super().__init__(config)
        config_dict = config.to_dict()
        print("Config dictionary:", config_dict)  # Add this line for debugging

        self.visual = get_backbone(config.backbone_architecture[0], **config_dict)
        self.transformer = get_backbone(config.backbone_architecture[1], **config_dict)

        # Logit scales for the inner product in the InfoNCE loss
        self.logit_inv_tau = nn.Parameter(torch.ones([]) * np.log(config.init_inv_tau))
        self.logit_inv_tau.requires_grad = config.learnable_inv_tau

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            return self.visual.fc.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    # NEED TO CHANGE IF THIS WORKS !!! 
    def encode_text(self, text):
        return self.transformer(text.type(self.dtype))
        # return self.visual(text.type(self.dtype))


        # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        # x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x).type(self.dtype)

        # # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        # return x

    def forward(self, text, image=None, attention_mask=None):
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, self.logit_inv_tau.exp()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        if config is None:
            config_path = pretrained_model_name_or_path if isinstance(pretrained_model_name_or_path, str) else ""
            config = cls.config_class.from_pretrained(config_path, *model_args, **kwargs)
        model = cls(config)
        state_dict = kwargs.get('state_dict', None)
        if state_dict is None:
            state_dict = torch.load(pretrained_model_name_or_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model


# class CLIPGeneral(PreTrainedModel):
#     config_class = CLIPGeneralConfig

#     def __init__(self, config: CLIPGeneralConfig):
#         super().__init__(config)
#         self.visual = get_backbone(config.backbone_architecture[0], **config.__dict__.get(f"{config.backbone_architecture[0]}-0", {}))
#         self.transformer = get_backbone(config.backbone_architecture[1], **config.__dict__.get(f"{config.backbone_architecture[1]}-1", {}))

#         # Logit scales for the inner product in the InfoNCE loss
#         self.logit_inv_tau = nn.Parameter(torch.ones([]) * np.log(config.init_inv_tau))
#         self.logit_inv_tau.requires_grad = config.learnable_inv_tau

#     @property
#     def dtype(self):
#         try:
#             return self.visual.conv1.weight.dtype
#         except:
#             return self.visual.fc.weight.dtype

#     def encode_image(self, image):
#         return self.visual(image.type(self.dtype))

#     def encode_text(self, text):
#         return self.transformer(text.type(self.dtype))

#     def forward(self, image, text):
#         if image is None:
#             return self.encode_text(text)
#         elif text is None:
#             return self.encode_image(image)
#         image_features = self.encode_image(image)
#         text_features = self.encode_text(text)

#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         return image_features, text_features, self.logit_inv_tau.exp()

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
#         if config is None:
#             config_path = pretrained_model_name_or_path if isinstance(pretrained_model_name_or_path, str) else ""
#             config = cls.config_class.from_pretrained(config_path, *model_args, **kwargs)
#         model = cls(config)
#         state_dict = kwargs.get('state_dict', None)
#         if state_dict is None:
#             state_dict = torch.load(pretrained_model_name_or_path, map_location="cpu")
#         model.load_state_dict(state_dict)
#         return model

# class CLIPGeneral(PreTrainedModel):
#     config_class = CLIPGeneralConfig

#     def __init__(self, config: CLIPGeneralConfig):
#         super().__init__(config)
#         self.text_encoder = nn.Sequential(
#             nn.Linear(config.hidden_size, config.hidden_size),
#             nn.ReLU(),
#             nn.Linear(config.hidden_size, config.hidden_size)
#         )

#     # def __init__(self,
#     #              init_inv_tau: float = 14.3,
#     #              learnable_inv_tau: bool = True,
#     #              backbone_architecture: List[str] = ['ResNet', 'MLP'],
#     #              **kwargs
#     #              ):
#     #     super().__init__()

#     #     self.visual = get_backbone(
#     #         backbone_architecture[0],
#     #         **kwargs.get(f"{backbone_architecture[0]}-0", kwargs))
#     #     self.transformer = get_backbone(
#     #         backbone_architecture[1],
#     #         **kwargs.get(f"{backbone_architecture[1]}-1", kwargs))

#     #     # Logit scales for the inner product in the InfoNCE loss
#     #     self.logit_inv_tau = nn.Parameter(torch.ones([]) * np.log(init_inv_tau))
#     #     self.logit_inv_tau.requires_grad = learnable_inv_tau

#     @property
#     def dtype(self):
#         try:
#             return self.visual.conv1.weight.dtype
#         except:
#             return self.visual.fc.weight.dtype

#     def encode_image(self, image):
#         return self.visual(image.type(self.dtype))

#     def encode_text(self, text):
#         return self.transformer(text.type(self.dtype))

#     def forward(self, image, text):
#         if image is None:
#             return self.encode_text(text)
#         elif text is None:
#             return self.encode_image(image)
#         image_features = self.encode_image(image)
#         text_features = self.encode_text(text)

#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         return image_features, text_features, self.logit_inv_tau.exp()

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
#         if config is None:
#             config_path = pretrained_model_name_or_path if isinstance(pretrained_model_name_or_path, str) else ""
#             config = cls.config_class.from_pretrained(config_path, *model_args, **kwargs)
#         model = cls(config)
#         state_dict = kwargs.get('state_dict', None)
#         if state_dict is None:
#             state_dict = torch.load(pretrained_model_name_or_path, map_location="cpu")
#         model.load_state_dict(state_dict)
#         return model

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 init_inv_tau: float = 14.3,
                 learnable_inv_tau: bool = True
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # Logit scales for the inner product in the InfoNCE loss
        self.logit_inv_tau = nn.Parameter(torch.ones([]) * np.log(init_inv_tau))
        self.logit_inv_tau.requires_grad = learnable_inv_tau

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features, self.logit_inv_tau.exp()


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
