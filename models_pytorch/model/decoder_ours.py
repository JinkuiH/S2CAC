import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock_3D(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=(1,2,2), mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock_3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv3dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x1, x2):
        attn_output, _ = self.multihead_attn(x1, x2, x2)
        return attn_output
    
class Decoder_3D_2BranchVit(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        embed_dim = 160

        self.embedding_A = nn.Parameter(torch.full((200, embed_dim), 0.5))
        self.position_embedding = nn.Parameter(torch.randn(200, embed_dim))


        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        
        self.block_len = 4
        if center:
            self.center = CenterBlock_3D(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock_3D(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        #print('Infor:', in_channels, skip_channels, out_channels)
        #Infor: [256, 256, 128, 64] [128, 64, 64, 0] (256, 128, 64, 32)
        last_ch = out_channels[-1]
        conv_flat = [nn.Sequential(nn.Conv3d(in_channels=in_ch, out_channels=embed_dim, kernel_size=1), nn.AdaptiveAvgPool3d((5, 8, 5)), nn.Flatten(start_dim=2))
                   for in_ch in out_channels]
        self.conv_flat = nn.ModuleList(conv_flat)

        self.SAL_first = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)

        self.cross_attention_layers = nn.ModuleList(
            [CrossAttentionLayer(embed_dim, 4) for _ in range(self.block_len)]
        )
        self.self_attention_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4) for _ in range(self.block_len)]
        )

        self.reg_head = nn.Linear(embed_dim, 1, bias=True)
        
    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        A = self.embedding_A.unsqueeze(0).expand(x.shape[0], -1, -1) #3,200,320
        position_emb = self.position_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)

        # reg_branch = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

            x_p = self.conv_flat[i](x).permute(0, 2, 1) + position_emb
            attn_output = self.cross_attention_layers[i](
                x_p.permute(1, 0, 2), A.permute(1, 0, 2)  # (1, n, 320)
            )# 

            A = self.self_attention_layers[i](attn_output)  # 更新A，形状为(n, 320)  #200,3,320
            A = A.permute(1, 0, 2)

        reg_out = self.reg_head(torch.mean(A, dim=1))

        return x,reg_out
