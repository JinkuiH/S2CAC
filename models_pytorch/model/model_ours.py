from typing import Optional, Union, List
from .decoder_ours import Decoder_3D_2BranchVit
from ..encoders import get_encoder
from ..base import SegmentationHead_3D
from torch import nn
from ..base import initialization as init


class S2CAC(nn.Module):

    def __init__(
        self,
        encoder_name: str = "resnet34_3D",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        temporal_size: int = 1,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=None,
        )
        self.decoder = Decoder_3D_2BranchVit(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center= False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead_3D(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            temporal_size=temporal_size
        )

        self.name = "u-{}".format(encoder_name)
        self.initialize()
    
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output, labels = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.training:
            return masks, labels  # 训练模式返回两个值
        else:
            return masks          # 评估模式返回一个值
