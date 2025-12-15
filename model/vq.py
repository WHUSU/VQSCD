import cv2
import numpy as np
import torch
from torch import nn
from models.vit import ViT
from models.conv import ConvDecoder, UNet
from models.diffusion import Diffusion
from vector_quantize_pytorch import ResidualVQ
from math import sqrt
from loss.perceptual import PerceptualLoss


class VQCT(nn.Module):
    def __init__(self,
         codebook_size=8192, dim_codebook=512, num_quantizers=1, commit_loss_weight=0.1,
         # encoder
         depth=8, dim=512, heads=8, mlp_dim=2048, image_channel=1,
         sample_steps=50
    ):
        super().__init__()

        self.encoder = ViT(image_size=512, patch_size=16, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, out_dim=dim_codebook, channels=image_channel)

        self.quantizer = ResidualVQ(
            dim=dim_codebook,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            shared_codebook=True,
            commitment_weight=1.,
            stochastic_sample_codes=True,
            **dict(
                kmeans_init=True,
                threshold_ema_dead_code=2,
            ),
            **dict(
                quantize_dropout=True,
                quantize_dropout_cutoff_index=1,
                quantize_dropout_multiple_of=1,
            )
        )
        self.commit_loss_weight = commit_loss_weight

        self.cond = ConvDecoder(in_dim=dim_codebook, mid_dim=512, out_dim=image_channel)
        self.decoder = Diffusion(
            UNet(in_channel=2, out_channel=1, inner_channel=32, channel_mults=[1, 2, 4, 8, 16], attn_res=[], res_blocks=1),
            sample_steps=sample_steps
        )
        self.f_recon_loss = nn.MSELoss()
        self.p_loss = PerceptualLoss().eval()

    def forward(self, x, x_):
        b = x.shape[0]
        # y = torch.cat([x, x], dim=0)
        y = x
        #print(y.mean())
        ####################  encode  ##########################
        # 输入是x和从x变换得到的增强数据x'
        feats = self.encoder(torch.cat([x, x_], dim=0))
        # b x d x 32²
        x, x_ = feats[:b], feats[b:]
        f_recon_loss = self.f_recon_loss(x, x_)
        ####################    vq    ##########################
        quantized, codes, commit_loss = self.quantizer(x)
        ####################  decode  ##########################
        vq_dim, f_res = quantized.shape[2], int(sqrt(quantized.shape[1]))
        x = quantized.view(-1, f_res, f_res, vq_dim).permute([0, 3, 1, 2])
        cond = self.cond(x)

        diff, eps = (y - cond[-1]), 1e-3
        recon_loss_pix = torch.mean(torch.sqrt((diff * diff) + (eps * eps))) + 0.5 * torch.mean(self.p_loss(cond[-1], y))
        recon_loss = self.decoder((y+0.8)/2, cond)

        loss = f_recon_loss + recon_loss + self.commit_loss_weight * commit_loss.sum() + recon_loss_pix
        return loss, f_recon_loss, recon_loss, commit_loss


    @torch.no_grad()
    def encode_decode(self, x, to_np_imgs=False, guide=False):
        content_guid = (x+0.7)/2 if guide else None
        b = x.shape[0]
        x = self.encoder(x)

        quantized, codes, commit_loss = self.quantizer(x)
        vq_dim, f_res = quantized.shape[2], int(sqrt(quantized.shape[1]))

        x = quantized.view(b, f_res, f_res, vq_dim).permute([0, 3, 1, 2])
        cond = self.cond(x)
        #x = cond[-1]

        # cond[-1] = cond[-1] * 0.5 + content_guid * 0.5
        #x = self.decoder.sample_ldm(cond, content_guid)
        x = self.decoder.sample_ddim(cond, content_guid, lambda1=0.005)

        if to_np_imgs:
            x = x * 2 - 0.8
            x = x.clamp(-1, 1)
            if x.shape[1] == 1:
                x = x.squeeze(1)
            x = (x + 1) / 2 * 255
            x = x.cpu().numpy().astype(np.uint8)

        return x