import numpy as np
import torch
from torch import nn
from models.sampler import LMS
import os
#device = torch.device("cuda:1")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#device = torch.device("cuda:1")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    def __init__(self,
            # denoise model
            denoise_model=None,
            # noise schedule
            timesteps=1000, linear_start=0.00085, linear_end=0.0120,
            # sampler
            sample_steps=50
        ):
        super(Diffusion, self).__init__()
        self.timesteps = timesteps
        # noise schedule
        self.betas, self.alphas, self.alphas_cumprod, self.alphas_cumprod_prev = None, None, None, None
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = None, None
        self.posterior_variance, self.posterior_log_variance_clipped = None, None
        self.posterior_mean_coef1, self.posterior_mean_coef2 = None, None
        self.sqrt_recip_alphas_cumprod, self.sqrt_recipm1_alphas_cumprod = None, None
        self.register_schedule(timesteps, linear_start, linear_end)
        # denoise model (UNet)
        self.unet = denoise_model
        # loss function
        self.loss_func = nn.MSELoss(reduction="none")
        # self.loss_func = nn.L1Loss(reduction="none")
        # sampler
        self.sampler = LMS(timesteps, linear_start, linear_end)
        self.sampler.set_timesteps(sample_steps)

    def register_schedule(self, timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float32) ** 2
        # print(betas)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas, self.alphas = betas.to(device), alphas.to(device)
        self.alphas_cumprod, self.alphas_cumprod_prev = torch.FloatTensor(alphas_cumprod).to(device), torch.FloatTensor(alphas_cumprod_prev).to(device)
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = torch.sqrt(self.alphas_cumprod), torch.sqrt(1. - self.alphas_cumprod)

        v_posterior = 0
        posterior_variance = (1 - v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + v_posterior * betas
        # print(posterior_variance)
        self.posterior_variance = posterior_variance.float().to(device)
        self.posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20)).float().to(device)
        # print(np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef1 = self.betas * (np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).float().to(device)
        self.posterior_mean_coef2 = (torch.FloatTensor(1. - alphas_cumprod_prev).to(device) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)).float().to(device)

        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod).float().to(device)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1).float().to(device)

    def add_noise_to_x0(self, x0, t, noise):
        batch_size = t.shape[0]
        xt = self.sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1) * x0 + \
             self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1) * noise
        return xt

    def forward(self, x, c):
        # x: 想生成的图像 batch x c x w x w
        # c: 条件        batch x c x w x w
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
        noise = torch.randn_like(x)
        x_noisy = self.add_noise_to_x0(x0=x, t=t, noise=noise)
        eta = self.unet(x_noisy, t, c)

        # cal loss
        target = noise
        loss = self.loss_func(target, eta).mean(dim=[1, 2, 3])
        # loss = self.loss_func(target.view(x.shape[0], -1), eta.view(x.shape[0], -1)).sum(1)
        return loss.mean()


    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = torch.randn_like(x_start)

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    ###############################  DDPM Sample ##################################################
    # def q_posterior(self, x_start, x_t, t):
    #     posterior_mean = (
    #             extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
    #             extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
    #     )
    #     posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
    #     posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
    #     return posterior_mean, posterior_variance, posterior_log_variance_clipped
    #
    # def predict_start_from_noise(self, x_t, t, noise):
    #     return (
    #         extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
    #         extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    #     )
    #
    # def p_mean_variance(self, x, t, y, clip_denoised: bool):
    #     model_out = self.unet(x, t, y)
    #     x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
    #     if clip_denoised:
    #         x_recon.clamp_(-1., 1.)
    #
    #     model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
    #     return model_mean, posterior_variance, posterior_log_variance
    #
    # @torch.no_grad()
    # def p_sample(self, x, t, y, clip_denoised=True):
    #     b, *_, device = *x.shape, x.device
    #     model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, y=y, clip_denoised=clip_denoised)
    #     noise = torch.randn_like(x).to(x.device)
    #     # no noise when t == 0
    #     nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
    #     return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    #
    # @torch.no_grad()
    # def p_sample_loop(self, x, return_intermediates=False):
    #     device = self.betas.device
    #     b = x.shape[0]
    #     img = torch.randn(x.shape, device=device)
    #     intermediates = [img]
    #     for i in reversed(range(0, self.timesteps)):
    #         img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
    #                             x,
    #                             clip_denoised=False)
    #         # if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
    #         #     intermediates.append(img)
    #         print("\rddpm sample: %d / %d" % (self.timesteps-i, self.timesteps), end="")
    #     print()
    #     if return_intermediates:
    #         return img, intermediates
    #     return img[:, 0, :, :]

    def sample_ldm(self, c, content_guid=None):
        b = c[0].shape[0]
        latents = torch.randn(b, 1, 512, 512)  # (1, 4, 64, 64)
        latents = latents * self.sampler.sigmas[0]  # sigmas[0]=157.40723
        latents = latents.to(device)
        # 循环步骤
        for i, t in enumerate(self.sampler.timesteps):  # timesteps=[999.  988.90909091 978.81818182 ...100个

            if content_guid is not None:
            #if content_guid is not None and i < self.sampler.timesteps.shape[0] - 10:
                #lam = 0.1
                if i < self.sampler.timesteps.shape[0] - 10:
                    lam = 0.001
                elif i < self.sampler.timesteps.shape[0] - 5:
                    lam = 1e10
                else:
                    lam = 1e10
                latents = (lam * latents + self.add_noise_to_x0(content_guid, torch.tensor([t] * b).cuda().long(), torch.randn_like(content_guid))) / (1 + lam)

            latent_model_input = latents  # (1, 4, 64, 64)
            sigma = self.sampler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
            timestamp = torch.tensor([t] * b).cuda().float()
            # timestamp = torch.LongTensor([t] * b).to(device)

            c_ = [] + c
            noise_pred = self.unet(latent_model_input, timestamp, c_)
            latents = self.sampler.step(noise_pred, i, latents)


                # latents = (lam * latents + content_guid) / (1 + lam)

            print("\rldm: %d / %d" % (i+1, self.sampler.num_inference_steps), end="")
        print()
        return latents[:, 0, :, :].clamp(-1, 1)

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a


    def sample_ddim(self, c, content, lambda1=0.01):

        sample_inter = 1  # (1 | (self.num_timesteps//10))

        shape = content.shape
        bs = shape[0]
        img = torch.randn(shape, device=device)

        betas = self.betas
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)

        if content is None:
            sampling_timesteps = 40
            skip = self.timesteps // sampling_timesteps
            seq = range(0, self.diffusion.module.timesteps, skip)
            seq = list(seq) + [999]
        else:
            #####原始方案
            seq = list(range(0, 400, 10)) + list(range(400, 1000, 200)) + [999]
            # seq = list(range(0, 800, 20)) + list(range(800, 2000, 200)) + [1999]
            # 线性采样
            # seq = list(range(0, 1000, 40))
            # 平方采样
            # seq = self.square_distribution(24,999)
            ######对数采样
            # seq = self.logarithmic_distribution(24, 999)
            ###正玄
            # seq = self.sinusoidal_distribution(24,999)
            # 指数分布
            # seq = self.custom_distribution(25,999)
            # 指数不确定
            # seq = self.uncertain_custom_distribution(24,999)
            print(seq)

        re_seq = list(reversed(seq))
        # seq = range(0, self.diffusion.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):

            index = re_seq.index(i) + 1
            if index < len(re_seq):
                continuous_sqrt_alpha_cumprod = torch.FloatTensor([sqrt_alphas_cumprod_prev[re_seq[index]]]).repeat(bs, 1).to(device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                # yt
                noisy_t = self.q_sample(
                    content,
                    continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)
                )
            # lam = lambda1 * sqrt_alphas_cumprod[i]
            lam = lambda1 * sqrt_alphas_cumprod[i]
            # junzhi, fangcha
            # out = self.p_sample(img, i)
            t = (torch.ones(bs) * i).to(device)
            next_t = (torch.ones(bs) * (j)).to(device)
            c_ = [] + c
            et = self.unet(img, t, c_)
            at = self.compute_alpha(self.betas, t.long())
            at_next = self.compute_alpha(self.betas, next_t.long())
            x0_t = (img - et * (1 - at).sqrt()) / at.sqrt()
            eta = 1.
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            if content is None:
                img = at_next.sqrt() * x0_t + c1 * torch.randn_like(content) + c2 * et
            else:
                mu = at_next.sqrt() * x0_t + c2 * et
                var = c1

                if i > 0:
                    img = (noisy_t + lam / var * mu) / (1 + lam / var)
                else:
                    img = mu
            print("\rddim: %d / %d" % (i, self.timesteps), end="")
        print()
        return img


if __name__ == '__main__':
    # diffusion_framework = DiffusionFramework()
    # vit = GiFT(input_size=256, patch_size=16, in_channels=2, hidden_size=512, depth=8, num_heads=8, mlp_ratio=4.0)
    # vit.to(device)
    #
    # x = torch.randn(4, 1, 256, 256).to(device)
    # c = torch.randn(4, 1, 256, 256).to(device)
    # t = torch.LongTensor((4, )).to(device)
    #
    # y = vit(x, t, c)
    # print(y.shape)

    pass