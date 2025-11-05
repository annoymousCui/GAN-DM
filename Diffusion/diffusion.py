from Diffusion.varianceSchedule import VarianceSchedule
from util.networkHelper import *


class DiffusionModel(nn.Module):
    def __init__(self,
                 schedule_name="linear_beta_schedule",
                 timesteps=400,
                 beta_start=0.0001,
                 beta_end=0.02,
                 denoise_model=None):
        super(DiffusionModel, self).__init__()

        self.denoise_model = denoise_model
        self.device = "cuda"

        # 方差生成
        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        self.testTimesteps = 320
        self.betas = variance_schedule_func(timesteps)
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        masks = None

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise_model(x, t, masks) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, img, batch_size, t):
        imgs = []
        print(t)

        for i in tqdm(reversed(range(0, t)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device=self.device, dtype=torch.long), i)
            imgs.append(img.cpu())
        return imgs


    def forward(self, contents, masks, styles, mode):
        contents = contents.to(self.device)
        styles = styles.to(self.device)

        if mode == "train":
            masks = masks.to(self.device)

            contents_noise = torch.randn_like(contents)
            batch_size = contents.shape[0]
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
            contents_x_noisy = self.q_sample(x_start=contents, t=t, noise=contents_noise)  # 加噪
            contents_predicted_noise = self.denoise_model(contents_x_noisy, t, masks=masks)
            pred_out = contents_x_noisy - contents_predicted_noise
            loss = F.smooth_l1_loss(contents_noise, contents_predicted_noise)

            return loss, contents_predicted_noise, contents_noise, contents_x_noisy, pred_out
        else:
            batch_size, channels, image_size = styles.shape[0:3]
            noise = torch.randn_like(styles)
            t = torch.randint(170, 200, (1,), device=self.device).long()
            x_noisy = self.q_sample(x_start=styles, t=t, noise=noise)
            images = self.p_sample_loop(x_noisy, batch_size, t)

            return images






