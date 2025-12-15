import numpy as np
from scipy import integrate


class LMS:
    def __init__(self, num_train_timesteps, beta_start, beta_end):
        # betas = [9.99999975e-05 1.19919918e-04 1.39839845e-04 1.59759758e-04 ...
        self.betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float32) ** 2
        # alphas = #[0.9999     0.9998801  0.99986017 ...
        self.alphas = 1.0 - self.betas
        # alphas_cumprod=累积乘积 [9.99899983e-01 9.99780059e-01 9.99640286e-01 9.99480605e-01 ...
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.num_train_timesteps = num_train_timesteps

    def set_timesteps(self, num_inference_steps=100):
        self.num_inference_steps = num_inference_steps
        # 1000：num_train_timesteps
        #self.timesteps = np.linspace(self.num_train_timesteps - 1, 0, num_inference_steps,
                                      #dtype=float)  # [999.         988.90909091 978.81818182 968.72727273 958.63636364 …… ] 100个
        self.timesteps = np.array(list(reversed(range(0, self.num_train_timesteps, self.num_train_timesteps//num_inference_steps))))
        # print(self.timesteps)
        low_idx = np.floor(self.timesteps).astype(int)  # [999 988 978 968 958  ...] 100个
        high_idx = np.ceil(self.timesteps).astype(int)  # [999 989 979 969 959  ...]  100个
        frac = np.mod(self.timesteps, 1.0)  # [0.         0.90909091 0.81818182 0.72727273 ... ] 小数部分

        sigmas = np.array(
            ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)  # [1.00013297e-02 1.48320440e-02  1000个
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]  # [1.57407227e+02 1.42219348e+02   100个
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float16)  # 最后加个零 101个
        self.derivatives = []

    def get_lms_coefficient(self, order, t, current_order):
        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
            return prod

        integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]

        return integrated_coeff

    def step(self, model_output, timestep, sample):
        order = 4
        sigma = self.sigmas[timestep]
        pred_original_sample = sample - sigma * model_output
        derivative = (sample - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)
        order = min(timestep + 1, order)
        lms_coeffs = [self.get_lms_coefficient(order, timestep, curr_order) for curr_order in range(order)]
        prev_sample = sample + sum(
            coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives)))
        return prev_sample