# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.model_utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
import sde_lib
from models import model_utils as mutils
from utils.utils import *
from tqdm import trange, tqdm


_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == "ode":
        raise NotImplementedError
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == "pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            config=config,
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            corrector_mse=config.sampling.corrector_mse,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
        )
    elif sampler_name.lower() == "cg":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_mri_CG(
            sde=sde,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            corrector_mse=config.sampling.corrector_mse,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            config=config,
        )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, config, sde, score_fn, probability_flow=False):
        super().__init__()
        self.config = config
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, config, sde, score_fn, snr, corrector_mse, n_steps):
        super().__init__()
        self.config = config
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.corrector_mse = corrector_mse
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, config, sde, score_fn, probability_flow=False):
        super().__init__(config, sde, score_fn, probability_flow)

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        if isinstance(self.sde, sde_lib.SpiritSDE):
            x, x_mean = self.rsde.sde(x, t, atb, kernel, atb_mask, csm)
        else:
            dt = -1.0 / self.rsde.N
            z = torch.randn_like(x)
            drift, diffusion = self.rsde.sde(x, t, atb, atb_mask)
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, config, sde, score_fn, probability_flow=False):
        super().__init__(config, sde, score_fn, probability_flow)

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        if isinstance(self.sde, sde_lib.SpiritSDE):
            x, x_mean = self.rsde.discretize(x, t, atb, kernel, atb_mask, csm)
        else:
            if isinstance(self.sde, sde_lib.VESDE) and self.config.sampling.method == 'cg':
                G1, G2, score = self.rsde.discretize(x, t, atb, csm, atb_mask)
                return G1, G2, score
            else:
                f, G = self.rsde.discretize(x, t, atb, csm, atb_mask)
            z = torch.randn_like(x)
            x_mean = x - f
            x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, config, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        return x, x


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, config, sde, score_fn, snr, corrector_mse, n_steps):
        super().__init__(config, sde, score_fn, snr, corrector_mse, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
            and not isinstance(sde, sde_lib.SpiritSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        corrector_mse = self.corrector_mse

        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            if isinstance(sde, sde_lib.VESDE) and self.config.training.csm:
                x1   = torch.unsqueeze(torch.permute(r2c(x), (1,2,3,0)),0)
                csm1 = torch.unsqueeze(torch.permute(r2c(csm), (1,2,3,0)),0)
                atb1 = atb[0:1,:,:,:]
                meas_grad = Emat_xyt_complex(x1, False, csm1, atb_mask) - atb1
                meas_grad = Emat_xyt_complex(meas_grad, True, csm1, atb_mask)
                meas_grad = c2r(torch.permute(torch.squeeze(meas_grad,0), (3,0,1,2)))
            else:
                meas_grad = Emat_xyt(x, False, None, atb_mask) - c2r(atb)
                meas_grad = Emat_xyt(meas_grad, True, None, atb_mask)
            grad = score_fn(x, t)
            meas_grad /= torch.norm(meas_grad)  #normalization
            meas_grad *= torch.norm(grad)   # ▽||Ax-b|| takes the same norm as score_fn(x, t)
            meas_grad *= corrector_mse   # scale by corrector_mse

            noise = torch.randn_like(x)
            grad_norm  = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size  = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            x_mean = x + step_size[:, None, None, None] * (grad - meas_grad)
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

            x = x.float()
            x_mean = x_mean.float()
        if self.config.sampling.method == 'pc':
            return x, x_mean
        if self.config.sampling.method == 'cg':
            return x, x_mean, grad
            

@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, config, sde, score_fn, snr, corrector_mse, n_steps):
        pass

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        return x, x


def shared_predictor_update_fn(
    x,
    t,
    atb,
    kernel,
    atb_mask,
    csm,
    config,
    sde,
    model,
    predictor,
    probability_flow,
    continuous,
):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(config, sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(config, sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, atb, kernel, atb_mask, csm)


def shared_corrector_update_fn(
    x,
    t,
    atb,
    kernel,
    atb_mask,
    csm,
    config,
    sde,
    model,
    corrector,
    continuous,
    snr,
    corrector_mse,
    n_steps,
):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(
            config, sde, score_fn, snr, corrector_mse, n_steps
        )
    else:
        corrector_obj = corrector(config, sde, score_fn, snr, corrector_mse, n_steps)
    return corrector_obj.update_fn(x, t, atb, kernel, atb_mask, csm)


def get_pc_sampler(
    config,
    sde,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    corrector_mse,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        config=config,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        config=config,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        corrector_mse=corrector_mse,
        n_steps=n_steps,
    )

    def pc_sampler(model, atb, kernel, atb_mask, csm):
        """The PC sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if isinstance(sde, sde_lib.VESDE):
                csm = c2r(csm)
            x = sde.prior_sampling(shape).to(device)  # this function is for ramdom sampling (GWN)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
            # for i in range(sde.N):
            for i in trange(sde.N):
                t = timesteps[i]
                # print('====================', i)
                vec_t = torch.ones(1, device=t.device) * t
                x, x_mean = corrector_update_fn(
                    x, vec_t, atb, kernel, atb_mask, csm, model=model
                )
                x, x_mean = predictor_update_fn(
                    x, vec_t, atb, kernel, atb_mask, csm, model=model
                )
                x = x.float()
                x_mean = x_mean.float()
                # file_name = f"{sde.N-i}.png"
                # import imageio as imgio
                # import numpy as np
                # reverse_step = [0,1,20,50,60,61,70,80,90,98,99]
                # if i in reverse_step:
                #     imgio.imwrite('paper_figure/reverse_process/' + file_name, 
                #             np.uint8(torch.abs(r2c(x_mean[0:1,:,:,:])).squeeze().detach().cpu().numpy()/torch.abs(r2c(x_mean[0:1,:,:,:])).max().item()*255))
                # imgio.imwrite('img_test/recon.png',
                #             np.uint8(torch.abs(r2c(x_mean[0:1,:,:,:])).squeeze().detach().cpu().numpy()/torch.abs(r2c(x_mean[0:1,:,:,:])).max().item()*255))
                # imgio.imwrite('img_test/recon_csm2.png', 
                #             np.uint8(torch.abs(r2c(x_mean[1:2,:,:,:])).squeeze().detach().cpu().numpy()/torch.abs(r2c(x_mean[1:2,:,:,:])).max().item()*255))
                # imgio.imwrite('img_test/recon_csm3.png', 
                #             np.uint8(torch.abs(r2c(x_mean[2:3,:,:,:])).squeeze().detach().cpu().numpy()/torch.abs(r2c(x_mean[2:3,:,:,:])).max().item()*255))
                # imgio.imwrite('img_test/recon_csm4.png', 
                #             np.uint8(torch.abs(r2c(x_mean[3:4,:,:,:])).squeeze().detach().cpu().numpy()/torch.abs(r2c(x_mean[3:4,:,:,:])).max().item()*255))
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)  #normalize the data

    return pc_sampler 
