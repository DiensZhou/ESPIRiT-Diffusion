"""Training and evaluation for score-based generative models. """
import os
import time
import tensorflow as tf
import logging
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Keep the import below for registering all model definitions
from models import ncsnpp, ddpm
import losses
import sampling
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
import scipy.io as sio
from collections import Counter
from tqdm import tqdm
from utils.utils import *
import utils.datasets as datasets

FLAGS = flags.FLAGS


def nmse(recon, label):
    nmse_value = (torch.norm(recon - label) ** 2) / (torch.norm(label) ** 2)
    return nmse_value.double()


def format_recon_for_subject_save(recon):
    return torch.permute(recon[:, 0], [1, 2, 0]).unsqueeze(0).detach().cpu()


def format_label_for_subject_save(label):
    return label[0, 0].unsqueeze(0).detach().cpu()


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """ 
        
    # The directory for saving test results during training
    sample_dir = os.path.join(workdir, "samples_in_train")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    tf.io.gfile.makedirs(checkpoint_dir)
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{config.sampling.ckpt}.pth")
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    print("load weights:", ckpt_path)
    # Resume training when intermediate checkpoints are detected
    initial_step = int(state["step"])

    # Build pytorch dataloader for training
    train_dl = datasets.get_dataset(config, "train")

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
    elif config.training.sde.lower() == "spiritsde":
        sde = sde_lib.SpiritSDE(config)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    zeropadding = config.data.padding
    optimize_fn = losses.optimization_manager(config)
    continuous  = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        config,
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    # In case there are multiple hosts (e.g., ZTPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))
    

    # state["step"] = checkpoint.get("step", 0)
    
    for epoch in range(config.training.epochs):
        loss_sum = 0
        for step, batch in enumerate(train_dl):
            t0 = time.time()
            """make sure that the size of k0, kernel and csm are following:"""
            # k0: (batch_size,coil_map,kx,ky)
            # kernel: (batch_size,coil_map,coil_map,kernel_size,kernel_size)
            # csm: (batch_size,coil_map,kx,ky)
            k0, csm, kernel = batch
            k0  =  k0.to(config.device)
            csm =  csm.to(config.device)
            kernel = kernel.to(config.device)
            k0  =  torch.permute(k0, (0,3,1,2))  # (batch=1,coil,kx,ky)
            if config.training.sde == "vesde":
                csm = torch.permute(torch.squeeze(csm), (3,2,0,1)) # (batch=4,coil,kx,ky)
            else:
                csm = torch.permute(csm, (0,3,1,2))

            if config.training.sde == "vesde" and config.training.csm:
                k0    = k0.repeat(csm.shape[0], 1, 1, 1)  # (coil_map,batch_size,kx,ky)
                label = Emat_xyt_complex(k0, True, csm, 1)
                if zeropadding:
                    label_real = F.pad(label.real, pad=(0, 0, 9, 9, 0, 0), mode='constant', value=0.0)
                    label_imag = F.pad(label.imag, pad=(0, 0, 9, 9, 0, 0), mode='constant', value=0.0)
                    label = torch.complex(label_real, label_imag)
                    csm_real   = F.pad(csm.real, pad=(0, 0, 9, 9, 0, 0), mode='constant', value=0.0)
                    csm_imag   = F.pad(csm.imag, pad=(0, 0, 9, 9, 0, 0), mode='constant', value=0.0)
                    csm   = torch.complex(csm_real, csm_imag)
            else:
                k0 = torch.permute(k0, (1, 0, 2, 3))
                label = Emat_xyt_complex(k0, True, None, 1.0)
                if zeropadding:
                    label_real = F.pad(label.real, pad=(0, 0, 9, 9, 0, 0), mode='constant', value=0.0)
                    label_imag = F.pad(label.imag, pad=(0, 0, 9, 9, 0, 0), mode='constant', value=0.0)
                    label = torch.complex(label_real, label_imag)
                    csm_real   = F.pad(csm.real, pad=(0, 0, 9, 9, 0, 0), mode='constant', value=0.0)
                    csm_imag   = F.pad(csm.imag, pad=(0, 0, 9, 9, 0, 0), mode='constant', value=0.0)
                    csm   = torch.complex(csm_real, csm_imag)

            label = c2r(label).type(torch.FloatTensor).to(config.device)
            label = scaler(label)

            loss = train_step_fn(state, label, kernel, csm)
            loss_sum += loss

            param_num = sum(param.numel() for param in state["model"].parameters())
            if step % 10 == 0:
                print(
                    "Epoch",
                    epoch + 1,
                    "/",
                    config.training.epochs,
                    "Step",
                    step,
                    "loss = ",
                    loss.cpu().data.numpy(),
                    "loss mean =",
                    loss_sum.cpu().data.numpy() / (step + 1),
                    "time",
                    time.time() - t0,
                    "param_num",
                    param_num,
                )

            # Report the loss on an evaluation dataset periodically
            if step % config.training.eval_freq == 0:
                pass

        # Save a checkpoint for every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pth"), state)


def sample(config, workdir):
    """Generate samples.

    Args:
      config: Configuration to use.
      workdir: Working directory.
    """
    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{config.sampling.ckpt}.pth")
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    print("load weights:", ckpt_path)

    SAMPLING_FOLDER_ID = "_".join(
        [
            FLAGS.config.sampling.acc,
            FLAGS.config.sampling.mode,
            FLAGS.config.sampling.center,
            FLAGS.config.sampling.mask_type,
            "ckpt",
            str(config.sampling.ckpt),
            FLAGS.config.sampling.predictor,
            FLAGS.config.sampling.corrector,
            str(config.sampling.snr),
            FLAGS.config.training.sde,
            str(FLAGS.config.model.eta),
            str(FLAGS.config.sampling.mse),
            str(FLAGS.config.sampling.corrector_mse),
        ]
    )
    # Build data pipeline

    test_dl = datasets.get_dataset(config, "test")
    FLAGS.config.sampling.folder = os.path.join(
        FLAGS.workdir, config.training.estimate_csm + "_acc" + SAMPLING_FOLDER_ID
    )
    tf.io.gfile.makedirs(FLAGS.config.sampling.folder)

    # Create data scaler and its inverse
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == "spiritsde":
        sde = sde_lib.SpiritSDE(config)
        sampling_eps = 1e-5  # TODO
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    """Build the sampling function when sampling is enabled, number stands for the number of coil map"""

    if config.training.sde == "vesde" and config.training.csm:
        sampling_shape = (
            1,
            config.data.num_channels,
            config.data.image_size_nx,
            config.data.image_size_ny,
        )
    elif config.sampling.mode == "fastMRI":
        sampling_shape = (
            15,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
    else:  # spirit diffusion
        sampling_shape = (
            18,
            config.data.num_channels,
            config.data.image_size_nx,
            config.data.image_size_ny,
        )
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, inverse_scaler, sampling_eps
    )
  
    if config.sampling.mode != "prospective":
        print("============no prospective mask!!!============")
        pattern_name = getattr(config.sampling, "mask_file", "mask_caipi/302x256caipi_acc8.8_center48.mat")
        if pattern_name.endswith(".mat"):
            mask_stem = pattern_name[:-4]
            mask_path = pattern_name
        else:
            mask_stem = pattern_name
            mask_path = pattern_name + ".mat"
        mask = scio.loadmat(mask_path)
        mask = torch.tensor(mask['mask']).to(config.device)
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        mask_name = os.path.basename(mask_stem)
        sub_str = mask_name + '/'
        print("mask_pattern:", sub_str)

    f    = open(os.path.join(workdir, "snr_results.txt"), "a")
    ssim = StructuralSimilarityIndexMeasure().to(config.device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(config.device)
    recon_res = torch.tensor([]).to(config.device)
    label_res = torch.tensor([]).to(config.device)
    timetime = []
    sample_results = {}
    current_sample = None
    dataset_samples = getattr(getattr(test_dl, "dataset", None), "sample_subjects", [])
    sample_total_counts = Counter(dataset_samples)
    sample_seen_counts = Counter()
    total_slices = len(test_dl.dataset) if hasattr(test_dl, "dataset") else None

    if config.sampling.mode == "retrospective":
        save_dir = getattr(config.sampling, "recon_dir", "./recon_result")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        def save_sample_result(sample_name):
            chunks = sample_results.get(sample_name)
            if not chunks or not chunks["recon"]:
                return
            sample_recon = torch.cat(chunks["recon"], dim=0).numpy()
            sample_label = torch.cat(chunks["label"], dim=0).numpy()
            recon_name = f"{sample_name}_recon.mat"
            recon_path = os.path.join(save_dir, recon_name)
            label_name = f"{sample_name}_label.mat"
            label_path = os.path.join(save_dir, label_name)
            sio.savemat(recon_path, {"recon": sample_recon})
            sio.savemat(label_path, {"label": sample_label})
            print("saved:", recon_path)
            print("saved:", label_path)
            del sample_results[sample_name]
    else:
        save_sample_result = None

    corrector_mse = str(config.sampling.corrector_mse)
    progress_iter = tqdm(test_dl, total=total_slices, desc="reconstruct slices", dynamic_ncols=True)
    for index, point in enumerate(progress_iter):
        if len(point) == 4:
            k0, csm, kernel, subject_batch = point
            sample_name = subject_batch[0] if isinstance(subject_batch, (list, tuple)) else str(subject_batch)
        else:
            k0, csm, kernel = point
            sample_name = f"sample{index:04d}"
        sample_seen_counts[sample_name] += 1
        sample_slice = sample_seen_counts[sample_name]
        sample_total = sample_total_counts.get(sample_name, 0)
        if sample_total:
            progress_iter.set_postfix(sample=sample_name, slice=f"{sample_slice}/{sample_total}")
            print(f"index: {index} | sample: {sample_name} | slice: {sample_slice}/{sample_total}", flush=True)
        else:
            progress_iter.set_postfix(sample=sample_name, slice=sample_slice)
            print(f"index: {index} | sample: {sample_name} | slice: {sample_slice}", flush=True)
        if (
            config.sampling.mode == "retrospective"
            and current_sample is not None
            and sample_name != current_sample
        ):
            save_sample_result(current_sample)
        current_sample = sample_name
        

        k0 = k0.to(config.device)
        kernel = kernel.to(config.device)
        csm = csm.to(config.device)
        
        k0  = torch.permute(k0, (0,3,1,2))  # (batch=1,coil,kx,ky)
        if config.training.sde == "vesde":
            csm = torch.permute(torch.squeeze(csm), (3,2,0,1)) # (batch=4,coil,kx,ky)
        else:
            if len(csm.shape) == 4:
                csm = torch.permute(csm, (0,3,1,2))
            else:
                csm = torch.permute(csm[:,:,:,:,0], (0,3,1,2))

        skip_recon = (not torch.any(k0 != 0).item()) or (not torch.any(csm != 0).item())

        if config.training.estimate_csm == "sos":
            label = Emat_xyt_complex(k0, True, None, 1.0).to(config.device)
            label = sos(label, dim=0).float()
        else:
            k0 = k0.repeat(csm.shape[0], 1, 1, 1)  # (coil_map,batch_size,kx,ky)
            label = Emat_xyt_complex(k0, True, csm, 1).to(config.device)

        if config.sampling.mode != "prospective":
            atb = k0 * mask
        else:
            atb  = k0
            mask = (atb[0:1,0:1,:,:]!=0)


        start = time.time()
        recon_is_complex = skip_recon
        if skip_recon:
            if config.training.sde == "vesde" and config.training.csm and config.sampling.csm == 'sense':
                label = label[0:1,:,:,:]
            recon = torch.zeros_like(label)
            n = 0
            print("kspace or csm is all zero; skip reconstruction and use zero recon.")
        elif config.training.sde == "vesde" and config.training.csm:
            if config.sampling.csm == 'sense':
                #SENSE-diffusion
                atb = atb[0:1,:,:,:]
                csm = csm[0:1,:,:,:]
                recon, n = sampling_fn(score_model, atb, kernel, mask, csm)
                label = label[0:1,:,:,:]
            elif config.sampling.csm == 'espirit':
                #espirit-diffusion
                recon, n = sampling_fn(score_model, atb, kernel, mask, csm)
            recon = r2c(recon)
            recon_is_complex = True
        if (not skip_recon) and config.training.sde == "spiritsde" and config.training.csm:
            atb = torch.permute(atb, (1, 0, 2, 3))
            kernel = torch.permute(kernel, (0, 4, 3, 1, 2))
            # kernel = torch.permute(kernel, (0, 3, 4, 1, 2))  # (coil_map,batch_size,coil_map,kernel_size,kernel_size)
            recon, n = sampling_fn(score_model, atb, kernel, mask, csm)
        label_us = Emat_xyt_complex(atb, True, csm, 1)
        
        end = time.time()
        recon_time = end - start
        timetime.append(recon_time)
        
        print(f"运行时间: {end - start:.3f} 秒")
        if not recon_is_complex:
            recon = r2c(recon)
        
        if config.training.estimate_csm == "sos":
            # recon = fft2c_2d(recon)
            if not skip_recon:
                recon = sos(recon, dim=0)
        else:
            if config.training.sde == "vesde":
                pass
                
            if config.training.sde == "spiritsde":
                recon = fft2c_2d(recon)
                recon = Emat_xyt_complex(recon.permute(1, 0, 2, 3), True, csm, 1.0).to(
                    config.device
                )
               
        print(f"平均耗时: {np.mean(timetime):.3f}s, 最长: {np.max(timetime):.3f}s, 最短: {np.min(timetime):.3f}s")
        if config.sampling.mode != "prospective":
            recon_float = recon.float()
            label_float = label.float()
            SSIM = ssim(recon_float, label_float)
            PSNR = psnr(recon_float, label_float)
            NMSE = nmse(recon_float, label_float)
            print("PSNR:", PSNR, "SSIM:", SSIM, "NMSE:", NMSE)
            print(FLAGS.config.sampling.folder)
            f.write(
                "eta="
                + str(FLAGS.config.model.eta)
                + ", mse="
                + str(FLAGS.config.sampling.mse)
                + ", corrector_mse="
                + str(FLAGS.config.sampling.corrector_mse)
                + ", snr="
                + str(FLAGS.config.sampling.snr)
                + ": PSNR = "
                + str(PSNR)
                + ", SSIM = "
                + str(SSIM)
                + "\n"
            )
        if config.sampling.mode == "retrospective":
            entry = sample_results.setdefault(sample_name, {"recon": [], "label": [], "time": []})
            entry["recon"].append(format_recon_for_subject_save(recon))
            entry["label"].append(format_label_for_subject_save(label))
            entry["time"].append(recon_time)
        else:
            recon_res = torch.cat([recon_res, torch.permute(recon,[2,3,1,0])], dim=2)
            label_res = torch.cat([label_res, torch.permute(label,[2,3,1,0])], dim=2)

    for sample_name in list(sample_results.keys()):
        save_sample_result(sample_name)

    f.write(
        "-----------------------------------------------------------------------------------\n"
    )
    f.close()
