import glob
import os
import torch
import h5py
import bisect
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.io import loadmat
import numpy as np
import math
from utils.utils import *
import torch.nn.functional as F

def crop_specialsize(atb, target_shape=(320, 256)):
    """
    对 atb (1, nc, nx, ny) 做对称裁剪或对称补零，
    使最后两个维度变成 target_shape。
    """
    _, _, nx, ny = atb.shape
    tx, ty = target_shape

    # --- 裁剪或补零 (nx 方向) ---
    if nx > tx:
        # 对称裁剪
        startx = (nx - tx) // 2
        endx = startx + tx
        atb = atb[:, :, startx:endx, :]
    elif nx < tx:
        # 对称补零
        pad_left = (tx - nx) // 2
        pad_right = tx - nx - pad_left
        atb = F.pad(atb, (0, 0, pad_left, pad_right))  # pad=(left,right,top,bottom)

    # --- 裁剪或补零 (ny 方向) ---
    _, _, nx, ny = atb.shape
    if ny > ty:
        starty = (ny - ty) // 2
        endy = starty + ty
        atb = atb[:, :, :, starty:endy]
    elif ny < ty:
        pad_top = (ty - ny) // 2
        pad_bottom = ty - ny - pad_top
        atb = F.pad(atb, (pad_top, pad_bottom, 0, 0))  # pad=(left,right,top,bottom)

    return atb

class ExampleDataSet(Dataset):
    def __init__(self, config):
        super(ExampleDataSet, self).__init__()
        self.config = config
        self.kspace_dir = "example_data/ksp.h5"
        self.maps_dir = "example_data/map_sos.h5"
        self.kernel_dir = "example_data/kernel_7x7_lamda_0p16.h5"

    def __getitem__(self, idx):

        with h5py.File(self.maps_dir, "r") as data:
            maps_idx = data["s_maps"][0]
            maps = np.asarray(maps_idx)

        with h5py.File(self.kernel_dir, "r") as data:
            kernel_idx = data["kernel"][0]
            kernel = np.asarray(kernel_idx)

        with h5py.File(self.kspace_dir, "r") as data:
            ksp_idx = data["kspace"][0]
            if self.config.data.normalize_type == "minmax":
                ksp_idx = torch.from_numpy(ksp_idx)
                maps = torch.from_numpy(maps)
                ksp_idx = torch.unsqueeze(ksp_idx, 0)
                maps = torch.unsqueeze(maps, 0)
                img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
                img_idx = normalize_complex(img_idx)
                ksp_idx = Emat_xyt_complex(img_idx, False, maps, 1)
                ksp_idx = torch.squeeze(ksp_idx, 0)
                maps = torch.squeeze(maps, 0)
            elif self.config.data.normalize_type == "std":
                minv = np.std(ksp_idx)
                ksp_idx = ksp_idx / (self.config.data.normalize_coeff * minv)

            kspace = np.asarray(ksp_idx)

        return kspace, maps, kernel

    def __len__(self):
        # Total number of slices from all scans
        return 1


class ExampleMatDataset(Dataset):
    def __init__(self, config, mat_files=None, return_name=False):
        self.config = config
        self.return_name = return_name
        if mat_files is None:
            mat_files = getattr(config.data, "sample_files", ["example1.mat", "example2.mat"])
        if isinstance(mat_files, str):
            mat_files = [item.strip() for item in mat_files.split(",") if item.strip()]
        self.mat_files = [os.path.abspath(path) for path in mat_files]
        if not self.mat_files:
            raise ValueError("No example .mat files were provided.")

        self.samples = []
        for mat_file in self.mat_files:
            data = loadmat(mat_file)
            for key in ("ksp", "csm", "kernel"):
                if key not in data:
                    raise KeyError(f"{mat_file} must contain '{key}'")
            ksp = data["ksp"]
            csm = data["csm"]
            kernel = data["kernel"]
            if not (ksp.shape[0] == csm.shape[0] == kernel.shape[0]):
                raise ValueError(
                    f"first dim mismatch in {mat_file}: "
                    f"ksp={ksp.shape}, csm={csm.shape}, kernel={kernel.shape}"
                )
            base_name = os.path.splitext(os.path.basename(mat_file))[0]
            for index in range(ksp.shape[0]):
                name = base_name if ksp.shape[0] == 1 else f"{base_name}_{index:03d}"
                self.samples.append((ksp[index], csm[index], kernel[index], name))

        self.sample_subjects = [sample[-1] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ksp, csm, kernel, name = self.samples[idx]
        item = (torch.from_numpy(ksp), torch.from_numpy(csm), torch.from_numpy(kernel))
        if self.return_name:
            return (*item, name)
        return item


class CFLDataset(Dataset):
    def __init__(self, config, root_dir, transform=None, slice_range=(200, 600), return_subject=False, lazy_subject=False):
        """
        初始化CFL数据集读取器
        
        参数:
            root_dir (string): 包含_csm和_ksp文件的根目录
            transform (callable, optional): 可选的数据转换函数
            slice_range (tuple): 切片范围，格式为(start, end)
        """
        self.config = config
        self.root_dirs = sorted(glob.glob(root_dir)) if any(ch in root_dir for ch in "*?[") else [root_dir]
        if not self.root_dirs:
            raise FileNotFoundError(f"No input directories match: {root_dir}")
        self.root_dir = root_dir
        self.transform = transform
        self.slice_start, self.slice_end = self._normalize_slice_range(slice_range)
        self.return_subject = return_subject
        self.sample_subject = getattr(config.data, "sample_subject", "")
        self.lazy_subject = lazy_subject
        self.file_pairs = self._find_file_pairs()

        if self.lazy_subject:
            self._build_lazy_subject_index()
            self.all_ksp_data = np.array([])
            self.all_csm_data = np.array([])
            self.all_kernel_data = np.array([])
            self._cached_subject = None
            self._cached_ksp = None
            self._cached_csm = None
            self._cached_kernel = None
        else:
            # 预加载并拼接所有数据（保持原始设计）
            self.all_ksp_data, self.all_csm_data, self.all_kernel_data, self.sample_subjects = self._concatenate_all_data()
        
        # 计算每个样本的切片大小
        self.slice_size = None if self.slice_end is None else self.slice_end - self.slice_start

    def _normalize_slice_range(self, slice_range):
        if slice_range is None:
            return 0, None
        slice_start, slice_end = slice_range
        slice_start = 0 if slice_start is None else int(slice_start)
        slice_end = None if slice_end is None else int(slice_end)
        if slice_start < 0:
            raise ValueError(f"slice_start must be >= 0, got {slice_start}")
        if slice_end is not None and slice_end < slice_start:
            raise ValueError(f"slice_end must be >= slice_start, got {slice_end} < {slice_start}")
        return slice_start, slice_end

    def _slice_first_dim(self, data):
        return data[self.slice_start:self.slice_end]

    def _readcfl_dims(self, name):
        with open(name + ".hdr", "rt") as h:
            h.readline()
            l = h.readline()
        dims = [int(i) for i in l.split()]
        n = np.prod(dims)
        dims_prod = np.cumprod(dims)
        return dims[:np.searchsorted(dims_prod, n) + 1]

    def _build_lazy_subject_index(self):
        self.subject_infos = []
        self.subject_cumulative_counts = []
        self.sample_subjects = []
        total = 0
        for subject, csm_file, ksp_file, kernel_file in self.file_pairs:
            dims = self._readcfl_dims(ksp_file)
            slice_end = dims[0] if self.slice_end is None else min(self.slice_end, dims[0])
            count = max(0, slice_end - self.slice_start)
            if count == 0:
                continue
            self.subject_infos.append(
                {
                    "subject": subject,
                    "csm_file": csm_file,
                    "ksp_file": ksp_file,
                    "kernel_file": kernel_file,
                    "start": total,
                    "count": count,
                }
            )
            total += count
            self.subject_cumulative_counts.append(total)
            self.sample_subjects.extend([subject] * count)
        print("Lazy subject loading enabled:", [info["subject"] for info in self.subject_infos], flush=True)
        
    def _find_file_pairs(self):
        """查找匹配的_csm和_ksp文件对"""
        file_pairs = []
        for root_dir in self.root_dirs:
            csm_files = [f for f in os.listdir(root_dir) if f.endswith('_csm.cfl')]
            for csm_file in sorted(csm_files):
                base_name = csm_file.replace('_csm.cfl', '')
                if self.sample_subject and base_name != self.sample_subject:
                    continue
                ksp_file = os.path.join(root_dir, f"{base_name}_ksp")
                csm_file1 = os.path.join(root_dir, f"{base_name}_csm")
                kernel_file = os.path.join(root_dir, f"{base_name}_kernel")
                if os.path.exists(ksp_file + '.cfl') and os.path.exists(ksp_file + '.hdr'):
                    file_pairs.append((base_name, csm_file1, ksp_file, kernel_file))
        if self.sample_subject:
            print(f"Subject filter: {self.sample_subject}", flush=True)
        return file_pairs

    def readcfl(self, name):
        # get dims from .hdr
        with open(name + ".hdr", "rt") as h:
            h.readline() # skip
            l = h.readline()
        dims = [int(i) for i in l.split()]

        # remove singleton dimensions from the end
        n = np.prod(dims)
        dims_prod = np.cumprod(dims)
        dims = dims[:np.searchsorted(dims_prod, n)+1]

        # load data and reshape into dims
        with open(name + ".cfl", "rb") as d:
            a = np.fromfile(d, dtype=np.complex64, count=n)
        return a.reshape(dims, order='F') # column-major

    def _load_subject_data(self, subject, csm_file, ksp_file, kernel_file):
        print(csm_file, ksp_file, 'is loading...')
        csm_data = self.readcfl(csm_file)
        ksp_data = self.readcfl(ksp_file)
        if os.path.exists(kernel_file + ".cfl") and os.path.exists(kernel_file + ".hdr"):
            ker_data = self.readcfl(kernel_file)
        else:
            ker_data = csm_data
        ksp_data = np.squeeze(ksp_data)
        if not np.array_equal(csm_data.shape[:4], ksp_data.shape):
            raise ValueError(f"CSM和KSP前四维不匹配: {csm_data.shape[:4]} vs {ksp_data.shape}")

        processed_ksp = self._process_ksp(ksp_data)
        normalize_scope = getattr(self.config.data, "normalize_scope", "subject")
        if normalize_scope == "subject":
            minv = np.std(processed_ksp)
        elif normalize_scope == "slice":
            minv = np.std(processed_ksp, axis=(1, 2, 3), keepdims=True)
        else:
            raise ValueError(f"Unknown normalize_scope: {normalize_scope}")
        minv = np.maximum(minv, 1e-12)
        processed_ksp = processed_ksp / (self.config.data.normalize_coeff * minv)
        processed_csm = self._process_csm(csm_data)
        processed_ker = self._process_csm(ker_data)
        if not (
            processed_ksp.shape[0]
            == processed_csm.shape[0]
            == processed_ker.shape[0]
        ):
            raise ValueError(
                "slice_range produced mismatched first dims: "
                f"ksp={processed_ksp.shape}, csm={processed_csm.shape}, kernel={processed_ker.shape}"
            )
        print('ksp.shape = ', processed_ksp.shape, '     csm.shape = ', processed_csm.shape,
              '     kernel.shape = ', processed_ker.shape)
        return processed_ksp, processed_csm, processed_ker

    def _concatenate_all_data(self):
        """拼接所有KSP和CSM数据(处理维度差异)"""
        all_ksp = []  # 四维数据列表
        all_csm = []  # 五维数据列表
        all_ker = []  # 五维数据列表
        num = 0
        sample_subjects = []
        for subject, csm_file, ksp_file, kernel_file in self.file_pairs:
            processed_ksp, processed_csm, processed_ker = self._load_subject_data(
                subject, csm_file, ksp_file, kernel_file
            )
            
            all_ksp.append(processed_ksp)
            all_csm.append(processed_csm)
            all_ker.append(processed_ker)
            sample_subjects.extend([subject] * processed_ksp.shape[0])
            num += 1
            if False and num > 7:  # no subject limit by default
                break
        
        # 沿第一个维度拼接（KSP四维，CSM五维）
        if all_ksp and all_csm:
            # KSP拼接后形状: (总样本数*slice_len, ...)
            ksp_concat = np.concatenate(all_ksp, axis=0)
            # CSM拼接后形状: (总样本数*slice_len, ..., 4)
            csm_concat = np.concatenate(all_csm, axis=0)
            # kernel拼接后形状: (总样本数*slice_len, ..., 4)
            ker_concat = np.concatenate(all_ker, axis=0)

            print('ksp_data.shape = ', ksp_concat.shape, '     csm_data.shape = ', csm_concat.shape, '     kernel_data.shape = ', ker_concat.shape)
            if self.config.training.sde.lower() == "vesde":
                return ksp_concat, csm_concat, ker_concat, sample_subjects
            elif self.config.training.sde.lower() == "spiritsde":
                return ksp_concat, csm_concat[:,:,:,:,0], ker_concat, sample_subjects
        else:
            return np.array([]), np.array([]), np.array([]), []
    

    def _process_ksp(self, x):
        """Undo FFT/shift along dim 0, then select the reconstruction slice_range."""
        nx, ny, nz, nc = np.shape(x)
        x = np.fft.ifftshift(x, axes=0)
        x = np.transpose(x, [3, 1, 2, 0])
        x = np.fft.ifft(x, axis=-1)
        x = np.transpose(x, [3, 1, 2, 0])
        x = np.fft.fftshift(x, axes=0)*math.sqrt(nx)
        return self._slice_first_dim(x)

    def _process_csm(self, data):
        """Select the same reconstruction slice_range along dim 0."""
        if data.ndim != 5:
            raise ValueError(f"CSM数据应为五维,实际维度: {data.ndim}")
        return self._slice_first_dim(data)

    def __len__(self):
        """返回数据集大小(总样本数*slice_len)"""
        if self.lazy_subject:
            return len(self.sample_subjects)
        return self.all_ksp_data.shape[0] if self.all_ksp_data.size else 0

    def __getitem__(self, idx):
        """获取单个样本(第一个维度作为batch维度)"""
        if idx >= len(self):
            raise IndexError(f"索引超出范围，最大值为{len(self)-1}")

        if self.lazy_subject:
            subject_index = bisect.bisect_right(self.subject_cumulative_counts, idx)
            info = self.subject_infos[subject_index]
            local_idx = idx - info["start"]
            subject = info["subject"]
            if self._cached_subject != subject:
                self._cached_ksp, self._cached_csm, self._cached_kernel = self._load_subject_data(
                    subject,
                    info["csm_file"],
                    info["ksp_file"],
                    info["kernel_file"],
                )
                self._cached_subject = subject
            if self.return_subject:
                return self._cached_ksp[local_idx], self._cached_csm[local_idx], self._cached_kernel[local_idx], subject
            return self._cached_ksp[local_idx], self._cached_csm[local_idx], self._cached_kernel[local_idx]

        # 返回第idx个样本（第一个维度作为batch维度）
        if self.return_subject:
            return self.all_ksp_data[idx], self.all_csm_data[idx], self.all_kernel_data[idx], self.sample_subjects[idx]
        return self.all_ksp_data[idx], self.all_csm_data[idx], self.all_kernel_data[idx]
    

def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)

    if config.data.dataset_name == "ExampleMAT":
        dataset = ExampleMatDataset(
            config,
            mat_files=getattr(config.data, "sample_files", ["example1.mat", "example2.mat"]),
            return_name=True,
        )
        data = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )
    elif config.data.dataset_name == "VWI":
        if mode == "train":
            train_root = getattr(config.data, "train_root_dir", ".")
            train_slice_start = getattr(config.data, "train_slice_start", 0)
            train_slice_end = getattr(config.data, "train_slice_end", 1000)
            dataset = CFLDataset(config, root_dir=train_root, slice_range=(train_slice_start, train_slice_end))
            data = DataLoader(
                dataset,
                batch_size=1,
                num_workers=1,
                shuffle=True,
                pin_memory=True,
            )
        else:
            if config.sampling.mode == "retrospective": 
                sample_root = getattr(config.data, "sample_root_dir", ".")
                sample_slice_start = getattr(config.data, "sample_slice_start", 0)
                sample_slice_end = getattr(config.data, "sample_slice_end", None)
                dataset = CFLDataset(
                    config,
                    root_dir=sample_root,
                    slice_range=(sample_slice_start, sample_slice_end),
                    return_subject=True,
                    lazy_subject=not bool(getattr(config.data, "sample_subject", "")),
                )
                # label = []
                # for batch in dataset:
                #     ksp = batch[0]
                #     csm = batch[1]
                #     csm  = csm.transpose(3,2,0,1)
                #     ksp  = np.expand_dims(ksp, axis = 0)
                #     ksp  = ksp.transpose (0,3,1,2)
                #     ksp  = np.repeat(ksp, csm.shape[0], axis=0)
                #     img  = np.sum(IFFT2c(ksp) * csm, axis=1)
                #     img  = img.transpose(1,2,0)
                #     img  = np.expand_dims(img, axis=0)
                #     label.append(img)
                # label = np.concatenate(label, axis=0)
            elif config.sampling.mode == "prospective": 
                sample_root = getattr(config.data, "sample_root_dir", ".")
                dataset = CFLDataset(config, root_dir=sample_root)
            dataloader_batch_size = 1 if config.sampling.mode == "retrospective" else config.sampling.batch_size
            data = DataLoader( 
                dataset,
                batch_size=dataloader_batch_size,
                shuffle=False,
                pin_memory=True,
            )

    print(mode, "data loaded")

    return data


class MatKspCsmKernelDataset(Dataset):
    """
    用于从 example .mat 读出的 ksp/csm/kernel 构建 Dataset
    默认保持 numpy 的 complex dtype -> torch complex tensor
    """
    def __init__(self, ksp: np.ndarray, csm: np.ndarray, kernel: np.ndarray, to_torch=True):
        assert ksp.shape[0] == csm.shape[0] == kernel.shape[0], \
            f"first dim mismatch: ksp {ksp.shape}, csm {csm.shape}, kernel {kernel.shape}"

        self.ksp = ksp
        self.csm = csm
        self.kernel = kernel
        self.to_torch = to_torch

    def __len__(self):
        return self.ksp.shape[0]

    def __getitem__(self, idx):
        k = self.ksp[idx]
        c = self.csm[idx]
        g = self.kernel[idx]

        if self.to_torch:
            # numpy -> torch（支持 complex64/128）
            k = torch.from_numpy(k)
            c = torch.from_numpy(c)
            g = torch.from_numpy(g)

        return k, c, g


def load_example_mat(mat_path="example1.mat"):
    """
    读取 example .mat，返回 (ksp, csm, kernel)
    """
    d = loadmat(mat_path)
    for key in ("ksp", "csm", "kernel"):
        if key not in d:
            raise KeyError(f"{mat_path} 中缺少变量 '{key}'，实际 keys={list(d.keys())}")

    ksp = d["ksp"]
    csm = d["csm"]
    kernel = d["kernel"]
    return ksp, csm, kernel


def make_dataloader_from_example_mat(config, mat_path="example1.mat", pin_memory=True, num_workers=0):
    """
    从 example .mat 读出数据并构建 DataLoader
    """
    ksp, csm, kernel = load_example_mat(mat_path)

    dataset = MatKspCsmKernelDataset(ksp, csm, kernel, to_torch=True)

    data = DataLoader(
        dataset,
        batch_size=config.sampling.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return data
