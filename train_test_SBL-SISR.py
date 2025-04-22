import os
import contextlib
import torch.linalg
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import numpy as np
from piqa import SSIM, PSNR
import logging
from typing import Dict, Tuple
import traceback
import sys


def handle_exception(exc_type, exc_value, exc_traceback):
    """全局异常处理函数"""
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
    sys.exit(1)


sys.excepthook = handle_exception

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SRDataset(Dataset):
    """超分辨率数据集类"""

    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 4, patch_size: int = 96, is_train: bool = True):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_img = Image.open(self.hr_paths[idx]).convert('YCbCr')
        lr_img = Image.open(self.lr_paths[idx]).convert('YCbCr')

        hr_y, hr_cb, hr_cr = hr_img.split()
        lr_y, lr_cb, lr_cr = lr_img.split()

        hr_y = self.transform(hr_y)
        lr_y = self.transform(lr_y)

        if self.is_train:
            h, w = lr_y.shape[1:]
            ix = random.randint(0, w - self.patch_size)
            iy = random.randint(0, h - self.patch_size)
            tx = ix * self.scale
            ty = iy * self.scale
            lr_patch = lr_y[:, iy:iy + self.patch_size, ix:ix + self.patch_size]
            hr_patch = hr_y[:, ty:ty + self.patch_size * self.scale, tx:tx + self.patch_size * self.scale]
        else:
            lr_patch = lr_y
            hr_patch = hr_y

        return lr_patch, hr_patch


class KSVD(nn.Module):
    """改进的K-SVD实现"""

    def __init__(self, n_atoms=256, sparsity=5, max_iter=10, tolerance=1e-6):
        super().__init__()
        self.n_atoms = n_atoms
        self.sparsity = sparsity
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.dictionary = nn.Parameter(self._init_dct_dictionary(n_atoms))

    def _init_dct_dictionary(self, n_atoms):
        basis = torch.zeros(n_atoms, n_atoms)
        for k in range(n_atoms):
            basis[k] = torch.cos(torch.arange(n_atoms) * (k * np.pi / n_atoms))
            if k > 0:
                basis[k] -= basis[k].mean()
        return basis / torch.norm(basis, dim=1, keepdim=True)

    def _omp(self, X, D, sparsity):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n_samples, n_features = X.shape
        n_atoms = D.shape[0]
        coefficients = torch.zeros(n_samples, n_atoms, device=X.device)

        for i in range(n_samples):
            residual = X[i].clone()
            indices = []
            for _ in range(min(sparsity, n_atoms)):
                projections = torch.abs(D @ residual.unsqueeze(-1)).squeeze()
                new_idx = torch.argmax(projections)
                if new_idx in indices:
                    break
                indices.append(new_idx)
                selected = D[indices]
                if selected.dim() == 1:
                    selected = selected.unsqueeze(0)
                try:
                    # 修改为使用torch.linalg.lstsq
                    coeff = torch.linalg.lstsq(selected, X[i].unsqueeze(-1)).solution
                except RuntimeError:
                    coeff = torch.pinverse(selected) @ X[i].unsqueeze(-1)
                residual = X[i] - (selected.T @ coeff).squeeze()
                if torch.norm(residual) < 1e-6:
                    break
            coefficients[i, indices] = coeff.squeeze()
        return coefficients

    def update_dictionary(self):
        """空方法，避免调用错误"""
        pass

    def forward(self, X):
        batch_size, channels, h, w = X.shape
        X_flat = X.view(batch_size, channels, -1).permute(0, 2, 1)
        X_flat = X_flat.reshape(-1, channels)
        return self.dictionary


class SparseCoder(nn.Module):
    def __init__(self, algorithm='fista', max_iter=100, tolerance=1e-6, lmbda=0.1):
        super().__init__()
        self.algorithm = algorithm.lower()
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.lmbda = lmbda

    def _soft_threshold(self, x, threshold):
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

    def forward(self, Phi, x, theta=None):
        lmbda = theta if theta is not None else self.lmbda
        batch_size, channels, h, w = x.shape
        n_atoms = Phi.shape[0]

        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)
        A = torch.zeros(batch_size, h * w, n_atoms, device=x.device)
        L = torch.norm(Phi.T @ Phi, 2)

        # 修改t为Tensor类型
        t = torch.tensor(1.0, device=x.device)
        y = A.clone()
        Phi_T = Phi.T

        for _ in range(self.max_iter):
            A_prev = A.clone()
            residual = y @ Phi - x_flat
            grad = y - (1 / L) * (residual @ Phi_T)
            A_new = self._soft_threshold(grad, lmbda / L)

            if self.algorithm == 'fista':
                t_new = (1 + torch.sqrt(torch.tensor(1.0, device=x.device) + 4 * t ** 2)) / 2
                y = A_new + ((t - 1) / t_new) * (A_new - A_prev)
                t = t_new
            else:
                y = A_new

            A = A_new
            if torch.norm(A - A_prev) < self.tolerance:
                break

        return A.permute(0, 2, 1).view(batch_size, n_atoms, h, w)


class MetricCalculator:
    def __init__(self, device: torch.device):
        self.psnr = PSNR().to(device)
        self.ssim = SSIM().to(device)

    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> Dict[str, float]:
        # 确保输入在[0,1]范围内
        sr = torch.clamp(sr, 0, 1)
        hr = torch.clamp(hr, 0, 1)

        # 将单通道图像复制为三通道以适配SSIM计算
        if sr.dim() == 3:  # (C, H, W)
            sr = sr.unsqueeze(0)
            hr = hr.unsqueeze(0)
        sr_rgb = sr.repeat(1, 3, 1, 1) if sr.size(1) == 1 else sr
        hr_rgb = hr.repeat(1, 3, 1, 1) if hr.size(1) == 1 else hr

        return {
            'PSNR': self.psnr(sr, hr).item(),
            'SSIM': self.ssim(sr_rgb, hr_rgb).item()
        }


class SuperResolutionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.scale = config['scale']
        self.ksvd = KSVD(config['n_atoms'], config['sparsity'])
        self.sparse_coder = SparseCoder(algorithm='fista', lmbda=config['fista_lambda'])

        # 修改重建层，添加上采样部分
        self.reconstructor = nn.Sequential(
            nn.Conv2d(config['n_atoms'], 64, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 2倍上采样
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 再2倍上采样，总共4倍
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 添加Sigmoid激活函数确保输出在[0,1]范围内
        )

        self.register_buffer('gamma', torch.ones(config['n_atoms']))
        self.register_buffer('theta', torch.ones(config['n_atoms']))
        self.register_buffer('beta', torch.tensor(1.0))

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        Phi = self.ksvd(lr)
        A = self.sparse_coder(Phi, lr, self.theta)
        sr = self.reconstructor(A)
        return sr


class SRTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['device'])
        self.use_amp = config['use_amp'] and 'cuda' in str(self.device)
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None
        self.autocast = torch.amp.autocast if self.use_amp else contextlib.nullcontext

        self.model = SuperResolutionModel(config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'])
        self.criterion = nn.L1Loss()

        self.train_loader, self.val_loader, self.test_loaders = self._init_data()
        self.metric_calculator = MetricCalculator(self.device)

    def _init_data(self) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
        train_set = SRDataset(
            hr_dir=self.config['div2k_hr'],
            lr_dir=self.config['div2k_lr'],
            scale=self.config['scale'],
            patch_size=self.config['patch_size'],
            is_train=True
        )
        val_set = SRDataset(
            hr_dir=self.config['DIV2K_valid_HR'],
            lr_dir=self.config['DIV2K_valid_LR_bicubic_X4'],
            scale=self.config['scale'],
            is_train=False
        )
        test_sets = {
            'Manga109': SRDataset(
                hr_dir=self.config['Manga109_hr'],
                lr_dir=self.config['Manga109_lr'],
                scale=self.config['scale'],
                is_train=False
            ),
            # 'B100': SRDataset(
            #     hr_dir=self.config['B100_hr'],
            #     lr_dir=self.config['B100_lr'],
            #     scale=self.config['scale'],
            #     is_train=False
            # ),
            # 'Urban100': SRDataset(
            #     hr_dir=self.config['urban100_hr'],
            #     lr_dir=self.config['urban100_lr'],
            #     scale=self.config['scale'],
            #     is_train=False
            # )
        }

        return (
            DataLoader(train_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=4),
            DataLoader(val_set, batch_size=1, shuffle=False),
            {name: DataLoader(ds, batch_size=1) for name, ds in test_sets.items()}
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (lr, hr) in enumerate(self.train_loader):
            try:
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                if lr.numel() > 1e6:
                    continue

                self.optimizer.zero_grad()

                with self.autocast():
                    sr = self.model(lr)
                    # 确保sr和hr尺寸匹配
                    if sr.shape[-2:] != hr.shape[-2:]:
                        hr = torch.nn.functional.interpolate(hr, size=sr.shape[-2:], mode='bilinear',
                                                             align_corners=False)
                    loss = self.criterion(sr, hr)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                if (batch_idx + 1) % 50 == 0 and 'cuda' in str(self.device):
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        metrics = {'PSNR': 0.0, 'SSIM': 0.0}

        for lr, hr in self.val_loader:
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            sr = self.model(lr)
            # 确保验证时尺寸匹配
            if sr.shape[-2:] != hr.shape[-2:]:
                hr = torch.nn.functional.interpolate(hr, size=sr.shape[-2:], mode='bilinear', align_corners=False)
            batch_metrics = self.metric_calculator(sr, hr)

            for k in metrics:
                metrics[k] += batch_metrics[k]

        for k in metrics:
            metrics[k] /= len(self.val_loader)

        return metrics

    @torch.no_grad()
    def test(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, loader in self.test_loaders.items():
            total_metrics = {'PSNR': 0.0, 'SSIM': 0.0}
            for lr, hr in loader:
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = self.model(lr)
                if sr.shape[-2:] != hr.shape[-2:]:
                    hr = torch.nn.functional.interpolate(hr, size=sr.shape[-2:], mode='bilinear', align_corners=False)
                batch_metrics = self.metric_calculator(sr, hr)

                for k in total_metrics:
                    total_metrics[k] += batch_metrics[k]

            for k in total_metrics:
                total_metrics[k] /= len(loader)

            results[name] = total_metrics

        return results

    def run(self):
        best_psnr = 0.0
        torch.backends.cudnn.benchmark = True

        for epoch in range(1, self.config['epochs'] + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()

            logger.info(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | '
                        f'Val PSNR: {val_metrics["PSNR"]:.2f} | Val SSIM: {val_metrics["SSIM"]:.4f}')

            if val_metrics['PSNR'] > best_psnr:
                best_psnr = val_metrics['PSNR']
                torch.save(self.model.state_dict(), 'best_model.pth')
                logger.info('New best model saved!')

        self.model.load_state_dict(torch.load('best_model.pth'))
        test_results = self.test()
        for name, metrics in test_results.items():
            logger.info(f'Test on {name} | PSNR: {metrics["PSNR"]:.2f} | SSIM: {metrics["SSIM"]:.4f}')


if __name__ == '__main__':
    config = {
        'div2k_hr': 'DIV2K/DIV2K_train_HR',
        'div2k_lr': 'DIV2K/DIV2K_train_LR_bicubic/X4',
        'DIV2K_valid_HR': 'DIV2K_valid_HR',
        # 'B100_hr': 'B100/HR',
        # 'B100_lr': 'B100/LR_bicubic/X4',
        # 'urban100_hr': 'Urban100/HR',
        # 'urban100_lr': 'Urban100/LR_bicubic/X4',

        'Manga109_hr': 'Manga109/HR',
        'Manga109_lr': 'Manga109/LR_bicubic/X4',

        'scale': 4,
        'n_atoms': 64,
        'sparsity': 3,
        'patch_size': 32,

        'ksvd_iter': 5,
        'fista_lambda': 0.1,
        'fista_iter': 50,

        'epochs': 50,
        'batch_size': 4,
        'lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_amp': False
    }

    trainer = SRTrainer(config)
    trainer.run()