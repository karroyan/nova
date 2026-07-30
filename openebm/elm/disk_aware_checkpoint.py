"""
磁盘空间感知的 Checkpoint 回调
在保存 checkpoint 前检查磁盘空间；base train 默认可清理旧 checkpoint，
SFT 可通过 cleanup_on_low_space=False 禁止自动清理。
"""
import os
import shutil
from pathlib import Path

try:
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    from pytorch_lightning.callbacks import ModelCheckpoint


class DiskAwareCheckpoint(ModelCheckpoint):
    """带磁盘空间检测的 ModelCheckpoint"""

    def __init__(self, *args, min_free_gb=10, cleanup_on_low_space=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_free_gb = min_free_gb
        self.cleanup_on_low_space = cleanup_on_low_space

    def _check_disk_space(self):
        """检查磁盘剩余空间（GB）"""
        if self.dirpath:
            # 确保目录存在
            Path(self.dirpath).mkdir(parents=True, exist_ok=True)
            stat = shutil.disk_usage(self.dirpath)
            free_gb = stat.free / (1024**3)
            return free_gb
        return float('inf')

    def _cleanup_old_checkpoints(self):
        """强制清理旧 checkpoint，只保留 last.ckpt"""
        if not self.dirpath:
            return

        ckpt_dir = Path(self.dirpath)
        if not ckpt_dir.exists():
            return

        # 找到所有 checkpoint 文件（排除 last.ckpt）
        ckpts = [f for f in ckpt_dir.glob("*.ckpt") if f.name != "last.ckpt"]

        # 删除所有非 last.ckpt 的文件
        for ckpt in ckpts:
            try:
                ckpt.unlink()
                print(f"[DiskAware] 已删除旧 checkpoint: {ckpt.name}")
            except Exception as e:
                print(f"[DiskAware] 删除失败 {ckpt.name}: {e}")

    def _save_checkpoint(self, trainer, filepath):
        """保存前检查磁盘空间"""
        free_gb = self._check_disk_space()

        if free_gb < self.min_free_gb:
            message = (
                f"[DiskAware] checkpoint 保存已停止: 剩余磁盘空间 "
                f"{free_gb:.1f}GB < 保留阈值 {self.min_free_gb:.1f}GB，目录: {self.dirpath}。"
            )
            if not self.cleanup_on_low_space:
                raise RuntimeError(f"{message} 为避免删除 best/top-k checkpoint，未自动清理旧模型。")

            print(f"{message} cleanup_on_low_space=True，开始清理旧 checkpoint...")
            self._cleanup_old_checkpoints()

            # 再次检查
            free_gb = self._check_disk_space()
            print(f"[DiskAware] 清理后剩余空间: {free_gb:.1f}GB")

        # 调用父类保存方法
        super()._save_checkpoint(trainer, filepath)

    def _temporarily_align_completed_for_save(self, trainer):
        """让 checkpoint 看到 completed 已更新后的边界，避免 resume 错一批。"""
        epoch_loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
        batch_progress = getattr(epoch_loop, "batch_progress", None)
        if batch_progress is None:
            return False

        total = getattr(batch_progress, "total", None)
        current = getattr(batch_progress, "current", None)
        if total is None or current is None:
            return False

        should_align = (
            total.processed == total.ready
            and total.completed + 1 == total.processed
            and current.processed == current.ready
            and current.completed + 1 == current.processed
        )
        if not should_align:
            return False

        batch_progress.increment_completed()
        return True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        aligned = self._temporarily_align_completed_for_save(trainer)
        try:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        finally:
            if aligned:
                epoch_loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
                batch_progress = getattr(epoch_loop, "batch_progress", None)
                if batch_progress is not None:
                    batch_progress.total.completed -= 1
                    batch_progress.current.completed -= 1
