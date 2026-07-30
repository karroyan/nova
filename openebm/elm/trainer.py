import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from torch.utils.data import random_split, Dataset
# from torchvision import transforms
# from torchvision.transforms import ToPILImage
from torch.distributed import all_reduce
import wandb
import gc

# from data.vid.ucf_dataloader import *
# from data.vid.kinetics_dataloader import *
# from data.img.imagenet_dataloader import *
# from data.img.coco_tiny_dataset import COCOTinyDataset
# from data.img.coco_medium_dataset import COCOMediumDataset
# from data.vid.aggregate_dataloader import *
# from data.vid.vid_synthetic_dataset import VIDSyntheticDataset
# from data.nlp.pajama_dataloader import RedPajamaDataset
# from data.nlp.fineweb_dataloader import FineWebDataset
# from data.nlp.collator import NLP_HF_Collator
# from data.nlp.bigbench_dataloader import BigBenchDataset
# from data.nlp.gsm8k_dataloader import GSM8KDataset
# from data.nlp.lambada_dataset import LambadaDataset
# from data.nlp.squad_dataloader import SQuADDataset
# from data.nlp.ai2arc_dataloader import AI2ArcDataset
# from data.nlp.planbench_dataloader import PlanBenchDataset
# from data.nlp.synthetic_dataset import NLPSyntheticDataset

from openebm.elm.collector import NLP_HF_Collator
from datasets import load_dataset, load_from_disk
import os
# from model.vid.ebt import EBT_VID
from openebm.elm.modeling_ebt import EBT_NLP
# from model.img.ebt_t2i import EBT_IMG_T2I
# from model.img.ebt_denoise import EBT_IMG_Denoise

# from model.vid.baseline_transformer import Baseline_Transformer_VID
# from model.nlp.baseline_transformer import Baseline_Transformer_NLP

# from model.img.dit_t2i import Diffusion_Transformer_IMG_T2I
# from model.img.dit_denoise import Diffusion_Transformer_IMG_Denoise


# from nanolightning.torchlightning_module import LightningModule
# from nanolightning.iteratabledataset import generate_dataloader, IterableDataset

try:
    from lightning.pytorch import LightningModule
except ImportError:
    from pytorch_lightning import LightningModule
from openebm.elm.dataset import IterableDataset, generate_dataloader
from openebm.elm.dataset_sft import generate_sft_dataloader


# Simple GSM8K Dataset class for inference
class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, split):
        self.hparams = hparams
        local_dataset_path = "/mnt/shared-storage-user/puyuan/code/EBT/data/gsm8k_offline"

        if os.path.exists(local_dataset_path):
            dataset = load_from_disk(local_dataset_path)
            self.dataset = dataset[split]
        else:
            hf_token = os.getenv('HF_TOKEN')
            hf_home = os.getenv('HF_HOME')
            dataset_dir = self.hparams.dataset_dir if hasattr(self.hparams, 'dataset_dir') and self.hparams.dataset_dir != "" else hf_home
            self.dataset = load_dataset("openai/gsm8k", "main", cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.hparams.execution_mode == "inference":
            return f"[[Question]]: {self.dataset[idx]['question']}\n[[Answer]]: ", self.dataset[idx]['answer']
        elif self.hparams.execution_mode == "pretrain":
            return f"Question: {self.dataset[idx]['question']}\nAnswer: {self.dataset[idx]['answer']}"
        elif self.hparams.execution_mode == "finetune":
            return f"[[Question]]: {self.dataset[idx]['question']}\n[[Answer]]: {self.dataset[idx]['answer']}"
        else:
            raise ValueError(f"Execution mode not supported: {self.hparams.execution_mode}")

# from utils import save_frames, denormalize, load_image_encoder, center_crop_arr
from openebm.elm.generate import generate_text, get_ppl
# from inference.vid.generate_video import generate_video
# from inference.img.generate_image import generate_image
from openebm.elm.optimization import WarmUpCosineAnnealingLR, WarmUpLinearWarmdownLR, LARS, exclude_bias_and_norm, StableAdamW, StableAdamWUnfused
from openebm.elm import logger as text_logger
from openebm.elm.metrics import get_torchmetrics
import sys
from transformers import AutoTokenizer

import ipdb
import sys


def _sft_trainer_debug(message):
    if os.environ.get("EBT_SFT_DEBUG", "0").lower() not in ("1", "true", "yes"):
        return
    rank = os.environ.get("RANK", "?")
    debug_ranks = os.environ.get("EBT_SFT_DEBUG_RANKS", "0").strip().lower()
    if debug_ranks not in ("all", "*"):
        enabled_ranks = {item.strip() for item in debug_ranks.split(",") if item.strip()}
        if rank not in enabled_ranks:
            return
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[EBT SFT Trainer][rank={rank} local={local_rank}] {message}", flush=True)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from nanochat.tokenizer import get_tokenizer, get_token_bytes


# ── v3 utilities for Sudoku adaptive-ratio + difficulty schedule ──────────────

# Three-phase schedule (P3 + P4) used in the v3 spec:
#   step <  600                        → 0.6   (warmup)
#   600 ≤ step < 2400                  → 0.85  (focus)
#   step ≥ 2400                        → 0.7   (consolidation)
# The retention guard then clips this around `valid_loss_sft` drift on each val tick.
_V3_RATIO_PHASES = [
    {'lo': 0,    'hi': 600,   'ratio': 0.6},
    {'lo': 600,  'hi': 2400,  'ratio': 0.85},
    {'lo': 2400, 'hi': 10**9, 'ratio': 0.7},
]
# Difficulty bucket weights aligned with SudokuSFTV2IterableDataset.DIFFICULTY_BUCKETS
# (hard 17-22, medium 23-28, easy 29-34). P5 in the v3 spec.
_V3_DIFFICULTY_PHASES = [
    {'lo': 0,    'hi': 600,   'weights': [0.33, 0.33, 0.34]},   # warmup uniform
    {'lo': 600,  'hi': 2400,  'weights': [0.55, 0.30, 0.15]},   # focus on hard tail
    {'lo': 2400, 'hi': 10**9, 'weights': [0.40, 0.35, 0.25]},   # consolidation
]
# Retention guard (P4): SFT-loss target used to keep core SFT capability from drifting.
# v2 baseline `valid_loss_sft` was 1.121 across the 0.4/0.6/0.8 sweep; we accept up to
# +0.03 over baseline before cutting `sudoku_ratio` by 0.05 (floor 0.50).
_V3_SFT_BASELINE = 1.121
_V3_SFT_TOLERANCE = 0.03
_V3_RATIO_FLOOR = 0.50
_V3_RATIO_CEIL = 0.85
_V3_RATIO_STEP = 0.05


def _v3_three_phase_value(step, phases, default):
    for p in phases:
        if p['lo'] <= step < p['hi']:
            return p['ratio'] if 'ratio' in p else p['weights']
    return default


def _v3_initial_bucket_weights(schedule_name):
    """Return the warmup-phase bucket weights for the given schedule, or None for `fixed`.

    `fixed`        → None (legacy uniform sampling over the whole pool).
    `three_phase`  → returns the warmup-phase weights; AdaptiveRatioCallback later
                     rotates buckets per phase.
    Anything else  → None (treated as `fixed`).
    """
    if schedule_name == 'three_phase':
        return list(_V3_DIFFICULTY_PHASES[0]['weights'])
    return None


class AdaptiveRatioCallback:
    """Lightning callback wiring v3 sudoku schedules + retention guard (P3 + P4 + P5).

    Three responsibilities:
      1. Mutates `train_dataloader.dataset.sudoku_ratio` per `_V3_RATIO_PHASES`
         on every training-batch start.
      2. Mutates `train_dataloader.dataset.sudoku_ds.bucket_weights` per
         `_V3_DIFFICULTY_PHASES` so RRN-train resampling tracks the schedule.
      3. After each validation epoch, reads `valid_loss_sft` (already all-reduced)
         and adjusts `sudoku_ratio` ± _V3_RATIO_STEP within [floor, ceil] based on
         the SFT drift — primary core-preservation mechanism in v3 since KD is
         deferred.

    Determinism: each rank runs the same callback against the same `global_step`,
    so DDP ranks stay in sync without explicit broadcast. The retention guard
    keys off the epoch-level `valid_loss_sft` from Lightning callback metrics,
    falling back to the module's cached validation metrics.
    """

    def __init__(
        self,
        ratio_schedule='three_phase_with_guard',
        difficulty_schedule='three_phase',
    ):
        try:
            from lightning.pytorch.callbacks import Callback as _LCB
        except ImportError:
            from pytorch_lightning.callbacks import Callback as _LCB
        # Subclass at construction time so we don't import Callback at module load
        # (the Lightning version is already established by the trainer module).
        AdaptiveRatioCallback._mixin_callback(self, _LCB)
        self.ratio_schedule = ratio_schedule
        self.difficulty_schedule = difficulty_schedule
        self._last_logged_step = -1

    @staticmethod
    def _mixin_callback(instance, callback_cls):
        # Ensure isinstance(instance, Callback) is true so Lightning accepts it.
        instance.__class__ = type(
            'AdaptiveRatioCallback', (AdaptiveRatioCallback, callback_cls), {},
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _get_mixed_dataset(self, trainer):
        train_dl = getattr(trainer, 'train_dataloader', None)
        if train_dl is None:
            return None
        ds = getattr(train_dl, 'dataset', None)
        # Only act on SudokuMixedIterableDataset (guard against other datasets).
        if ds is None or not hasattr(ds, 'sudoku_ratio') or not hasattr(ds, 'sudoku_ds'):
            return None
        return ds

    def _phase_ratio(self, step):
        return _v3_three_phase_value(step, _V3_RATIO_PHASES, default=0.6)

    def _phase_bucket_weights(self, step):
        return list(_v3_three_phase_value(step, _V3_DIFFICULTY_PHASES, default=[1, 1, 1]))

    # ── Lightning hooks ────────────────────────────────────────────────

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):  # type: ignore[override]
        ds = self._get_mixed_dataset(trainer)
        if ds is None:
            return
        step = int(trainer.global_step)
        # 1. base ratio from the three-phase schedule
        if self.ratio_schedule in ('three_phase', 'three_phase_with_guard'):
            base = self._phase_ratio(step)
            # Don't overwrite a guard-adjusted ratio mid-phase: only sync the *floor*
            # of the schedule. We apply phase-on-transition edges; guard nudges live
            # in on_validation_epoch_end.
            current = float(ds.sudoku_ratio)
            phase_lo = max(_V3_RATIO_FLOOR, base - 0.10)
            phase_hi = min(_V3_RATIO_CEIL, base + 0.05)
            if not (phase_lo <= current <= phase_hi):
                ds.sudoku_ratio = float(base)
        elif self.ratio_schedule == 'fixed':
            pass  # leave alone
        # 2. difficulty bucket weights
        if self.difficulty_schedule == 'three_phase':
            sudoku_ds = getattr(ds, 'sudoku_ds', None)
            if sudoku_ds is not None and hasattr(sudoku_ds, 'bucket_weights'):
                sudoku_ds.bucket_weights = self._phase_bucket_weights(step)
        # 3. periodic log so the run.log shows the active schedule
        if step != self._last_logged_step and (step % 100 == 0):
            try:
                pl_module.log_metrics(
                    {
                        'sudoku_ratio_active': float(ds.sudoku_ratio),
                    },
                    'train',
                )
            except Exception:
                pass
            self._last_logged_step = step

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if self.ratio_schedule != 'three_phase_with_guard':
            return
        ds = self._get_mixed_dataset(trainer)
        if ds is None:
            return
        sft_loss = None
        callback_metrics = getattr(trainer, 'callback_metrics', None)
        if callback_metrics is not None:
            try:
                sft_loss = callback_metrics.get('valid_loss_sft')
            except AttributeError:
                pass
        if sft_loss is None:
            last_metrics = getattr(pl_module, '_last_valid_metrics', None)
            if not last_metrics:
                return
            sft_loss = last_metrics.get('valid_loss_sft')
        if sft_loss is None:
            return
        if isinstance(sft_loss, torch.Tensor):
            sft_loss = sft_loss.detach().float().item()
        delta = float(sft_loss) - _V3_SFT_BASELINE
        before = float(ds.sudoku_ratio)
        if delta > _V3_SFT_TOLERANCE:
            new_r = max(_V3_RATIO_FLOOR, before - _V3_RATIO_STEP)
        elif delta < 0.0:
            new_r = min(_V3_RATIO_CEIL, before + _V3_RATIO_STEP)
        else:
            new_r = before
        if abs(new_r - before) > 1e-6:
            ds.sudoku_ratio = float(new_r)
            try:
                pl_module.log_metrics({'sudoku_ratio_guard_adjusted': float(new_r)}, 'valid')
            except Exception:
                pass
            print(f"[AdaptiveRatioCallback] valid_loss_sft={sft_loss:.4f} "
                  f"(Δ={delta:+.4f}) → sudoku_ratio {before:.2f} → {new_r:.2f}")


class ModelTrainer(LightningModule):
    def __init__(self, hparams, trained_model = None):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        # self.txt_logger = hparams.txt_logger if txt_logger == None else txt_logger # txt_logger is no longer supported

        # Initialize tracking for test metrics
        self.test_losses = []
        self.test_perplexities = []

        # Training throughput tracking
        self._train_step_start_time = None
        self._train_start_time = None  # wall-clock start for ETA

        # Dataloader resume state: 用于从 checkpoint 恢复 dataloader 位置
        self._dataloader_resume_state = None

        if self.hparams.modality == "NLP":
            if "execution_mode" in self.hparams and "save_generation_logs_dir" in self.hparams and self.hparams.execution_mode == "inference": # two of these are sanity check for loading pretrained ckpt that may not have newer params
                print("setting up infer logger")
                self.infer_logger = text_logger.setup_jsonl_logger(log_filename = "results.jsonl", base_log_dir=self.hparams.save_generation_logs_dir)
        # if self.hparams.modality == "VID": #is computer vision
        #     self.image_dims = self.hparams.image_dims # list size two
        #     self.num_generated_videos = 0
        #     if self.hparams.custom_image_normalization:
        #         self.transform = transforms.Compose([
        #             transforms.Resize((self.image_dims[0], self.image_dims[1])),
        #             transforms.ToTensor(),
        #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #         ])

        #         normal_lookup = { #NOTE is std, mean
        #             "ucf101": ([1.04731617, 1.04372056, 1.02795228], [-0.40689788, -0.36098219, -0.25687788]),
        #             "k400": ([1.00370078, 0.99871626, 0.97407404], [-0.24295556, -0.24931058, -0.13959686]),
        #             "smth": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
        #             "ImageNet": ([1, 1, 1], [0, 0, 0])
        #         }

        #         normal_lookup["something"] = normal_lookup["smth"]
        #         normal_lookup["ImageNet1k"] = normal_lookup["ImageNet"]
        #         self.normal_lookup = normal_lookup

        #         if self.hparams.dataset_name in normal_lookup:
        #             std, mean = normal_lookup[self.hparams.dataset_name]
        #             self.transform.transforms.append(transforms.Normalize(mean=mean, std=std))
        #         elif self.hparams.dataset_name in ["aggregate"]: # these are combined datasets
        #             pass
        #         else:
        #             raise ValueError(f"{self.hparams.dataset_name} not in normal lookup")
                    
        #     else:
        #         if self.hparams.vae_normalization:
        #             self.transform = transforms.Compose([
        #                 transforms.Resize((self.image_dims[0], self.image_dims[1])),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize([0.5], [0.5])
        #             ])
        #         else: # imagenet standardization
        #             self.transform = transforms.Compose([
        #                 transforms.Resize((self.image_dims[0], self.image_dims[1])),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #             ])
        #     self.reset_image_encoder_decoder = False
        # if self.hparams.modality == "IMG": # using transform from DiT codebase https://github.com/facebookresearch/DiT
        #     self.transform = transforms.Compose([
        #         transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.hparams.image_dims[0])),
        #         # transforms.RandomHorizontalFlip(), # remove this since adds more modes
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        #     ])

        # self.to_pil = ToPILImage()
        self.full_ds = None

        # IMPORTANT: Tokenizer configuration for NanoChat dataset
        # The tokenizer is loaded via get_tokenizer() which uses NanoChat custom BPE tokenizer
        # from $NANOCHAT_BASE_DIR/tokenizer/ (vocab_size=32768)
        # The --tokenizer parameter passed from command line is IGNORED for NanoChat!
        self.hparams.tokenizer_obj = tokenizer = get_tokenizer() # Store tokenizer object
        # Keep tokenizer path as string for generate_text compatibility
        if not hasattr(self.hparams, 'tokenizer_path'):
            self.hparams.tokenizer_path = self.hparams.tokenizer if isinstance(self.hparams.tokenizer, str) else "EleutherAI/gpt-neox-20b"

        # Load token_bytes for BPB (bits per byte) calculation
        # token_bytes maps each token id to its byte length, used in validation/test metrics
        try:
            self.token_bytes = get_token_bytes(device="cpu")  # Will be moved to GPU when needed
            print(f"  Token bytes loaded: shape={self.token_bytes.shape}")
        except Exception as e:
            print(f"  Warning: Could not load token_bytes: {e}")
            print(f"  BPB metrics will not be available")
            self.token_bytes = None

        # Print tokenizer info for clarity
        print(f"=" * 80)
        print(f"TOKENIZER INFO:")
        print(f"  Actual tokenizer used: NanoChat custom BPE tokenizer")
        print(f"  Tokenizer location: $NANOCHAT_BASE_DIR/tokenizer/")
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
        print(f"  Command-line --tokenizer parameter: {self.hparams.tokenizer} (IGNORED)")
        print(f"=" * 80)

        if trained_model is not None:
            self.model = trained_model
        else:
            self.model = EBT_NLP(self.hparams)
            # if self.hparams.model_name == "ebt":
            #     if self.hparams.modality == "VID":
            #         self.model = EBT_VID(self.hparams)
            #     elif self.hparams.modality == "NLP":
            #         self.model = EBT_NLP(self.hparams)
            #     elif self.hparams.modality == "IMG": # these are bidirectional not AR
            #         if self.hparams.image_task == "t2i":
            #             self.model = EBT_IMG_T2I(self.hparams) 
            #         elif self.hparams.image_task == "denoising":
            #             self.model = EBT_IMG_Denoise(self.hparams)
            #         else:
            #             raise ValueError(f"task type: {self.hparams.image_task} not supported in base model trainer as a model as of now")
            #     else:
            #         raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            # elif self.hparams.model_name == "baseline_transformer":
            #     if self.hparams.modality == "VID":
            #         self.model = Baseline_Transformer_VID(self.hparams)
            #     elif self.hparams.modality == "NLP":
            #         self.model = Baseline_Transformer_NLP(self.hparams)
            #     else:
            #         raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            # elif self.hparams.model_name == "dit":
            #     if self.hparams.modality == "IMG":
            #         if self.hparams.image_task == "t2i":
            #             self.model = Diffusion_Transformer_IMG_T2I(self.hparams) # this is bidirectional not AR
            #         elif self.hparams.image_task == "denoising":
            #             self.model = Diffusion_Transformer_IMG_Denoise(self.hparams) # this is bidirectional not AR
            #     else:
            #         raise ValueError(f"Modality: {self.hparams.modality} not supported as a base model trainer model as of now")
            # else:
            #     raise ValueError(f"do not recognize model name: {self.hparams.model_name}")

        # torch.compile 支持
        # EBT 训练时 autograd.grad(create_graph=True) 产生二阶梯度,
        # torch.compile (aot_autograd) 不支持 double backward, 因此训练时跳过编译.
        # 推理时 learning=False → create_graph=False, 可以安全编译.
        if self.hparams.compile_model:
            compile_mode = getattr(self.hparams, 'compile_mode', 'transformer_only')
            compile_backend = getattr(self.hparams, 'compile_backend', 'inductor')
            compile_dynamic = getattr(self.hparams, 'compile_dynamic', False)

            if compile_mode == 'full':
                # 编译整个模型 (可能与 autograd.grad 不兼容)
                print(f"\n{'='*80}")
                print(f"[torch.compile] 开始编译整个模型...")
                print(f"[torch.compile] 模式: full | 后端: {compile_backend} | 动态: {compile_dynamic}")
                print(f"[torch.compile] 警告: EBT 的 MCMC 循环使用 autograd.grad，可能导致编译失败")
                print(f"[torch.compile] 首次编译可能需要 5-15 分钟，请耐心等待...")
                print(f"{'='*80}\n")
                import time
                start_time = time.time()
                self.model = torch.compile(self.model, backend=compile_backend, dynamic=compile_dynamic)
                compile_time = time.time() - start_time
                print(f"\n{'='*80}")
                print(f"[torch.compile] ✓ 模型编译完成 (耗时: {compile_time:.1f}s)")
                print(f"{'='*80}\n")

            elif compile_mode == 'transformer_only':
                # 仅编译 transformer 部分 (避开 MCMC )
                # 保留 eager 引用供 _mcmc_step_excluded 中 create_graph=True 时使用
                print(f"[torch.compile] 仅编译 transformer 部分 (mode=transformer_only, backend={compile_backend})")
                if hasattr(self.model, 'transformer'):
                    self.model.transformer_eager = self.model.transformer  # 保留 eager 引用
                    self.model.transformer = torch.compile(
                        self.model.transformer,
                        backend=compile_backend,
                        dynamic=compile_dynamic
                    )
                    print(f"[torch.compile] transformer 编译成功，transformer_eager 已保留用于 MCMC")
                else:
                    print(f"[torch.compile] 警告: 模型没有 transformer 属性，跳过编译")

            elif compile_mode == 'disabled':
                print(f"[torch.compile] 编译已禁用")

            elif (self.hparams.execution_mode == "inference") or getattr(self.hparams, 'only_test', False):
                # 推理模式: learning=False → 无 double backward, 可以安全编译
                if compile_mode == 'full':
                    print(f"[torch.compile] 推理模式: 编译整个模型 (backend={compile_backend})")
                    self.model = torch.compile(self.model, backend=compile_backend, dynamic=compile_dynamic)
                elif compile_mode == 'transformer_only':
                    if hasattr(self.model, 'transformer'):
                        print(f"[torch.compile] 推理模式: 编译 transformer (backend={compile_backend})")
                        self.model.transformer = torch.compile(
                            self.model.transformer, backend=compile_backend, dynamic=compile_dynamic
                        )
                    else:
                        print(f"[torch.compile] 警告: 模型没有 transformer 属性，跳过")
                else:
                    raise ValueError(f"未知 compile_mode: {compile_mode}")

            else:
                # 训练模式: 跳过编译 (EBT MCMC 需要 double backward)
                print(f"[torch.compile] 训练模式下跳过编译 (EBT MCMC 需要 create_graph=True, aot_autograd 不支持 double backward)")

        phases = ['train', 'valid', 'test']
        self.torchmetrics_dict = nn.ModuleDict()
        self.metrics = []
        for metric in self.hparams.metrics_list:
            self.metrics.append(metric)
        if len(self.metrics) > 0:
            assert self.hparams.num_classes != -1, "please set num_classes to the appropriate amount for the in use metrics. if are using accuracy and num_classes varies just set it to something that makes sense (shouldnt matter in that case)"
            assert self.hparams.metrics_task != "", "please set metrics_task to the appropriate value for your metrics"
        for phase in phases:
            for metric in self.metrics:
                self.torchmetrics_dict[f"{phase}_{metric}"] = get_torchmetrics(metric, self.hparams.metrics_average_type, self.hparams.num_classes, self.hparams.metrics_task)

        if self.hparams.wandb_watch:
            for name, module in self.model.named_modules(): # for activation logging
                module.name = name

    def on_sanity_check_start(self):
        _sft_trainer_debug("sanity_check_start")

    def on_sanity_check_end(self):
        _sft_trainer_debug("sanity_check_end")

    def on_validation_start(self):
        _sft_trainer_debug("validation_start")

    def on_validation_end(self):
        _sft_trainer_debug("validation_end")

    def on_train_start(self):
        _sft_trainer_debug("train_start")

        # --- RNG 恢复 (在 val sanity check 之后、第一个 training step 之前) ---
        import random
        rng = getattr(self, '_rng_resume_state', None)
        if rng is not None:
            torch.random.set_rng_state(rng['torch_cpu'])
            if rng.get('torch_cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng['torch_cuda'])
            random.setstate(rng['python'])
            self._rng_resume_state = None
            print(f"[Exact Resume] RNG states restored for rank {self.global_rank}")

        if self.hparams.debug_unused_parameters: 
            for name, param in self.model.named_parameters():
                if param.requires_grad and "image_encoder" not in name: # NOTE need to modify this code to exclude specific frozen portions
                    print(f"registering param - {name}")
                    param.register_hook(self.create_hook(name))
                else:
                    self.model.parameters_not_to_check.add(name)

    def create_hook(self, name): #this is only used for debugging with `debug_unused_parameters`
        def hook(grad):
            self.model.used_parameters.add(name)  # Adjusted to self.model.used_parameters
        return hook
    
    @staticmethod
    def wandb_activation_hook(run, step):
        """ Weights & Biases stats logging hook (optimized). """
        def hook(module, input, output):
            if isinstance(output, tuple):
                pass 
            else:
                try:
                    # Optimize: Log stats on GPU instead of moving full tensor to CPU for Histogram
                    data = output.detach().float()
                    run.experiment.log(
                        {
                            f"activations/{module.name}_mean": data.mean().item(),
                            f"activations/{module.name}_std": data.std().item(),
                            f"activations/{module.name}_min": data.min().item(),
                            f"activations/{module.name}_max": data.max().item(),
                        }, 
                        step=step
                    )
                except RuntimeError:
                    # Skip logging for tensors without storage (e.g. inside torch.func.grad)
                    pass

        return hook
    
    def training_step(self, batch, batch_idx):
        # Activation logging only when wandb_watch is on AND level is "all"
        if not self.hparams.no_wandb and self.hparams.wandb_watch and getattr(self.hparams, 'wandb_watch_level', 'parameters') == 'all' and self.global_step % self.hparams.wandb_watch_log_freq == 0: # activation logging
            hook_handles = []
            hook_function = self.wandb_activation_hook(run=self.logger, step=self.global_step)
            for module in self.model.modules():
                if any(param.requires_grad for param in module.parameters(recurse=False)): # only do for unfrozen params that are training
                    handle = module.register_forward_hook(hook_function)
                    hook_handles.append(handle)
            
            eval_step_dict = self.eval_step(batch, "train")
            for handle in hook_handles:
                handle.remove()

        else:
            eval_step_dict = self.eval_step(batch, "train")
        
        self.log_metrics(eval_step_dict, "train")
        return eval_step_dict['loss']   
    
    def on_after_backward(self):
        if self.hparams.log_gradients:
            total_norm = 0.0
            num_parameters = 0
            num_grads_exceeding_clip_val = 0
            total_gradients = 0 # this is different from num_parameters since .parameters is for tensors of params but doesnt count each invididual parameter
            for param in self.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm  # Add the norm value to the total sum
                    num_parameters += 1
                    
                    total_gradients += torch.numel(param.grad)
                    num_grads_exceeding_clip_val += torch.sum(param.grad.abs() > self.hparams.gradient_clip_val)
                    
            assert num_parameters > 0, "no gradients after backwards detected please investigate"
            average_norm = (total_norm / num_parameters).detach()
            percentage_clipped = ((num_grads_exceeding_clip_val / total_gradients) * 100).detach()              
            
            things_to_log = {} 
            things_to_log['avg_gradient_norms'] = average_norm
            things_to_log['pct_gradient_clipped'] = percentage_clipped
            self.log_metrics(things_to_log, "train", log_torchmetrics = False)
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        #NOTE when using this may need to explicitly add code like 'if "image_encoder" not in name' for frozen params (with requires_grad == False)
        if self.hparams.debug_unused_parameters:
            all_parameters = {name for name, _ in self.model.named_parameters()}
            unused_parameters = all_parameters - self.model.used_parameters - self.model.parameters_not_to_check

            print(f"number of parameters total: {len(all_parameters)}")
            print(f"number of unused_parameters: {len(unused_parameters)}")
            print(f"Unused parameters: {unused_parameters}")
            print(f"Used parameters: {self.model.used_parameters}")

        if self.hparams.manual_gc_collect_every_n_steps != -1:
            if self.global_step > 0 and self.global_step % self.hparams.manual_gc_collect_every_n_steps == 0:
                print("calling GC manually")
                gc.collect()
                torch.cuda.empty_cache()

        # --- Muon momentum 预热调度 (参考 NanoChat base_train.py:360-363) ---
        # Muon momentum 从 0.85 线性预热到 0.95，前 300 步完成
        # 通过 --muon_momentum_warmup_steps 控制（默认 300，设 0 禁用）
        muon_warmup_steps = getattr(self.hparams, 'muon_momentum_warmup_steps', 300)
        if muon_warmup_steps > 0 and self.global_step <= muon_warmup_steps:
            if hasattr(self, 'trainer') and self.trainer.optimizers:
                optimizer = self.trainer.optimizers[0]
                if hasattr(optimizer, 'param_groups'):
                    target_momentum = getattr(self.hparams, 'muon_momentum', 0.95)
                    base_momentum = 0.85
                    frac = min(self.global_step / muon_warmup_steps, 1.0)
                    current_momentum = (1 - frac) * base_momentum + frac * target_momentum
                    for group in optimizer.param_groups:
                        if group.get('kind') == 'muon':
                            group['momentum'] = current_momentum

        # Record step end time for dt calculation
        import time as _time
        now = _time.time()
        if self._train_step_start_time is not None:
            self._last_dt = now - self._train_step_start_time
        else:
            self._last_dt = None
        self._train_step_start_time = now
        if self._train_start_time is None:
            self._train_start_time = now

    # def on_train_epoch_end(self): ## not effective for EBT
    #     if self.hparams.optimizer != "adamw": # e.g. for lars need to manually update epoch
    #         optimizer = self.trainer.optimizers[0]
    #         optimizer.update_epoch(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        # 保存 per-rank dataloader 位置 + RNG 状态到 checkpoint，用于精确续训
        import torch.distributed as dist
        import random

        # 1. 收集当前 rank 的 dataloader state（精确版，含 doc_buffer）
        local_dl_state = None
        try:
            train_dl = self.trainer.train_dataloader
            if train_dl is not None:
                dataset = train_dl.dataset
                if hasattr(dataset, 'get_dataloader_state'):
                    local_dl_state = dataset.get_dataloader_state()
                elif hasattr(dataset, 'last_state_dict') and dataset.last_state_dict is not None:
                    local_dl_state = dataset.last_state_dict
        except Exception:
            pass  # 非训练阶段可能没有 train_dataloader

        # 2. 收集当前 rank 的 RNG state
        local_rng_state = {
            'torch_cpu': torch.random.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'python': random.getstate(),
        }

        # 3. DDP: all_gather 收集所有 rank 的状态到 rank 0
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_dl_states = [None] * dist.get_world_size()
            dist.all_gather_object(all_dl_states, local_dl_state)
            all_rng_states = [None] * dist.get_world_size()
            dist.all_gather_object(all_rng_states, local_rng_state)
        else:
            all_dl_states = [local_dl_state]
            all_rng_states = [local_rng_state]

        # 4. 写入 checkpoint
        checkpoint['dataloader_state_dict_by_rank'] = all_dl_states
        checkpoint['dataloader_state_dict'] = all_dl_states[0]  # 旧格式兼容
        checkpoint['rng_states_by_rank'] = all_rng_states

        print(f"[Checkpoint] 保存 per-rank dataloader state ({len(all_dl_states)} ranks) + RNG states")

    def on_load_checkpoint(self, checkpoint):
        # --- 修复 torch.compile _orig_mod 前缀不匹配 ---
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            has_orig_mod_keys = any('_orig_mod.' in k for k in state_dict)
            model_has_orig_mod = any('_orig_mod.' in k for k in self.state_dict())

            if has_orig_mod_keys and not model_has_orig_mod:
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace('._orig_mod.', '.')] = v
                checkpoint['state_dict'] = new_state_dict
                print(f"[Checkpoint] Stripped '_orig_mod' prefix from {len(state_dict)} keys")
            elif not has_orig_mod_keys and model_has_orig_mod:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict['model._orig_mod.' + k[len('model.'):]] = v
                    else:
                        new_state_dict[k] = v
                checkpoint['state_dict'] = new_state_dict
                print(f"[Checkpoint] Added '_orig_mod' prefix to {len(state_dict)} keys")

        # 从 checkpoint 恢复 per-rank dataloader 位置 + RNG 状态
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Dataloader state: 优先 per-rank，回退旧格式
        if 'dataloader_state_dict_by_rank' in checkpoint:
            states = checkpoint['dataloader_state_dict_by_rank']
            self._dataloader_resume_state = states[rank] if rank < len(states) else None
        elif 'dataloader_state_dict' in checkpoint:
            self._dataloader_resume_state = checkpoint['dataloader_state_dict']

        # RNG state: per-rank
        if 'rng_states_by_rank' in checkpoint:
            rng_states = checkpoint['rng_states_by_rank']
            self._rng_resume_state = rng_states[rank] if rank < len(rng_states) else None
        else:
            self._rng_resume_state = None

        if self._dataloader_resume_state:
            if 'cursor' in self._dataloader_resume_state:
                print(
                    f"[Checkpoint] Rank {rank} 恢复 SFT dataloader state: "
                    f"cursor={self._dataloader_resume_state.get('cursor')}, "
                    f"consumed={self._dataloader_resume_state.get('consumed')}, "
                    f"epoch={self._dataloader_resume_state.get('epoch')}, "
                    f"it={self._dataloader_resume_state.get('it')}, "
                    f"state_version={self._dataloader_resume_state.get('state_version', 'legacy')}"
                )
            else:
                print(f"[Checkpoint] Rank {rank} 恢复 dataloader state: "
                      f"pq_idx={self._dataloader_resume_state.get('pq_idx')}, "
                      f"rg_idx={self._dataloader_resume_state.get('rg_idx')}, "
                      f"state_version={self._dataloader_resume_state.get('state_version', 'legacy')}")
        if self._rng_resume_state:
            print(f"[Checkpoint] Rank {rank} 恢复 RNG state: keys={list(self._rng_resume_state.keys())}")

    def on_validation_epoch_start(self):
        """Reset validation accumulators at the start of each validation epoch."""
        self._val_source_buf = {}
        self._val_source_bpb = {}
        self._val_source_loss_buf = {}
        self._val_loss_sum = None
        self._val_loss_tokens = None
        self._val_bpb_nats = 0.0
        self._val_bpb_bytes = 0
        self._val_supervised_tokens = 0
        self._val_empty_supervision_batches = 0

    def _cache_valid_metric(self, key, value):
        """Cache validation metrics with and without the Lightning phase prefix."""
        if not hasattr(self, '_last_valid_metrics'):
            self._last_valid_metrics = {}
        self._last_valid_metrics[key] = value
        self._last_valid_metrics[f'valid_{key}'] = value

    def _active_sudoku_ratio(self, default=0.6):
        """Read the mutable mixed-dataset ratio, falling back to static hparams."""
        try:
            train_dl = getattr(self.trainer, 'train_dataloader', None)
            dataset = getattr(train_dl, 'dataset', None)
            if dataset is not None and hasattr(dataset, 'sudoku_ratio'):
                return float(dataset.sudoku_ratio)
        except Exception:
            pass
        return float(getattr(self.hparams, 'sudoku_ratio', default))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx < 2:
            _sft_trainer_debug(
                f"validation_step_start batch_idx={batch_idx} dataloader_idx={dataloader_idx}"
            )

        token_bytes = self.token_bytes
        if token_bytes is not None and token_bytes.device != self.device:
            token_bytes = token_bytes.to(self.device)
        eval_step_dict = self.eval_step(batch, "valid", token_bytes)

        if batch_idx < 2:
            keys = ",".join(sorted(eval_step_dict.keys()))
            _sft_trainer_debug(
                f"validation_step_eval_done batch_idx={batch_idx} "
                f"dataloader_idx={dataloader_idx} keys={keys}"
            )
        if getattr(self.trainer, "sanity_checking", False):
            if batch_idx < 2:
                _sft_trainer_debug(
                    f"validation_step_sanity_skip_logging batch_idx={batch_idx} "
                    f"dataloader_idx={dataloader_idx}"
                )
            return

        sources = getattr(self, '_val_loader_sources', None)
        has_source = sources is not None and 0 <= dataloader_idx < len(sources)
        src = sources[dataloader_idx] if has_source else None

        loss_value = eval_step_dict.get('loss', None)
        supervised_tokens = eval_step_dict.get('supervised_tokens', 0)
        loss_tensor = None
        token_tensor = None
        if loss_value is not None:
            if isinstance(loss_value, torch.Tensor):
                loss_tensor = loss_value.detach().to(device=self.device, dtype=torch.float32)
            else:
                loss_tensor = torch.tensor(loss_value, device=self.device, dtype=torch.float32)
            if isinstance(supervised_tokens, torch.Tensor):
                token_tensor = supervised_tokens.detach().to(device=self.device, dtype=torch.float32)
            else:
                token_tensor = torch.tensor(supervised_tokens, device=self.device, dtype=torch.float32)
            if self._val_loss_sum is None:
                self._val_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
                self._val_loss_tokens = torch.zeros((), device=self.device, dtype=torch.float32)
            self._val_loss_sum = self._val_loss_sum + loss_tensor * token_tensor
            self._val_loss_tokens = self._val_loss_tokens + token_tensor
            if has_source:
                source_loss = self._val_source_loss_buf.setdefault(
                    src,
                    {
                        'sum': torch.zeros((), device=self.device, dtype=torch.float32),
                        'tokens': torch.zeros((), device=self.device, dtype=torch.float32),
                    },
                )
                source_loss['sum'] = source_loss['sum'] + loss_tensor * token_tensor
                source_loss['tokens'] = source_loss['tokens'] + token_tensor

        if has_source:
            tagged = {
                f"{k}_{src}": v
                for k, v in eval_step_dict.items()
                if k not in ('loss', 'bpb', 'bpb_nats', 'bpb_bytes')
            }
            self.log_metrics(tagged, "valid")
            buf = self._val_source_buf.setdefault(src, {})
            for k, v in eval_step_dict.items():
                if k in ('bpb', 'bpb_nats', 'bpb_bytes'):
                    continue
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    val = v.detach().float()
                elif isinstance(v, (int, float)):
                    val = torch.tensor(float(v), device=self.device)
                else:
                    continue
                slot = buf.setdefault(
                    k,
                    {
                        'sum': torch.zeros((), device=self.device),
                        'count': torch.zeros((), device=self.device),
                    },
                )
                slot['sum'] = slot['sum'] + val
                slot['count'] = slot['count'] + 1
        else:
            self.log_metrics(eval_step_dict, "valid")

        # 累积 BPB 的 nats/bytes，用于 epoch-level 正确计算
        # (BPB = sum(nats) / (log2 * sum(bytes)), 不能对 per-batch BPB 做算术平均)
        bpb_nats = eval_step_dict.get('bpb_nats', 0)
        bpb_bytes = eval_step_dict.get('bpb_bytes', 0)
        if isinstance(bpb_nats, torch.Tensor):
            bpb_nats = bpb_nats.item()
        if isinstance(bpb_bytes, torch.Tensor):
            bpb_bytes = bpb_bytes.item()
        self._val_bpb_nats += bpb_nats
        self._val_bpb_bytes += bpb_bytes
        if has_source:
            bpb_slot = self._val_source_bpb.setdefault(src, {'nats': 0.0, 'bytes': 0})
            bpb_slot['nats'] += bpb_nats
            bpb_slot['bytes'] += bpb_bytes

        supervised_tokens = eval_step_dict.get('supervised_tokens', 0)
        empty_supervision_batch = eval_step_dict.get('empty_supervision_batch', 0)
        if isinstance(supervised_tokens, torch.Tensor):
            supervised_tokens = supervised_tokens.item()
        if isinstance(empty_supervision_batch, torch.Tensor):
            empty_supervision_batch = empty_supervision_batch.item()
        self._val_supervised_tokens += supervised_tokens
        self._val_empty_supervision_batches += empty_supervision_batch

        suffix = f"_{src}" if has_source else ""
        for k, v in eval_step_dict.items():
            if k in ('bpb', 'bpb_nats', 'bpb_bytes'):
                continue
            if k == 'loss' and not has_source:
                continue
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                self._cache_valid_metric(f"{k}{suffix}", v.detach().item())
            elif isinstance(v, (int, float)):
                self._cache_valid_metric(f"{k}{suffix}", v)

        if batch_idx < 2:
            _sft_trainer_debug(
                f"validation_step_done batch_idx={batch_idx} dataloader_idx={dataloader_idx}"
            )

    def on_validation_epoch_end(self):
        """Compute mixed validation metrics plus token-weighted epoch loss/BPB."""
        if getattr(self.trainer, "sanity_checking", False):
            _sft_trainer_debug("validation_epoch_end_sanity_skip_logging")
            return

        import math
        import torch.distributed as dist

        loss_sum = self._val_loss_sum
        loss_tokens = self._val_loss_tokens
        if loss_sum is None:
            loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            loss_tokens = torch.zeros((), device=self.device, dtype=torch.float32)
        else:
            loss_sum = loss_sum.clone()
            loss_tokens = loss_tokens.clone()

        if dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_tokens, op=dist.ReduceOp.SUM)

        if loss_tokens.item() > 0:
            epoch_loss = loss_sum / loss_tokens
        else:
            epoch_loss = torch.tensor(float('inf'), device=self.device, dtype=torch.float32)
        epoch_loss_value = epoch_loss.detach().item()

        sources = getattr(self, '_val_loader_sources', None)
        buf = getattr(self, '_val_source_buf', None)
        if sources and buf:
            ratio = self._active_sudoku_ratio(default=0.6)
            weights = {'sudoku': ratio, 'sft': 1.0 - ratio}
            balanced_weights = {'sudoku': 0.5, 'sft': 0.5}
            all_keys = set()
            for src in sources:
                all_keys.update(buf.get(src, {}).keys())

            source_means = {}
            for src in sources:
                for k, slot in buf.get(src, {}).items():
                    if slot['count'].item() > 0:
                        source_means.setdefault(src, {})[k] = (slot['sum'] / slot['count']).item()

            source_loss_buf = getattr(self, '_val_source_loss_buf', {})
            for src, slot in source_loss_buf.items():
                src_loss_sum = slot['sum'].clone()
                src_loss_tokens = slot['tokens'].clone()
                if dist.is_initialized():
                    dist.all_reduce(src_loss_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(src_loss_tokens, op=dist.ReduceOp.SUM)
                if src_loss_tokens.item() > 0:
                    source_means.setdefault(src, {})['loss'] = (
                        src_loss_sum / src_loss_tokens
                    ).detach().item()

            mixed = {}
            balanced = {}
            for k in all_keys:
                total = 0.0
                wsum = 0.0
                btotal = 0.0
                bwsum = 0.0
                for src in sources:
                    if k not in source_means.get(src, {}):
                        continue
                    mean = source_means[src][k]
                    w = weights.get(src, 0.0)
                    bw = balanced_weights.get(src, 0.0)
                    total += w * mean
                    wsum += w
                    btotal += bw * mean
                    bwsum += bw
                if wsum > 0:
                    out_key = 'loss_ratio_weighted' if k == 'loss' else k
                    mixed[out_key] = total / wsum
                if bwsum > 0:
                    balanced[k + '_balanced'] = btotal / bwsum

            source_bpb = getattr(self, '_val_source_bpb', {})
            for src, slot in source_bpb.items():
                if slot['bytes'] > 0:
                    bpb = slot['nats'] / (math.log(2) * slot['bytes'])
                    mixed[f'bpb_{src}'] = bpb
                    self._cache_valid_metric(f'bpb_{src}', bpb)

            for src, means in source_means.items():
                for k, v in means.items():
                    self._cache_valid_metric(f'{k}_{src}', v)
                    if k == 'loss':
                        self.log(f"valid_loss_{src}", v, prog_bar=False, sync_dist=False)

            if mixed:
                self.log_metrics(mixed, "valid")
                for k, v in mixed.items():
                    self._cache_valid_metric(k, v)
            if balanced:
                self.log_metrics(balanced, "valid")
                for k, v in balanced.items():
                    self._cache_valid_metric(k, v)

        if self._val_bpb_bytes > 0:
            epoch_bpb = self._val_bpb_nats / (math.log(2) * self._val_bpb_bytes)
        else:
            epoch_bpb = float('inf')

        self._cache_valid_metric('loss', epoch_loss_value)
        self._cache_valid_metric('bpb', epoch_bpb)
        self._cache_valid_metric('supervised_tokens', self._val_supervised_tokens)
        self._cache_valid_metric('empty_supervision_batch', self._val_empty_supervision_batches)
        self.log("valid_loss", epoch_loss, prog_bar=True, sync_dist=False)

        if getattr(self.trainer, "is_global_zero", True):
            valid_summary = (
                f"[Validation] global_step={self.global_step} | "
                f"valid_loss: {epoch_loss.detach().item():.6f} | "
                f"valid_bpb: {epoch_bpb:.6f}"
            )
            valid_summary += (
                f" | valid_supervised_tokens: {self._val_supervised_tokens} | "
                f"valid_empty_supervision_batch: {self._val_empty_supervision_batches}"
            )
            print(valid_summary, flush=True)

        # 直接上报正确的 epoch-level BPB 到 wandb，覆盖 Lightning 的算术平均值
        if self.logger is not None:
            try:
                self.logger.experiment.log({
                    'valid_loss': epoch_loss_value,
                    'valid_bpb': epoch_bpb,
                    'valid_supervised_tokens': self._val_supervised_tokens,
                    'valid_empty_supervision_batch': self._val_empty_supervision_batches,
                }, step=self.global_step)
            except Exception:
                pass

    def on_test_epoch_start(self):
        """Reset test metrics at the start of test epoch"""
        import time
        self.test_losses = []
        self.test_perplexities = []
        self.test_energies = {}
        self.test_start_time = time.time()
        self.test_generation_count = 0  # track generated samples for GSM8K etc.

        # Print header
        import sys
        sys.stdout.write(f"\n{'='*100}\n")
        sys.stdout.write(f"{'🚀 STARTING EVALUATION':^100}\n")
        sys.stdout.write(f"{'='*100}\n\n")
        sys.stdout.flush()

    def test_step(self, batch, batch_idx):
        if self.hparams.execution_mode == "inference":
            if self.hparams.modality == "NLP":
                # For GSM8K and other generation tasks that use DataLoader with collate_fn
                if self.hparams.dataset_name == "gsm8k":
                    outputs = generate_text(self.model, batch, self.hparams)
                    self.test_generation_count += len(outputs)
                    for output in outputs:
                        self.infer_logger.log_data(output)

                # For nanochat_shard_eval with dual-mode evaluation (PPL + Generation)
                elif self.hparams.dataset_name == "nanochat_shard_eval":
                    # Check if generation is enabled
                    enable_generation = getattr(self.hparams, 'enable_nanochat_generation', True)

                    # 1. Always compute PPL on full sequence
                    batch_dict = batch  # Already a dict from DataLoader
                    ppl_outputs = get_ppl(self.model, batch_dict, self.hparams)

                    # Track metrics for averaging
                    self.test_losses.append(ppl_outputs['loss'].item())
                    self.test_perplexities.append(ppl_outputs['perplexity'].item())

                    # Track energy metrics if available
                    if not hasattr(self, 'test_energies'):
                        self.test_energies = {}
                    for key, value in ppl_outputs.items():
                        if 'energy' in key:
                            if key not in self.test_energies:
                                self.test_energies[key] = []
                            self.test_energies[key].append(value)

                    # 2. If generation is enabled and prompt data is available, do text generation
                    if enable_generation and 'prompt_ids' in batch:
                        # Prepare batch for generation (similar to GSM8K format)
                        # questions = prompts, answers = targets
                        questions = {
                            'input_ids': batch['prompt_ids'],
                            'attention_mask': batch['prompt_attention_mask']
                        }
                        # Pad target_ids to same length before stacking
                        target_list = batch['target_ids']
                        max_target_len = max(t.shape[0] for t in target_list)
                        padded_targets = []
                        for t in target_list:
                            pad_len = max_target_len - t.shape[0]
                            if pad_len > 0:
                                padded_t = torch.cat([t, torch.zeros(pad_len, dtype=t.dtype, device=t.device)])
                            else:
                                padded_t = t
                            padded_targets.append(padded_t)
                        answers = {
                            'input_ids': torch.stack(padded_targets)
                        }

                        generation_batch = (questions, answers)
                        generation_outputs = generate_text(self.model, generation_batch, self.hparams)

                        # Log generation results with additional context
                        for i, output in enumerate(generation_outputs):
                            # Add PPL info and shard index
                            output['loss'] = ppl_outputs['loss'].item()
                            output['ppl'] = ppl_outputs['perplexity'].item()
                            output['shard_idx'] = batch['shard_indices'][i]
                            # Note: prompt and target are already in output from generate_text()
                            # No need to add duplicate fields

                            self.infer_logger.log_data(output)

                    # Print progress every 5 batches
                    if batch_idx % 5 == 0:
                        import sys
                        import numpy as np
                        import time

                        # Calculate statistics
                        current_avg_loss = np.mean(self.test_losses)
                        current_avg_ppl = np.mean(self.test_perplexities)
                        current_std_loss = np.std(self.test_losses) if len(self.test_losses) > 1 else 0.0
                        current_std_ppl = np.std(self.test_perplexities) if len(self.test_perplexities) > 1 else 0.0

                        # Estimate time remaining
                        if not hasattr(self, 'test_start_time'):
                            self.test_start_time = time.time()
                        elapsed = time.time() - self.test_start_time
                        batches_done = batch_idx + 1
                        batches_total = self.hparams.limit_test_batches if self.hparams.limit_test_batches != 1 else 100
                        eta = elapsed / batches_done * (batches_total - batches_done) if batches_done > 0 else 0

                        sys.stdout.write(f"\n{'─'*100}\n")
                        sys.stdout.write(f"📊 Batch {batch_idx:3d}/{batches_total} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s\n")
                        sys.stdout.write(f"{'─'*100}\n")
                        sys.stdout.write(f"  Current Batch:  Loss={ppl_outputs['loss'].item():.4f}  PPL={ppl_outputs['perplexity'].item():.2f}\n")
                        sys.stdout.write(f"  Running Avg:    Loss={current_avg_loss:.4f} (±{current_std_loss:.4f})  PPL={current_avg_ppl:.2f} (±{current_std_ppl:.2f})\n")

                        # Show energy metrics if available
                        if self.test_energies:
                            energy_strs = []
                            for key, values in self.test_energies.items():
                                avg_energy = np.mean(values)
                                energy_strs.append(f"{key}={avg_energy:.2f}")
                            sys.stdout.write(f"  Energy Metrics: {' | '.join(energy_strs)}\n")

                        if enable_generation and 'prompt_ids' in batch:
                            sys.stdout.write(f"  Generation: Enabled (prompt→target)\n")

                        sys.stdout.write(f"{'─'*100}\n")
                        sys.stdout.flush()

                    # Log metrics
                    filtered_outputs = {k: v for k, v in ppl_outputs.items() if 'energy' not in k}
                    self.log_metrics(filtered_outputs, "test")

                else:
                    # For nanochat and other PPL evaluation tasks
                    # nanochat uses IterableDataset which returns (x, y) tuples
                    # During training: x[0].squeeze(dim=0) is used (see modeling_ebt.py:189)
                    # Convert to dict format for get_ppl
                    if isinstance(batch, tuple):
                        # batch is (x, y) from IterableDataset
                        x, y = batch
                        # x might have shape [1, B, S] or [B, S], squeeze to ensure [B, S]
                        if x.dim() == 3 and x.shape[0] == 1:
                            x = x.squeeze(0)  # [1, B, S] -> [B, S]
                        # Add channel dimension for get_ppl: [B, S] -> [B, 1, S]
                        batch_dict = {'input_ids': x.unsqueeze(1)}
                    else:
                        # batch is already a dict from DataLoader
                        batch_dict = batch

                    # Compute PPL and save sample outputs
                    ppl_outputs = get_ppl(self.model, batch_dict, self.hparams)

                    # Track metrics for averaging
                    self.test_losses.append(ppl_outputs['loss'].item())
                    self.test_perplexities.append(ppl_outputs['perplexity'].item())

                    # Track energy metrics if available
                    if not hasattr(self, 'test_energies'):
                        self.test_energies = {}
                    for key, value in ppl_outputs.items():
                        if 'energy' in key:
                            if key not in self.test_energies:
                                self.test_energies[key] = []
                            self.test_energies[key].append(value)

                    # Print progress every 5 batches (more frequent)
                    if batch_idx % 5 == 0:
                        import sys
                        import numpy as np
                        import time

                        # Calculate statistics
                        current_avg_loss = np.mean(self.test_losses)
                        current_avg_ppl = np.mean(self.test_perplexities)
                        current_std_loss = np.std(self.test_losses) if len(self.test_losses) > 1 else 0.0
                        current_std_ppl = np.std(self.test_perplexities) if len(self.test_perplexities) > 1 else 0.0

                        # Estimate time remaining
                        if not hasattr(self, 'test_start_time'):
                            self.test_start_time = time.time()
                        elapsed = time.time() - self.test_start_time
                        batches_done = batch_idx + 1
                        batches_total = self.hparams.limit_test_batches if self.hparams.limit_test_batches != 1 else 100
                        eta = elapsed / batches_done * (batches_total - batches_done) if batches_done > 0 else 0

                        sys.stdout.write(f"\n{'─'*100}\n")
                        sys.stdout.write(f"📊 Batch {batch_idx:3d}/{batches_total} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s\n")
                        sys.stdout.write(f"{'─'*100}\n")
                        sys.stdout.write(f"  Current Batch:  Loss={ppl_outputs['loss'].item():.4f}  PPL={ppl_outputs['perplexity'].item():.2f}\n")
                        sys.stdout.write(f"  Running Avg:    Loss={current_avg_loss:.4f} (±{current_std_loss:.4f})  PPL={current_avg_ppl:.2f} (±{current_std_ppl:.2f})\n")

                        # Show energy metrics if available
                        if self.test_energies:
                            energy_strs = []
                            for key, values in self.test_energies.items():
                                avg_energy = np.mean(values)
                                energy_strs.append(f"{key}={avg_energy:.2f}")
                            sys.stdout.write(f"  Energy Metrics: {' | '.join(energy_strs)}\n")

                        sys.stdout.write(f"{'─'*100}\n")
                        sys.stdout.flush()

                    # Save sample inputs/outputs to inference logger for first few batches
                    if batch_idx < 5 and hasattr(self, 'infer_logger'):
                        tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else None
                        if tokenizer is not None:
                            # Wrap nanochat tokenizer if needed
                            from openebm.elm.nanochat_tokenizer_adapter import NanoChatTokenizerWrapper
                            if hasattr(tokenizer, 'enc') and hasattr(tokenizer.enc, 'encode'):
                                # It's a RustBPETokenizer, wrap it for HF compatibility
                                tokenizer = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer)

                            # Log samples from this batch
                            sample_ids = batch_dict['input_ids'][:2]  # First 2 samples
                            for i, ids in enumerate(sample_ids):
                                # Decode full sequence
                                full_text = tokenizer.decode(ids.squeeze().tolist(), skip_special_tokens=True)

                                # Create output record
                                output_record = {
                                    "batch_idx": batch_idx,
                                    "sample_idx": i,
                                    "text": full_text[:500],  # First 500 chars
                                    "loss": ppl_outputs['loss'].item(),
                                    "perplexity": ppl_outputs['perplexity'].item(),
                                }
                                self.infer_logger.log_data(output_record)

                                # Also print to console for first few samples
                                if batch_idx < 3:
                                    import sys
                                    sys.stdout.write(f"\n{'─'*100}\n")
                                    sys.stdout.write(f"📄 Sample Text [Batch {batch_idx}, Sample {i}]:\n")
                                    sys.stdout.write(f"{'─'*100}\n")
                                    sys.stdout.write(f"{full_text[:300]}...\n")
                                    sys.stdout.write(f"{'─'*100}\n")
                                    sys.stdout.write(f"   Loss: {ppl_outputs['loss'].item():.4f} | PPL: {ppl_outputs['perplexity'].item():.2f}\n")
                                    sys.stdout.write(f"{'─'*100}\n\n")
                                    sys.stdout.flush()

                    # Log metrics (filter out energy metrics to avoid clutter)
                    filtered_outputs = {k: v for k, v in ppl_outputs.items() if 'energy' not in k}
                    self.log_metrics(filtered_outputs, "test")

                # TODO
                # outputs = get_ppl(self.model, batch, self.hparams)
                # self.log_metrics(outputs, "test")
            # elif self.hparams.modality == "VID":
            #     if not self.reset_image_encoder_decoder: # this is done to prevent bug where loading ckpt image encoder doesnt work well, not sure why ckpt image decoder doesnt load well, maybe related to HF
            #         self.model.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size).to(self.device)
            #         self.model.image_encoder.eval()
            #         self.reset_image_encoder_decoder = True

            #     outputs = generate_video(self.model, batch, self.hparams, decode_frames = self.hparams.infer_generate_video) # outputs['video'] has shame shape as batch: B, S, C, W, H

            #     if self.hparams.infer_generate_video:
            #         denormalized_predicted_videos = denormalize(outputs['video'], self.hparams.dataset_name, self.device, self.hparams.custom_image_normalization, self.hparams.vae_normalization)
            #         denormalized_batch = denormalize(batch, self.hparams.dataset_name, self.device, self.hparams.custom_image_normalization, self.hparams.vae_normalization)
            #         batch_size = outputs['video'].shape[0]
            #         if self.trainer.world_size > 1:
            #             batch_size_tensor = torch.tensor(batch_size, device=self.device)
            #             all_reduce(batch_size_tensor)
            #             total_batch_size = batch_size_tensor.item()
            #             video_start_idx = self.num_generated_videos + (self.global_rank * batch_size)
            #         else:
            #             total_batch_size = batch_size
            #             video_start_idx = self.num_generated_videos

            #         save_frames(denormalized_predicted_videos, self.hparams.save_generation_logs_dir, 'fake', video_start_idx) 
            #         save_frames(denormalized_batch, self.hparams.save_generation_logs_dir, 'real', video_start_idx)
            #         if self.hparams.debug_videos:
            #             save_frames(denormalized_predicted_videos[0].unsqueeze(dim=0), self.hparams.save_generation_logs_dir, 'debug', video_start_idx)
                    
            #         self.num_generated_videos += total_batch_size
            #     outputs.pop('video')
            #     self.log_metrics(outputs, "test")
            # elif self.hparams.modality == "IMG":
            #     outputs = generate_image(self.model, batch, self.hparams)
            #     self.log_metrics(outputs, "test")
            
            else:
                raise NotImplementedError(f"Inference mode not supported for modality {self.hparams.modality} yet")
        else: # all other modes just get metrics
            if self.hparams.modality == "NLP" and self.hparams.model_name == "ebt" and self.hparams.infer_ebt_advanced: # special case where we dont want to use inference mode but still use ebt advanced inference to get log ppl, energies, etc (that way dont need to generate text 1 by 1)
                outputs = get_ppl(self.model, batch, self.hparams)
                self.log_metrics(outputs, "test")
            else:
                eval_step_dict = self.eval_step(batch, "test")
                self.log_metrics(eval_step_dict, "test")

    def on_test_epoch_end(self):
        """Print comprehensive summary statistics at the end of test epoch"""
        import sys
        import numpy as np
        import time

        total_time = time.time() - self.test_start_time
        has_ppl = len(self.test_losses) > 0
        has_generation = getattr(self, 'test_generation_count', 0) > 0

        if not has_ppl and not has_generation:
            return

        sys.stdout.write(f"\n\n")
        sys.stdout.write(f"{'='*100}\n")
        sys.stdout.write(f"{'EVALUATION RESULTS SUMMARY':^100}\n")
        sys.stdout.write(f"{'='*100}\n\n")

        # Dataset info
        sys.stdout.write(f"Dataset Information:\n")
        sys.stdout.write(f"   Dataset:           {self.hparams.dataset_name}\n")
        if has_ppl:
            sys.stdout.write(f"   Total Batches:     {len(self.test_losses)}\n")
            sys.stdout.write(f"   Total Samples:     ~{len(self.test_losses) * self.hparams.batch_size_per_device}\n")
        if has_generation:
            sys.stdout.write(f"   Generated Samples: {self.test_generation_count}\n")
        sys.stdout.write(f"   Batch Size:        {self.hparams.batch_size_per_device}\n")
        sys.stdout.write(f"   Context Length:    {self.hparams.context_length}\n\n")

        # Model info
        sys.stdout.write(f"Model Information:\n")
        sys.stdout.write(f"   Model Type:        {self.hparams.model_name.upper()}\n")
        sys.stdout.write(f"   Model Size:        {self.hparams.model_size}\n")
        if self.hparams.model_name == "ebt":
            sys.stdout.write(f"   MCMC Steps:        {self.hparams.mcmc_num_steps}\n")
            sys.stdout.write(f"   MCMC Step Size:    {self.hparams.mcmc_step_size}\n")
            sys.stdout.write(f"   EBT Type:          {self.hparams.ebt_type}\n")
        sys.stdout.write(f"\n")

        # Timing info
        sys.stdout.write(f"Performance:\n")
        sys.stdout.write(f"   Total Time:        {total_time:.2f}s\n")
        if has_ppl:
            samples_per_sec = len(self.test_losses) * self.hparams.batch_size_per_device / total_time
            sys.stdout.write(f"   Time per Batch:    {total_time/len(self.test_losses):.3f}s\n")
            sys.stdout.write(f"   Throughput:        {samples_per_sec:.2f} samples/s\n")
        if has_generation:
            gen_per_sec = self.test_generation_count / total_time
            sys.stdout.write(f"   Generation Speed:  {gen_per_sec:.2f} samples/s\n")
        sys.stdout.write(f"\n")

        if has_ppl:
            # Calculate statistics
            avg_loss = np.mean(self.test_losses)
            avg_ppl = np.mean(self.test_perplexities)
            std_loss = np.std(self.test_losses)
            std_ppl = np.std(self.test_perplexities)
            min_loss = np.min(self.test_losses)
            max_loss = np.max(self.test_losses)
            min_ppl = np.min(self.test_perplexities)
            max_ppl = np.max(self.test_perplexities)
            median_loss = np.median(self.test_losses)
            median_ppl = np.median(self.test_perplexities)
            p25_loss, p75_loss = np.percentile(self.test_losses, [25, 75])
            p25_ppl, p75_ppl = np.percentile(self.test_perplexities, [25, 75])

            # Loss statistics
            sys.stdout.write(f"Cross-Entropy Loss Statistics:\n")
            sys.stdout.write(f"   Mean:              {avg_loss:.4f}\n")
            sys.stdout.write(f"   Median:            {median_loss:.4f}\n")
            sys.stdout.write(f"   Std Dev:           {std_loss:.4f}\n")
            sys.stdout.write(f"   Min:               {min_loss:.4f}\n")
            sys.stdout.write(f"   Max:               {max_loss:.4f}\n")
            sys.stdout.write(f"   25th Percentile:   {p25_loss:.4f}\n")
            sys.stdout.write(f"   75th Percentile:   {p75_loss:.4f}\n\n")

            # Perplexity statistics
            sys.stdout.write(f"Perplexity (PPL) Statistics:\n")
            sys.stdout.write(f"   Mean:              {avg_ppl:.2f}\n")
            sys.stdout.write(f"   Median:            {median_ppl:.2f}\n")
            sys.stdout.write(f"   Std Dev:           {std_ppl:.2f}\n")
            sys.stdout.write(f"   Min:               {min_ppl:.2f}\n")
            sys.stdout.write(f"   Max:               {max_ppl:.2f}\n")
            sys.stdout.write(f"   25th Percentile:   {p25_ppl:.2f}\n")
            sys.stdout.write(f"   75th Percentile:   {p75_ppl:.2f}\n\n")

            # Energy statistics if available
            if hasattr(self, 'test_energies') and self.test_energies:
                sys.stdout.write(f"Energy Landscape Statistics:\n")
                for key, values in sorted(self.test_energies.items()):
                    avg_energy = np.mean(values)
                    std_energy = np.std(values)
                    min_energy = np.min(values)
                    max_energy = np.max(values)
                    sys.stdout.write(f"   {key}:\n")
                    sys.stdout.write(f"      Mean: {avg_energy:.4f} +/- {std_energy:.4f}\n")
                    sys.stdout.write(f"      Range: [{min_energy:.4f}, {max_energy:.4f}]\n")
                sys.stdout.write(f"\n")

            # ASCII histogram for PPL distribution
            sys.stdout.write(f"Perplexity Distribution (Histogram):\n")
            hist, bin_edges = np.histogram(self.test_perplexities, bins=10)
            max_count = max(hist)
            for i in range(len(hist)):
                bar_length = int(40 * hist[i] / max_count) if max_count > 0 else 0
                bar = '#' * bar_length
                sys.stdout.write(f"   [{bin_edges[i]:6.2f}-{bin_edges[i+1]:6.2f}]: {bar} ({hist[i]})\n")
            sys.stdout.write(f"\n")

            # Quality assessment
            sys.stdout.write(f"Quality Assessment:\n")
            if avg_ppl < 20:
                quality = "Excellent"
            elif avg_ppl < 40:
                quality = "Good"
            elif avg_ppl < 60:
                quality = "Fair"
            else:
                quality = "Needs Improvement"
            sys.stdout.write(f"   Overall: {quality} (PPL={avg_ppl:.2f})\n\n")

        # GSM8K / generation-only summary
        if has_generation and not has_ppl:
            sys.stdout.write(f"Generation Summary:\n")
            sys.stdout.write(f"   Total Generated:   {self.test_generation_count}\n")
            sys.stdout.write(f"   Total Time:        {total_time:.2f}s\n")
            sys.stdout.write(f"   Avg Time/Sample:   {total_time/self.test_generation_count:.2f}s\n\n")

        # Output files
        if hasattr(self.hparams, 'save_generation_logs_dir'):
            import os
            results_file = os.path.join(self.hparams.save_generation_logs_dir, "results.jsonl")
            if os.path.exists(results_file):
                num_samples = sum(1 for _ in open(results_file))
                sys.stdout.write(f"Output Files:\n")
                sys.stdout.write(f"   Results:           {results_file}\n")
                sys.stdout.write(f"   Num Samples:       {num_samples}\n\n")

        # Machine-parseable summary block for bash grep
        sys.stdout.write(f"{'='*100}\n")
        sys.stdout.write(f"[EVAL_SUMMARY] dataset={self.hparams.dataset_name}")
        if has_ppl:
            sys.stdout.write(f" loss={avg_loss:.4f} ppl={avg_ppl:.2f}")
        if has_generation:
            sys.stdout.write(f" generated={self.test_generation_count}")
        sys.stdout.write(f" time={total_time:.1f}s\n")
        sys.stdout.write(f"{'='*100}\n\n")
        sys.stdout.flush()

    def eval_step(self, batch, phase, token_bytes=None):
        things_to_log = self.model.forward_loss_wrapper(batch, phase, token_bytes=token_bytes) # things_to_log will be a dict of various things being logged. it NEEDS TO contain the 'loss' key as this is used to backprop

        if len(self.metrics) > 0:
            raise NotImplementedError("Need to implement torchmetrics stuff, i.e. looping through self.torchmetrics_dict.keys(), checking to make sure 'phase in key', and updating based off predicted and labels i.e. self.torchmetrics_dict[key].update(logits, labels), more info https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html (just be careful make sure to detach logits before using them and only update current phase). recommended to possibly return things_to_log and logits from forward_loss_wrapper to do this easily")

        return things_to_log
    
    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self): # this is a PL hook that returns optimizer and lr scheduler
        return self.configure_optimizers_nlp()
        # if self.hparams.modality == "NLP":
        #     return self.configure_optimizers_nlp()
        # elif self.hparams.modality == "VID":
        #     return self.configure_optimizers_vid()
        # elif self.hparams.modality == "IMG":
        #     return self.configure_optimizers_img()
        # else:
        #     raise NotImplementedError(f"Modality {self.hparams.modality} does not have configure optimizers supported yet")
        
    def get_optimizer(self, optimizer_parameters): # function for once gotten optimizer_parameters to get optimizer, i.e. adamw, lars, etc
        if self.hparams.optimizer == "lars":
            lars_exclude_bias_and_norm = None if not self.hparams.lars_exclude_bias_bn_wd else exclude_bias_and_norm
            optimizer = LARS(optimizer_parameters, lr=self.hparams.peak_learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.beta1, eta=self.hparams.lars_trust_coeff, weight_decay_filter=lars_exclude_bias_and_norm, lars_adaptation_filter=lars_exclude_bias_and_norm)
        elif self.hparams.optimizer == "stableadamw":
            optimizer = StableAdamWUnfused(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
        else:
            optimizer = torch.optim.AdamW(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
        return optimizer
    
    def on_warm_up_finished(self):
        if hasattr(self.model, 'warm_up_finished'):
            self.model.warm_up_finished()
            print("Warm up finished, calling self.model.warm_up_finished()")
        else:
            print("Warm up finished, no self.model.warm_up_finished() exists so not doing anything")
    
    def get_lr_scheduler(self, optimizer):
        # Option 2: 动态 Weight Decay
        enable_wd_decay = getattr(self.hparams, 'dynamic_wd', False)

        # Option 3: Linear Warmdown LR 调度
        use_linear_warmdown = getattr(self.hparams, 'linear_warmdown', False)

        if use_linear_warmdown:
            warmup_ratio = getattr(self.hparams, 'warmup_ratio', 0.0)
            warmdown_ratio = getattr(self.hparams, 'warmdown_ratio', 0.5)
            final_lr_frac = getattr(self.hparams, 'final_lr_frac', 0.0)
            resume_warmup_steps = getattr(self.hparams, 'resume_warmup_steps', 0)

            lr_scheduler = WarmUpLinearWarmdownLR(
                optimizer,
                warmup_ratio=warmup_ratio,
                warmdown_ratio=warmdown_ratio,
                final_lr_frac=final_lr_frac,
                total_steps=self.hparams.max_scheduling_steps,
                warm_up_finished_func=self.on_warm_up_finished,
                enable_wd_decay=enable_wd_decay,
                resume_warmup_steps=resume_warmup_steps
            )
        else:
            # 原始 Cosine Annealing 调度
            cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_scheduling_steps - self.hparams.warm_up_steps,
                eta_min=self.hparams.peak_learning_rate / self.hparams.min_lr_scale
            )
            total_steps = self.hparams.max_scheduling_steps if enable_wd_decay else None

            lr_scheduler = WarmUpCosineAnnealingLR(
                optimizer,
                warm_up_steps=self.hparams.warm_up_steps,
                warm_up_base_lr_divider=self.hparams.warm_up_base_lr_divider,
                cosine_scheduler=cosine_annealing_scheduler,
                warm_up_finished_func=self.on_warm_up_finished,
                total_steps=total_steps,
                enable_wd_decay=enable_wd_decay
            )
        return lr_scheduler

    def _split_tf_head_params(self):
        tf_head_matrix_params = []
        tf_head_scalar_params = []
        if not bool(getattr(self.hparams, 'use_tf_head', False)):
            return tf_head_matrix_params, tf_head_scalar_params
        if not hasattr(self.model, 'tf_head') or self.model.tf_head is None:
            return tf_head_matrix_params, tf_head_scalar_params

        for _, param in self.model.tf_head.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                tf_head_matrix_params.append(param)
            else:
                tf_head_scalar_params.append(param)
        return tf_head_matrix_params, tf_head_scalar_params

    def _validate_optimizer_param_coverage(self, optimizer):
        named_trainable_params = [
            (name, param) for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        id_to_name = {id(param): name for name, param in named_trainable_params}
        seen_param_groups = {}
        duplicate_params = []

        for group_idx, group in enumerate(optimizer.param_groups):
            for param in group['params']:
                if not getattr(param, 'requires_grad', False):
                    continue
                param_id = id(param)
                if param_id in seen_param_groups:
                    duplicate_params.append(
                        (id_to_name.get(param_id, '<unnamed>'), seen_param_groups[param_id], group_idx)
                    )
                else:
                    seen_param_groups[param_id] = group_idx

        missing_params = [
            (name, param.numel()) for name, param in named_trainable_params
            if id(param) not in seen_param_groups
        ]
        if not missing_params and not duplicate_params:
            return

        messages = ["Optimizer parameter coverage check failed."]
        if missing_params:
            preview = ", ".join(f"{name}({numel:,})" for name, numel in missing_params[:20])
            if len(missing_params) > 20:
                preview += f", ... +{len(missing_params) - 20} more"
            messages.append(f"Missing trainable params: {preview}")
        if duplicate_params:
            preview = ", ".join(
                f"{name}(groups {first}->{second})" for name, first, second in duplicate_params[:20]
            )
            if len(duplicate_params) > 20:
                preview += f", ... +{len(duplicate_params) - 20} more"
            messages.append(f"Duplicate optimizer params: {preview}")
        raise RuntimeError("\n".join(messages))
    
    def get_optimizer_scheduler_dict(self, optimizer_parameters):
        optimizer = self.get_optimizer(optimizer_parameters)
        self._validate_optimizer_param_coverage(optimizer)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        # lr_schedule will work each step
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def _alpha_optimizer_params(self):
        if isinstance(self.model.alpha, nn.ParameterList):
            return [p for p in self.model.alpha.parameters() if p.requires_grad]
        return [self.model.alpha] if self.model.alpha.requires_grad else []

    def _configure_muon_adamw_optimizer(self):
        """
        Muon + AdamW 混合优化器 (复用 nanochat/optim.py 的 MuonAdamW)

        参数分组策略 (参考 NanoChat gpt.py:setup_optimizer):
        - alpha: AdamW, 高 LR (mcmc_step_size_lr_multiplier × peak_lr), 无 weight decay [EBT 特有]
        - embeddings: AdamW, 独立绝对 LR (对齐 NanoChat embedding_lr), 无 weight decay
        - vocab_to_embed: AdamW, 独立绝对 LR (EBT 特有, 保守), 无 weight decay
        - transformer / TF-head 标量 (ndim < 2): AdamW, 独立绝对 LR, 无 weight decay
        - transformer / TF-head 矩阵 (ndim >= 2): Muon, 按 shape 分组 (Muon 要求同组参数 shape 相同)
        - 可选: post_update_state_unembed 的 TF-head 矩阵可通过
          --post_update_state_tf_head_adamw 放入 AdamW，避免大 vocab head 进入 Muon

        LR 设计原理:
        - embedding 不在 MCMC 循环内, 梯度行为与 NanoChat 一致, 可用高 LR
        - vocab_to_embed 在 MCMC 循环内 (autograd.grad create_graph=True), 二阶梯度, 需保守
        - transformer scalar (RMSNorm) 在 MCMC 循环内, 需适度保守
        - 当 adamw_*_lr > 0 时使用绝对 LR, 否则 fallback 到 peak_lr × mult

        注意: 使用 MuonAdamW (单 GPU 版), 不用 DistMuonAdamW,
        因为 PL DDP 已经处理梯度同步, DistMuonAdamW 自己管理分布式通信会冲突。
        """
        from nanochat.optim import MuonAdamW

        # --- Muon 超参数 ---
        muon_lr = getattr(self.hparams, 'muon_lr', 0.02)
        muon_momentum = getattr(self.hparams, 'muon_momentum', 0.95)
        muon_ns_steps = getattr(self.hparams, 'muon_ns_steps', 5)
        muon_beta2 = getattr(self.hparams, 'muon_beta2', 0.95)
        adam_betas = (self.hparams.beta1, self.hparams.beta2)

        # --- AdamW LR: 绝对值 or fallback to peak_lr × mult ---
        adamw_embedding_lr = getattr(self.hparams, 'adamw_embedding_lr', -1)
        adamw_vocab_to_embed_lr = getattr(self.hparams, 'adamw_vocab_to_embed_lr', -1)
        adamw_scalar_lr = getattr(self.hparams, 'adamw_scalar_lr', -1)
        use_dmodel_scaling = getattr(self.hparams, 'adamw_dmodel_lr_scaling', False)

        # dmodel scaling: lr × (dim/768)^-0.5 (参考 NanoChat gpt.py:362)
        dmodel_scale = 1.0
        if use_dmodel_scaling:
            model_dim = self.hparams.embedding_dim
            dmodel_scale = (model_dim / 768) ** -0.5
            print(f"[Muon+AdamW] dmodel LR scaling: (dim={model_dim}/768)^-0.5 = {dmodel_scale:.4f}")

        # 计算各组 LR
        if adamw_embedding_lr > 0:
            embedding_lr = adamw_embedding_lr * dmodel_scale
        else:
            embedding_lr_mult = getattr(self.hparams, 'embedding_lr_mult', 0.3)
            embedding_lr = self.hparams.peak_learning_rate * embedding_lr_mult

        if adamw_vocab_to_embed_lr > 0:
            vocab_to_embed_lr = adamw_vocab_to_embed_lr * dmodel_scale
        else:
            vocab_to_embed_lr_mult = getattr(self.hparams, 'vocab_to_embed_lr_mult', 0.1)
            vocab_to_embed_lr = self.hparams.peak_learning_rate * vocab_to_embed_lr_mult

        if adamw_scalar_lr > 0:
            scalar_lr = adamw_scalar_lr * dmodel_scale
        else:
            scalar_lr_mult = getattr(self.hparams, 'scalar_lr_mult', 0.5)
            scalar_lr = self.hparams.peak_learning_rate * scalar_lr_mult

        alpha_lr = self.hparams.mcmc_step_size_lr_multiplier * self.hparams.peak_learning_rate

        # --- 参数收集 ---
        alpha_params = self._alpha_optimizer_params()
        embedding_params = list(self.model.embeddings.parameters())

        vocab_to_embed_params = []
        if hasattr(self.model, 'vocab_to_embed') and self.model.vocab_to_embed is not None:
            vocab_to_embed_params = list(self.model.vocab_to_embed.parameters())

        # VE 参数单独收集，分配给 AdamW (不能放入 Muon)
        ve_embed_params = []
        ve_gate_params = []
        energy_head_matrix_params = []
        transformer_matrix_params = []
        transformer_scalar_params = []
        pre_update_hidden_unembed = getattr(self.hparams, 'tf_head_type', '') == 'pre_update_hidden_unembed'
        for name, param in self.model.transformer.named_parameters():
            if not param.requires_grad:
                continue
            clean_name = name.replace('_orig_mod.', '')
            if 'value_embeds.' in clean_name:
                ve_embed_params.append(param)
            elif 've_gate.' in clean_name:
                ve_gate_params.append(param)
            elif pre_update_hidden_unembed and clean_name.startswith('final_layer.') and param.ndim >= 2:
                # pre_update_hidden_unembed takes CE logits from the trunk hidden
                # before the MCMC update. With the default inter-step detach, the
                # scalar energy head can legitimately receive no CE gradient.
                # Muon stacks every grad in a shape group and cannot handle None.
                energy_head_matrix_params.append(param)
            elif param.ndim >= 2:
                transformer_matrix_params.append(param)
            else:
                transformer_scalar_params.append(param)
        tf_head_matrix_params, tf_head_scalar_params = self._split_tf_head_params()
        post_state_tf_head_adamw = (
            getattr(self.hparams, 'tf_head_type', '') == 'post_update_state_unembed'
            and bool(getattr(self.hparams, 'post_update_state_tf_head_adamw', False))
        )
        tf_head_matrix_adamw_params = tf_head_matrix_params if post_state_tf_head_adamw else []
        tf_head_matrix_muon_params = [] if post_state_tf_head_adamw else tf_head_matrix_params

        # --- 构建 param_groups ---
        param_groups = []

        # AdamW groups
        if alpha_params:
            param_groups.append(dict(
                kind='adamw', params=alpha_params,
                lr=alpha_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        if embedding_params:
            param_groups.append(dict(
                kind='adamw', params=embedding_params,
                lr=embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        if vocab_to_embed_params:
            param_groups.append(dict(
                kind='adamw', params=vocab_to_embed_params,
                lr=vocab_to_embed_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        scalar_params = transformer_scalar_params + tf_head_scalar_params
        if scalar_params:
            param_groups.append(dict(
                kind='adamw', params=scalar_params,
                lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        # VE embedding 参数: AdamW, 使用 embedding_lr (与 NanoChat 一致)
        if ve_embed_params:
            param_groups.append(dict(
                kind='adamw', params=ve_embed_params,
                lr=embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        # VE gate 参数: AdamW, 使用 scalar_lr
        if ve_gate_params:
            param_groups.append(dict(
                kind='adamw', params=ve_gate_params,
                lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        if energy_head_matrix_params:
            param_groups.append(dict(
                kind='adamw', params=energy_head_matrix_params,
                lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        if tf_head_matrix_adamw_params:
            param_groups.append(dict(
                kind='adamw', params=tf_head_matrix_adamw_params,
                lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))

        # Muon groups: 按 shape 分组 (Muon 要求同组参数 shape 相同用于 stack)
        shape_groups = {}
        muon_matrix_params = transformer_matrix_params + tf_head_matrix_muon_params
        for p in muon_matrix_params:
            shape_groups.setdefault(p.shape, []).append(p)

        for shape in sorted(shape_groups.keys()):
            group_params = shape_groups[shape]
            param_groups.append(dict(
                kind='muon', params=group_params,
                lr=muon_lr, momentum=muon_momentum,
                ns_steps=muon_ns_steps, beta2=muon_beta2,
                weight_decay=self.hparams.weight_decay,
            ))

        # --- 创建优化器 ---
        # PL 调用 optimizer.step(closure=closure), 但 MuonAdamW.step() 不接受 closure 参数
        # 包装一下使其兼容 PL 的调用约定
        use_cpu_offload = getattr(self.hparams, 'cpu_offload_optimizer', False)
        if use_cpu_offload:
            class PLMuonAdamW(MuonAdamW):
                """MuonAdamW + CPU offload: AdamW 和 Muon 优化器状态均存放在 CPU。"""

                @torch.no_grad()
                def step(self, closure=None):
                    if closure is not None:
                        with torch.enable_grad():
                            closure()

                    # 遍历所有 param group，按 kind 分别处理
                    for group in self.param_groups:
                        kind = group.get('kind')

                        if kind == 'adamw':
                            # AdamW: 逐参数搬运 exp_avg / exp_avg_sq
                            for p in group['params']:
                                if p.grad is None:
                                    continue
                                state = self.state[p]
                                if not state:
                                    continue
                                for k in ('exp_avg', 'exp_avg_sq'):
                                    if k in state and state[k].device.type == 'cpu':
                                        state[k] = state[k].to(p.device, non_blocking=False)

                        elif kind == 'muon':
                            # Muon: group-level buffer 存在 params[0] 的 state 里
                            if not group['params']:
                                continue
                            p0 = group['params'][0]
                            state = self.state[p0]
                            if not state:
                                continue
                            for k in ('momentum_buffer', 'second_momentum_buffer'):
                                if k in state and state[k].device.type == 'cpu':
                                    state[k] = state[k].to(p0.device, non_blocking=False)

                    # 执行实际的优化器 step（fused kernel 要求 state 在 GPU 上）
                    super().step()

                    # step 完成后，将所有 state 搬回 CPU
                    for group in self.param_groups:
                        kind = group.get('kind')

                        if kind == 'adamw':
                            for p in group['params']:
                                state = self.state[p]
                                for k in ('exp_avg', 'exp_avg_sq'):
                                    if k in state and state[k].device.type != 'cpu':
                                        cpu_t = state[k].to('cpu', non_blocking=False)
                                        del state[k]
                                        state[k] = cpu_t

                        elif kind == 'muon':
                            if not group['params']:
                                continue
                            p0 = group['params'][0]
                            state = self.state[p0]
                            for k in ('momentum_buffer', 'second_momentum_buffer'):
                                if k in state and state[k].device.type != 'cpu':
                                    cpu_t = state[k].to('cpu', non_blocking=False)
                                    del state[k]
                                    state[k] = cpu_t

                    torch.cuda.synchronize()
        else:
            class PLMuonAdamW(MuonAdamW):
                """MuonAdamW wrapper compatible with PyTorch Lightning's optimizer.step(closure=closure)."""
                @torch.no_grad()
                def step(self, closure=None):
                    if closure is not None:
                        with torch.enable_grad():
                            closure()
                    super().step()

        optimizer = PLMuonAdamW(param_groups)
        self._validate_optimizer_param_coverage(optimizer)

        # 设置 initial_lr (PL LR scheduler 需要)
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']

        # --- LR Scheduler ---
        lr_scheduler = self.get_lr_scheduler(optimizer)

        # --- 日志 ---
        num_tf_head_matrix_params = sum(p.numel() for p in tf_head_matrix_params)
        num_tf_head_scalar_params = sum(p.numel() for p in tf_head_scalar_params)
        num_tf_head_params = num_tf_head_matrix_params + num_tf_head_scalar_params
        num_tf_head_matrix_muon_params = sum(p.numel() for p in tf_head_matrix_muon_params)
        num_tf_head_matrix_adamw_params = sum(p.numel() for p in tf_head_matrix_adamw_params)
        num_muon_params = sum(p.numel() for p in muon_matrix_params)
        num_ve_params = (
            sum(p.numel() for p in ve_embed_params) +
            sum(p.numel() for p in ve_gate_params)
        )
        num_adamw_params = (
            sum(p.numel() for p in alpha_params) +
            sum(p.numel() for p in embedding_params) +
            sum(p.numel() for p in vocab_to_embed_params) +
            sum(p.numel() for p in transformer_scalar_params) +
            num_tf_head_scalar_params +
            num_ve_params +
            sum(p.numel() for p in energy_head_matrix_params)
        )
        total_optimized_params = num_muon_params + num_adamw_params
        print(f"=" * 80)
        print(f"[Muon+AdamW] 混合优化器已启用:")
        print(f"  Muon groups: {len(shape_groups)} (按 shape 分组)")
        print(f"  Muon params: {num_muon_params:,} ({num_muon_params/total_optimized_params*100:.1f}%)")
        print(f"  AdamW params: {num_adamw_params:,} ({num_adamw_params/total_optimized_params*100:.1f}%)")
        if num_tf_head_params > 0:
            print(
                f"  TF head params: {num_tf_head_params:,} "
                f"(matrix/Muon: {num_tf_head_matrix_muon_params:,}, "
                f"matrix/AdamW: {num_tf_head_matrix_adamw_params:,}, "
                f"scalar/AdamW: {num_tf_head_scalar_params:,})"
            )
        if num_ve_params > 0:
            print(f"  VE params: {num_ve_params:,} (AdamW, embedding_lr)")
        if energy_head_matrix_params:
            print(
                "  Energy final-layer matrix params: "
                f"{sum(p.numel() for p in energy_head_matrix_params):,} "
                "(AdamW for pre_update_hidden_unembed)"
            )
        print(f"  Muon LR: {muon_lr}, momentum: {muon_momentum}, ns_steps: {muon_ns_steps}, beta2: {muon_beta2}")
        print(f"  Alpha LR: {alpha_lr} (AdamW) [EBT 特有]")
        print(f"  Embedding LR: {embedding_lr} (AdamW)")
        print(f"  vocab_to_embed LR: {vocab_to_embed_lr} (AdamW) [EBT 特有, MCMC 内部]")
        print(f"  Scalar LR: {scalar_lr} (AdamW)")
        if use_dmodel_scaling:
            print(f"  dmodel scaling: {dmodel_scale:.4f} (dim={self.hparams.embedding_dim})")
        for shape, params in sorted(shape_groups.items()):
            print(f"  Muon group shape={shape}: {len(params)} params")
        print(f"=" * 80)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def configure_optimizers_nlp(self):
        if self.hparams.model_name == "ebt":
            # Muon + AdamW 混合优化器 - 通过 --optimizer muon_adamw 启用
            use_muon = getattr(self.hparams, 'optimizer', 'adamw') == 'muon_adamw'

            if use_muon:
                return self._configure_muon_adamw_optimizer()

            # Option 1: 分层学习率 - 通过 --layered_lr 启用
            use_layered_lr = getattr(self.hparams, 'layered_lr', False)

            if use_layered_lr:
                # 分层参数组 (参考 NanoChat base_train.py)
                alpha_param = self._alpha_optimizer_params()
                embedding_params = list(self.model.embeddings.parameters())

                # vocab_to_embed 参数 (类似 unembedding)
                vocab_to_embed_params = []
                if hasattr(self.model, 'vocab_to_embed') and self.model.vocab_to_embed is not None:
                    vocab_to_embed_params = list(self.model.vocab_to_embed.parameters())

                # Transformer 参数分类: 矩阵 vs 标量/向量
                transformer_matrix_params = []
                transformer_scalar_params = []
                for name, param in self.model.transformer.named_parameters():
                    if param.ndim >= 2:  # 矩阵参数 (weights)
                        transformer_matrix_params.append(param)
                    else:  # 标量/向量参数 (biases, layer norms, etc.)
                        transformer_scalar_params.append(param)
                tf_head_matrix_params, tf_head_scalar_params = self._split_tf_head_params()

                # 学习率倍数 (可通过命令行参数覆盖)
                embedding_lr_mult = getattr(self.hparams, 'embedding_lr_mult', 0.3)
                vocab_to_embed_lr_mult = getattr(self.hparams, 'vocab_to_embed_lr_mult', 0.1)
                scalar_lr_mult = getattr(self.hparams, 'scalar_lr_mult', 0.5)

                optimizer_parameters = [
                    # Alpha: 高学习率，无 weight decay
                    {'params': alpha_param, 'weight_decay': 0.0,
                     'lr': self.hparams.mcmc_step_size_lr_multiplier * self.hparams.peak_learning_rate},
                    # Embedding: 中等学习率，无 weight decay
                    # {'params': embedding_params, 'weight_decay': self.hparams.weight_decay,
                    {'params': embedding_params, 'weight_decay': 0.0,
                     'lr': self.hparams.peak_learning_rate * embedding_lr_mult},
                    # vocab_to_embed: 较低学习率
                    # {'params': vocab_to_embed_params, 'weight_decay': self.hparams.weight_decay,
                    {'params': vocab_to_embed_params, 'weight_decay': 0.0,
                     'lr': self.hparams.peak_learning_rate * vocab_to_embed_lr_mult},
                    # Transformer 矩阵: 主学习率
                    {'params': transformer_matrix_params + tf_head_matrix_params, 'weight_decay': self.hparams.weight_decay,
                     'lr': self.hparams.peak_learning_rate},
                    # Transformer 标量: 较高学习率，无 weight decay
                    {'params': transformer_scalar_params + tf_head_scalar_params, 'weight_decay': 0.0,
                     'lr': self.hparams.peak_learning_rate * scalar_lr_mult},
                ]

                # 过滤空参数组
                optimizer_parameters = [p for p in optimizer_parameters if len(p['params']) > 0]

                print(f"[Option 1] 分层学习率已启用:")
                print(f"  - Alpha LR: {self.hparams.mcmc_step_size_lr_multiplier * self.hparams.peak_learning_rate}")
                print(f"  - Embedding LR: {self.hparams.peak_learning_rate * embedding_lr_mult}")
                print(f"  - vocab_to_embed LR: {self.hparams.peak_learning_rate * vocab_to_embed_lr_mult}")
                print(f"  - Transformer Matrix LR: {self.hparams.peak_learning_rate}")
                print(f"  - Transformer Scalar LR: {self.hparams.peak_learning_rate * scalar_lr_mult}")
                if tf_head_matrix_params or tf_head_scalar_params:
                    print(
                        f"  - TF head params: "
                        f"{sum(p.numel() for p in tf_head_matrix_params) + sum(p.numel() for p in tf_head_scalar_params):,}"
                    )
            else:
                # 原始实现
                alpha_param = self._alpha_optimizer_params()
                other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha'])]
                assert len(other_params) > 1, "Could not gather model params correctly please investigate"

                optimizer_parameters = []
                if alpha_param:
                    optimizer_parameters.append(
                        {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate}
                    )
                optimizer_parameters.append(
                    {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}
                )

            return self.get_optimizer_scheduler_dict(optimizer_parameters)
            
        elif self.hparams.model_name == "baseline_transformer":
            all_params = [param for _, param in self.model.named_parameters()]
            optimizer_parameters = [
                {'params': all_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
        
        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")

        
    def configure_optimizers_vid(self):
        if self.hparams.model_name == "ebt":
            alpha_param = self._alpha_optimizer_params()
            encoder_params = list(self.model.image_encoder.parameters())
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha', 'image_encoder'])]
            assert len(other_params) > 1, "Could not gather model params correctly please investigate"
            
            optimizer_parameters = [
                {'params': encoder_params, 'weight_decay': 0.0, 'lr': 0.0},
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            if alpha_param:
                optimizer_parameters.insert(0, {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate})  # No weight decay for alpha
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
            
        elif self.hparams.model_name == "baseline_transformer":
            encoder_params = list(self.model.image_encoder.parameters())
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['image_encoder'])]

            optimizer_parameters = [
                {'params': encoder_params, 'weight_decay': 0, 'lr': 0},
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
        
        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")
        
    def configure_optimizers_img(self):
        if self.hparams.model_name == "ebt":
            alpha_param = self._alpha_optimizer_params()
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha', 'image_encoder', 'text_encoder'])]
            assert len(other_params) > 1, "Could not gather model params correctly please investigate"
            
            optimizer_parameters = [
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate} # Weight decay for other parameters
            ]
            if alpha_param:
                optimizer_parameters.insert(0, {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate})  # No weight decay for alpha
            
            # if self.hparams.image_task == "t2i": # do this bc other models wont have these 'sub' models
            #     image_encoder_params = list(self.model.image_encoder.parameters())
            #     optimizer_parameters.insert(1, {'params': image_encoder_params, 'weight_decay': 0, 'lr': 0})
            #     text_encoder_params = list(self.model.text_encoder.parameters())
            #     optimizer_parameters.insert(2, {'params': text_encoder_params, 'weight_decay': 0, 'lr': 0})
            
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
            
        # elif self.hparams.model_name == "dit":
        #     other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['image_encoder', 'text_encoder'])]

        #     optimizer_parameters = [
        #         {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
        #     ]
        #     if self.hparams.image_task == "t2i":
        #         image_encoder_params = list(self.model.image_encoder.parameters())
        #         optimizer_parameters.insert(0, {'params': image_encoder_params, 'weight_decay': 0, 'lr': 0})
        #         text_encoder_params = list(self.model.text_encoder.parameters())
        #         optimizer_parameters.insert(1, {'params': text_encoder_params, 'weight_decay': 0, 'lr': 0})
            
        #     return self.get_optimizer_scheduler_dict(optimizer_parameters)
        
        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")

    # def create_full_ds(self):
    #     if self.hparams.dataset_name == "coco_tiny":
    #         self.full_ds = COCOTinyDataset(self.hparams, split = "train", transform = self.transform)
    #     if self.hparams.dataset_name == "ucf101":
    #         self.full_ds = UCF101Dataset(self.hparams, split = "train", transform = self.transform)
    #     elif self.hparams.dataset_name == "vid_synthetic":
    #         self.full_ds = VIDSyntheticDataset(self.hparams)
    #     elif self.hparams.dataset_name == "pajama":
    #         self.full_ds = RedPajamaDataset(self.hparams)
    #     elif self.hparams.dataset_name == 'fineweb':
    #         self.full_ds = FineWebDataset(self.hparams)
    #     elif "bigbench" in self.hparams.dataset_name:
    #         x = self.hparams.dataset_name
    #         self.full_ds = BigBenchDataset(self.hparams, "train", x[x.find('_') + 1 :])
    #     elif self.hparams.dataset_name == "planbench":
    #         self.full_ds = PlanBenchDataset(self.hparams, split = "train")
    #     elif self.hparams.dataset_name == "nlp_synthetic":
    #         self.full_ds = NLPSyntheticDataset(self.hparams)
    #     elif self.hparams.dataset_name == "aggregate": # aggregate VID dataset combining ssv2 and k400
    #         self.full_ds = AggregateDataset(self.hparams, split = "train", transform = self.transform, normal_lookup=self.normal_lookup)
    #     else:
    #         raise NotImplementedError(f"haven't implemented dataset {self.hparams.dataset_name} full_ds yet")

    # def setup(self, stage=None):
    #     # NOTE when passing stage into datasets/dataloaders use string rep not the stage param from this func since is a PL enum
    #     # Assign train/val datasets for use in dataloaders 
    #     assert self.hparams.test_split_pct == 0, "Haven't implemented nonzero value for test_split_pct yet"

    #     if stage == "fit":
    #         # all of these conditions need to have manual split
    #         if self.hparams.dataset_name in ["coco_tiny", "ucf101", "vid_synthetic", "pajama", "fineweb", "bigbench", "planbench", "nlp_synthetic"]:
    #             self.create_full_ds()
    #             train_samples = int(len(self.full_ds) * (1 - self.hparams.validation_split_pct))
    #             valid_samples = len(self.full_ds) - train_samples
    #             self.train_ds, self.val_ds = random_split(self.full_ds, [train_samples, valid_samples])
    #         elif self.hparams.dataset_name == "aggregate":
    #             self.create_full_ds()
    #             self.train_ds, self.val_ds = self.full_ds.train_val_split(val_split_pct = self.hparams.validation_split_pct)
    #         elif self.hparams.dataset_name == 'k400':
    #             self.train_ds = Kinetics400Dataset(self.hparams, split = 'train', transform = self.transform)
    #             self.val_ds = Kinetics400Dataset(self.hparams, split = 'val', transform = self.transform)
    #         elif self.hparams.dataset_name in ('something' , 'smth'):
    #             self.train_ds = SomethingDataset(self.hparams, split = 'train', transform = self.transform)
    #             self.val_ds = SomethingDataset(self.hparams, split = 'val', transform = self.transform)
    #         elif self.hparams.dataset_name in ('imagenet' , 'imagenet1k'):
    #             self.train_ds = ImageNetDataset(self.hparams, split = 'train', transform = self.transform)
    #             self.val_ds = ImageNetDataset(self.hparams, split = 'val', transform = self.transform)
    #         elif self.hparams.dataset_name == 'coco_medium':
    #             self.train_ds = COCOMediumDataset(self.hparams, split = "train", transform = self.transform)
    #             self.val_ds = COCOMediumDataset(self.hparams, split = "validation", transform = self.transform)
    #         elif self.hparams.dataset_name == "gsm8k":
    #             self.train_ds = GSM8KDataset(self.hparams, split = "train")
    #             self.val_ds = GSM8KDataset(self.hparams, split = "test") # no val just test https://huggingface.co/datasets/openai/gsm8k
    #         elif self.hparams.dataset_name == "ai2arc":
    #             self.train_ds = AI2ArcDataset(self.hparams, split = 'train')
    #             self.val_ds = AI2ArcDataset(self.hparams, split = 'validation')
    #         elif self.hparams.dataset_name == "squad":
    #             self.train_ds = SQuADDataset(self.hparams, split = 'train')
    #             self.val_ds = SQuADDataset(self.hparams, split = 'validation')
    #         else:
    #             raise NotImplementedError("Haven't implemented this dataset yet")
    #         print(f"{self.hparams.dataset_name} length of train_dataset: {len(self.train_ds)} and val_dataset: {len(self.val_ds)}")
            
    #     # Assign test dataset for use in dataloader(s)
    #     elif stage == "test":
    #         if self.hparams.dataset_name == "ucf101":
    #             self.test_ds = UCF101Dataset(self.hparams, split = "test", transform = self.transform)
    #         elif self.hparams.dataset_name in ('kinetics400' , 'k400'):
    #             self.test_ds = Kinetics400Dataset(self.hparams, split = "test", transform = self.transform)
    #         elif self.hparams.dataset_name in ('something' , 'smth'):
    #             self.test_ds = SomethingDataset(self.hparams, split = "test", transform = self.transform)
    #         elif self.hparams.dataset_name in ('imagenet' , 'imagenet1k'):
    #             self.test_ds = ImageNetDataset(self.hparams, split = "test", transform = self.transform)
    #         elif self.hparams.dataset_name == 'aggregate':
    #             self.test_ds = AggregateDataset(self.hparams, split = "test", transform = self.transform)
    #         elif self.hparams.dataset_name == "coco_tiny":
    #             self.test_ds = COCOTinyDataset(self.hparams, split = "validation", transform = self.transform) # use validation since there is no test split, splitted train into val
    #         elif self.hparams.dataset_name == "coco_medium":
    #             self.test_ds = COCOMediumDataset(self.hparams, split = "test", transform = self.transform)
    #         elif self.hparams.dataset_name == "pajama": # for now am assuming test split == val split, so dont save train or full ds here, just to get val split
    #             full_ds = RedPajamaDataset(self.hparams)
    #             train_samples = int(len(full_ds) * (1 - self.hparams.validation_split_pct))
    #             test_samples = len(full_ds) - train_samples
    #             _, self.test_ds = random_split(full_ds, [train_samples, test_samples])
    #         elif self.hparams.dataset_name == "fineweb":
    #             raise NotImplementedError(f"haven't implemented fineweb dataset test split yet")
    #         elif "bigbench" in self.hparams.dataset_name:
    #             x = self.hparams.dataset_name
    #             self.test_ds = BigBenchDataset(self.hparams, "validation", x[x.find('_') + 1 :]) #use val for testing as Bigbench only has train/val
    #         elif self.hparams.dataset_name == "gsm8k":
    #             self.test_ds = GSM8KDataset(self.hparams, split="test")
    #         elif self.hparams.dataset_name == "lambada":
    #             self.test_ds = LambadaDataset(self.hparams, split="test")
    #         elif self.hparams.dataset_name == "squad":
    #             self.test_ds = SQuADDataset(self.hparams, split="validation") # no test split use val
    #         elif self.hparams.dataset_name == "planbench":
    #             raise NotImplementedError(f"no planbench test split")
    #         elif self.hparams.dataset_name == "ai2arc":
    #             self.test_ds = AI2ArcDataset(self.hparams, split = "test")
    #         else:
    #             raise NotImplementedError("haven't implemented this dataset yet")
    #         print(f"{self.hparams.dataset_name} length of test_ds: {len(self.test_ds)}")
    #     else:
    #         raise ValueError(f"Unknown stage: {stage}, please investigate")
    
    def get_collate_fn(self):
        collate_fn = None if not self.hparams.modality == "NLP" else NLP_HF_Collator(self.hparams) #NOTE this assumes all modalities except NLP DONT have collator, may not be true in the future
        if self.hparams.dataset_name == "nlp_synthetic": #NOTE this is a hack to get around the fact that synthetic dataset cant return real text and thus cant use collate_fn
            collate_fn = None
        return collate_fn
    
    def  train_dataloader(self):
        # Use tokenizer_obj for dataloader
        tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else self.hparams.tokenizer
        base_train_dataset = getattr(self.hparams, 'base_train_dataset', 'fineweb')
        base_train_data_dir = getattr(self.hparams, 'base_train_data_dir', '')
        base_train_data_dir = base_train_data_dir or None

        # 从 checkpoint 恢复的 dataloader 位置（只用一次）
        resume_state = getattr(self, '_dataloader_resume_state', None)
        self._dataloader_resume_state = None

        if getattr(self.hparams, 'dataset_name', 'nanochat') == 'nanochat_sft':
            _sft_trainer_debug(
                f"train_dataloader_start batch_size={self.hparams.batch_size_per_device} "
                f"max_len={self.hparams.context_length} max_iter={self.hparams.max_steps * self.hparams.accumulate_grad_batches}"
            )
            train_dataloader = generate_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.max_steps * self.hparams.accumulate_grad_batches,
                split="train",
                device=self.device,
                resume_state_dict=resume_state,
            )
            _sft_trainer_debug("train_dataloader_done")
        elif getattr(self.hparams, 'dataset_name', 'nanochat') == 'sudoku_sft':
            from openebm.elm.data.sudoku_dataset import generate_sudoku_sft_dataloader
            train_dataloader = generate_sudoku_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.max_steps * self.hparams.accumulate_grad_batches,
                split="train",
                device=self.device,
                resume_state_dict=resume_state,
            )
        elif getattr(self.hparams, 'dataset_name', 'nanochat') == 'sudoku_mixed':
            from openebm.elm.data.sudoku_mixed_dataset import generate_sudoku_mixed_dataloader
            # v3: derive bucket weights from --sudoku_difficulty_schedule (if any). The
            # AdaptiveRatioCallback will mutate dataset.sudoku_ds.bucket_weights per
            # phase so only the warmup-phase initial value is set here.
            blank_weight = float(getattr(self.hparams, 'sudoku_blank_loss_weight', 1.0))
            difficulty_sched = getattr(self.hparams, 'sudoku_difficulty_schedule', 'fixed')
            initial_bucket_weights = _v3_initial_bucket_weights(difficulty_sched)
            train_dataloader = generate_sudoku_mixed_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.max_steps * self.hparams.accumulate_grad_batches,
                split="train",
                device=self.device,
                resume_state_dict=resume_state,
                sudoku_ratio=getattr(self.hparams, 'sudoku_ratio', 0.6),
                blank_loss_weight=blank_weight,
                difficulty_bucket_weights=initial_bucket_weights,
            )
        else:
            train_dataloader = generate_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.max_steps * self.hparams.accumulate_grad_batches, # 显示的1个epoch对应设置的self.hparams.max_steps个训练步数
                split="train",
                device=self.device,
                resume_state_dict=resume_state,
                base_train_dataset=base_train_dataset,
                base_train_data_dir=base_train_data_dir,
            )
        return train_dataloader

    def val_dataloader(self):
        # IMPORTANT: NanoChat dataset split information
        # The train/val split is HARDCODED in nanochat/dataloader.py line 37:
        # - Training: parquet_paths[:-1] (all files except the last one)
        # - Validation: parquet_paths[-1:] (only the last file)
        # With 370 total shards, this gives 369 train + 1 val (0.27% validation)
        # The --validation_split_pct parameter does NOT control this split!

        # Use tokenizer_obj for dataloader
        tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else self.hparams.tokenizer
        base_train_dataset = getattr(self.hparams, 'base_train_dataset', 'fineweb')
        base_train_data_dir = getattr(self.hparams, 'base_train_data_dir', '')
        base_train_data_dir = base_train_data_dir or None

        if getattr(self.hparams, 'dataset_name', 'nanochat') == 'nanochat_sft':
            _sft_trainer_debug(
                f"val_dataloader_start batch_size={self.hparams.batch_size_per_device} "
                f"max_len={self.hparams.context_length} max_iter={self.hparams.val_steps}"
            )
            val_dataloader = generate_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
            )
            _sft_trainer_debug("val_dataloader_done")
        elif getattr(self.hparams, 'dataset_name', 'nanochat') == 'sudoku_sft':
            from openebm.elm.data.sudoku_dataset import generate_sudoku_sft_dataloader
            val_dataloader = generate_sudoku_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
            )
        elif getattr(self.hparams, 'dataset_name', 'nanochat') == 'sudoku_mixed':
            # 方案A: 验证阶段拆分为两个独立 dataloader,
            # 分别评估 Sudoku v2 与 nanochat SFT, 再按 sudoku_ratio 加权得到 mixed 指标
            from openebm.elm.data.sudoku_dataset_v2 import generate_sudoku_sft_v2_dataloader
            from openebm.elm.dataset_sft import generate_sft_dataloader
            sudoku_val_loader = generate_sudoku_sft_v2_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
                augment=False,
            )
            sft_val_loader = generate_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
            )
            # 顺序对应 self._val_loader_sources, validation_step 用 dataloader_idx 区分
            self._val_loader_sources = ['sudoku', 'sft']
            val_dataloader = [sudoku_val_loader, sft_val_loader]
        else:
            val_dataloader = generate_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
                resume_state_dict=None,
                base_train_dataset=base_train_dataset,
                base_train_data_dir=base_train_data_dir,
            )

        return val_dataloader

    def test_dataloader(self):
        # For inference mode with specific datasets, use DataLoader with collate_fn
        if self.hparams.execution_mode == "inference" and self.hparams.dataset_name == "gsm8k":
            test_ds = GSM8KDataset(self.hparams, split="test")
            return DataLoader(
                test_ds,
                batch_size=self.hparams.batch_size_per_device,
                num_workers=0,  # Keep 0 for simplicity
                collate_fn=self.get_collate_fn(),
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
        elif self.hparams.execution_mode == "inference" and self.hparams.dataset_name == "nanochat_shard_eval":
            # Custom NanoChat shard evaluation dataset
            from openebm.elm.dataset_nanochat_eval import NanoChatShardEvalDataset, collate_fn_nanochat_eval

            # Parse shard indices from comma-separated string
            shard_indices_str = getattr(self.hparams, 'eval_shard_indices', '0,15')
            if isinstance(shard_indices_str, str):
                shard_indices = [int(x.strip()) for x in shard_indices_str.split(',')]
            else:
                shard_indices = shard_indices_str

            max_samples_per_shard = getattr(self.hparams, 'max_samples_per_shard', 50)
            enable_generation = getattr(self.hparams, 'enable_nanochat_generation', True)
            generation_split_ratio = getattr(self.hparams, 'generation_split_ratio', 0.5)
            min_generation_length = getattr(self.hparams, 'min_generation_length', 64)

            test_ds = NanoChatShardEvalDataset(
                tokenizer=self.hparams.tokenizer_obj,
                context_length=self.hparams.context_length,
                shard_indices=shard_indices,
                max_samples_per_shard=max_samples_per_shard,
                enable_generation=enable_generation,
                generation_split_ratio=generation_split_ratio,
                min_generation_length=min_generation_length
            )

            return DataLoader(
                test_ds,
                batch_size=self.hparams.batch_size_per_device,
                num_workers=0,
                collate_fn=collate_fn_nanochat_eval,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
        else:
            # Default: use val_dataloader for pretrain mode
            return self.val_dataloader()
        

    def log_metrics(self, metrics_dict, phase, log_torchmetrics = True):
        # first log torchmetrics if there are any
        if log_torchmetrics and len(self.metrics) > 0:
            phase_dict = {key : value for key, value in self.torchmetrics_dict.items() if phase in key}
            self.log_dict(phase_dict, on_step = False, on_epoch = True) # for these always do on_epoch

        # log all other metrics in metrics_dict
        scalar_metrics = {}
        keys = list(metrics_dict.keys()) # Iterate over a copy of the keys to avoid modification issues during iteration
        for key in keys:
            # 跳过 BPB 相关指标，它们不应通过 Lightning log_dict 上报：
            # - bpb_nats/bpb_bytes: BPB 累积中间量
            # - bpb (非 train 阶段): BPB 是比率指标 (nats/bytes)，
            #   Lightning 的 on_epoch=True 会对 per-batch bpb 做算术平均，这是数学错误的
            #   正确做法是在 on_validation_epoch_end 中从累积 nats/bytes 重新计算
            if key in ('bpb_nats', 'bpb_bytes'):
                continue
            if key == 'bpb' and phase != 'train':
                continue
            if key == 'loss' and phase == 'valid':
                continue
            if key in ('supervised_tokens', 'empty_supervision_batch') and phase != 'train':
                continue

            value = metrics_dict[key]
            # if 'image' in key: # images
            #     image = self.to_pil(value)
            #     wandb_image = wandb.Image(image, mode="RGB")
            #     self.logger.experiment.log({f'{phase}_{key}': wandb_image})

            # elif 'video' in key: # videos
            #     video_np = value.cpu().numpy()
            #     assert video_np.ndim != 5, "video should not include batch dimension, either fix that or add support"
            #     if video_np.shape[1] in [1, 3]:
            #         pass  # Axes are already correct
            #     elif video_np.shape[-1] in [1, 3]:
            #         # If video_np is (frames, height, width, channels), transpose axes
            #         video_np = video_np.transpose(0, 3, 1, 2)
            #     else:
            #         raise ValueError(f"Unexpected video shape: {video_np.shape}")
            #     if video_np.dtype != np.uint8:
            #         video_np = (video_np * 255).astype(np.uint8)
            #     wandb_video = wandb.Video(video_np, fps=4, format="mp4")
            #     self.logger.experiment.log({f'{phase}_{key}': wandb_video})

            if isinstance(value, torch.Tensor) and value.numel() > 1: # histogram
                # Optimize: Log stats instead of Histogram to avoid CPU sync/copy
                self.logger.experiment.log({
                    f"{phase}_{key}_mean": value.detach().mean(),
                    f"{phase}_{key}_std": value.detach().std(),
                })

            elif isinstance(value, torch.Tensor) and value.dim() == 0: # two types of scalar, tensor (here) and int/float (below)
                scalar_metrics[f"{phase}_{key}"] = value.detach()
            elif isinstance(value, (int, float)):
                scalar_metrics[f"{phase}_{key}"] = value
            else:
                raise ValueError(f"unsupported type/format in log_metrics, type:, {type(value)}, key: {key}")

        if scalar_metrics:
            if phase == "train":
                # 训练阶段：on_step=True, on_epoch=False
                # 每个 train step 独立上报，不跨 step 累积。
                self.log_dict(scalar_metrics, sync_dist=True, prog_bar=True,
                              on_step=True, on_epoch=False)
            else:
                # 验证/测试阶段：on_step=False, on_epoch=True
                # Lightning 保证每次 val_loop 是独立的 validation epoch，
                # on_epoch=True 只在当次 val 的 batch 内累积均值，
                # 不会跨多次 val_check_interval 累积（不存在"280次val被平均"的问题）。
                # ModelCheckpoint 在 on_validation_end 查询 epoch-level 指标，
                # 必须用 on_epoch=True 才能让 valid_loss 出现在 returned metrics 里。
                self.log_dict(scalar_metrics, sync_dist=True, prog_bar=True,
                              on_step=False, on_epoch=True)

        # === 增强的调试日志 (仅 train 阶段) ===
        # lr/wd/Alpha 等只在训练步骤记录，避免在 validation 阶段触发
        # "log on epoch level in distributed setting" 的 warning
        if phase == "train" and len(self.trainer.optimizers) > 0:
            optimizer = self.trainer.optimizers[0]

            # 记录所有参数组的学习率
            for i, group in enumerate(optimizer.param_groups):
                group_lr = group['lr']
                group_wd = group.get('weight_decay', 0)
                self.log(f"lr/param_group_{i}", group_lr, prog_bar=False,
                         on_step=True, on_epoch=False)
                self.log(f"wd/param_group_{i}", group_wd, prog_bar=False,
                         on_step=True, on_epoch=False)

            # 主学习率 (最后一个参数组，通常是 transformer 参数)
            current_lr = optimizer.param_groups[-1]['lr']
            self.log("Global_LR", current_lr, on_step=True, on_epoch=False)

            # Alpha 参数的学习率 (第一个参数组)
            if len(optimizer.param_groups) > 1:
                alpha_lr = optimizer.param_groups[0]['lr']
                self.log("Alpha_LR", alpha_lr, on_step=True, on_epoch=False)

        # Alpha (MCMC step size) 值 (仅 train 阶段，避免 validation 阶段 warning)
        if phase == "train" and self.hparams.mcmc_step_size_learnable:
            if isinstance(self.model.alpha, nn.ParameterList):
                for i, p in enumerate(self.model.alpha):
                    self.log(f"Alpha_MCMC_Step_{i}", p.detach(), on_step=True, on_epoch=False, prog_bar=True)
            else:
                self.log("Alpha_MCMC_Step_Size", self.model.alpha.detach(),
                         on_step=True, on_epoch=False, prog_bar=True)

        # Langevin dynamics noise (仅 train 阶段)
        if phase == "train" and self.hparams.langevin_dynamics_noise_learnable:
            self.log("Langevin_dynamics_noise", self.model.langevin_dynamics_noise_std.detach(),
                     on_step=True, on_epoch=False)

        # 训练进度信息 (仅在训练阶段, 仅 rank 0 打印)
        if phase == "train" and hasattr(self, 'trainer') and self.trainer is not None:
            import time as _time

            current_step = self.global_step
            max_steps = self.hparams.max_steps
            progress_pct = 100.0 * current_step / max_steps if max_steps > 0 else 0
            self.log("step", float(current_step), prog_bar=True, on_step=True, on_epoch=False)
            self.log("progress_pct", progress_pct, prog_bar=False, on_step=True, on_epoch=False)

            # GPU 内存使用 (如果可用)
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                self.log("gpu_mem_allocated_gb", gpu_mem_allocated, prog_bar=False,
                         on_step=True, on_epoch=False)
                self.log("gpu_mem_reserved_gb", gpu_mem_reserved, prog_bar=False,
                         on_step=True, on_epoch=False)

            # === 丰富训练日志 (仅 rank 0 打印, 避免 DDP 重复) ===
            if self.trainer.is_global_zero:
                # --- 时间统计 ---
                dt_ms = (getattr(self, '_last_dt', None) or 0.0) * 1000.0
                wall_elapsed = 0.0
                if self._train_start_time is not None:
                    wall_elapsed = _time.time() - self._train_start_time
                total_min = wall_elapsed / 60.0

                # --- LR ratio (相对于 peak_lr) ---
                lrm = 1.0
                if len(self.trainer.optimizers) > 0:
                    opt = self.trainer.optimizers[0]
                    # 取最后一个参数组（通常是 transformer/muon 主参数组）的 lr
                    cur_lr = opt.param_groups[-1]['lr']
                    peak_lr = self.hparams.peak_learning_rate
                    lrm = cur_lr / peak_lr if peak_lr > 0 else 1.0

                # --- tok/sec: tokens processed per second (全局) ---
                # 每个 optimizer step 消耗 tokens = num_gpus × batch_per_device × context_length × grad_accum
                num_gpus = getattr(self.hparams, 'num_gpus', 1)
                tokens_per_step = (num_gpus
                                   * self.hparams.batch_size_per_device
                                   * self.hparams.context_length
                                   * self.hparams.accumulate_grad_batches)
                tok_per_sec = tokens_per_step / (dt_ms / 1000.0) if dt_ms > 0 else 0.0

                # --- MFU (Model FLOP Utilization) ---
                # 参考 PaLM / nanoGPT 计算方式:
                # FLOPs per token ≈ 6 × num_params（前向 + 反向）
                # MFU = actual_tok_per_sec × flops_per_token / peak_flops_per_sec
                # H200 peak bfloat16 FLOPS ≈ 989 TFLOPS per GPU
                try:
                    num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    flops_per_token = 6 * num_params  # forward + backward
                    # Peak FLOPS: H200=989T, A100=312T; 这里保守用 312T/GPU (A100)
                    # peak_flops_per_gpu = 312e12
                    peak_flops_per_gpu = 989e12
                    gpu_peak_flops = num_gpus * peak_flops_per_gpu
                    actual_flops_per_sec = tok_per_sec * flops_per_token
                    mfu = 100.0 * actual_flops_per_sec / gpu_peak_flops if gpu_peak_flops > 0 else 0.0
                except Exception:
                    mfu = 0.0

                # --- Epoch ---
                epoch = self.current_epoch + 1

                # --- ETA ---
                if current_step > 0 and wall_elapsed > 0 and max_steps > 0:
                    steps_remaining = max_steps - current_step
                    sec_per_step = wall_elapsed / current_step
                    eta_min = steps_remaining * sec_per_step / 60.0
                    eta_str = f" | eta: {eta_min:.1f}m"
                else:
                    eta_str = ""

                # --- 当前 loss ---
                loss_val = metrics_dict.get('loss', 0.0)
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.item()

                # --- 最新 valid 指标 ---
                last_valid = getattr(self, '_last_valid_metrics', {})
                callback_metrics = getattr(self.trainer, 'callback_metrics', {})

                def _metric_to_float(metric):
                    if metric is None:
                        return None
                    if isinstance(metric, torch.Tensor):
                        return metric.detach().item()
                    if isinstance(metric, (int, float)):
                        return metric
                    return None

                valid_loss_val = _metric_to_float(
                    callback_metrics.get('valid_loss') if callback_metrics is not None else None
                )
                if valid_loss_val is None:
                    valid_loss_val = last_valid.get('loss', None)
                valid_bpb_val = last_valid.get('bpb', None)
                valid_ppl_val = _metric_to_float(
                    callback_metrics.get('valid_perplexity') if callback_metrics is not None else None
                )
                if valid_ppl_val is None:
                    valid_ppl_val = last_valid.get('perplexity', None)
                valid_str = ""
                if valid_loss_val is not None:
                    valid_str += f" | valid_loss: {valid_loss_val:.4f}"
                if valid_bpb_val is not None:
                    valid_str += f" | valid_bpb: {valid_bpb_val:.4f}"
                if valid_ppl_val is not None:
                    valid_str += f" | valid_ppl: {valid_ppl_val:.2f}"
                # 方案A: 当存在拆分来源时, 额外展示 sudoku / sft 各自指标
                for src in ('sudoku', 'sft'):
                    sub_loss = last_valid.get(f'loss_{src}', None)
                    sub_bpb = last_valid.get(f'bpb_{src}', None)
                    sub_ppl = last_valid.get(f'perplexity_{src}', None)
                    if sub_loss is not None:
                        valid_str += f" | {src}_loss: {sub_loss:.4f}"
                    if sub_bpb is not None:
                        valid_str += f" | {src}_bpb: {sub_bpb:.4f}"
                    if sub_ppl is not None:
                        valid_str += f" | {src}_ppl: {sub_ppl:.2f}"

                # --- 打印 ---
                # alpha_val_str = ""
                # if self.hparams.mcmc_step_size_learnable:
                #     alpha_val = self.model.alpha.detach()
                #     alpha_grad_str = f" grad={self.model.alpha.grad.item():.6f}" if self.model.alpha.grad is not None else " grad=None"
                #     alpha_val_str = f" | alpha: {alpha_val.item():.6f} ({alpha_val.dtype}){alpha_grad_str}"
                print(
                    f"step {current_step:05d}/{max_steps} ({progress_pct:.2f}%) | "
                    f"loss: {loss_val:.6f}"
                    f"{valid_str} | "
                    f"lrm: {lrm:.2f} | "
                    f"dt: {dt_ms:.2f}ms | "
                    f"tok/sec: {tok_per_sec:,.0f} | "
                    f"mfu: {mfu:.2f} | "
                    f"epoch: {epoch} | "
                    f"total time: {total_min:.2f}m"
                    f"{eta_str}",
                    # f"{alpha_val_str}",
                    flush=True,
                )
