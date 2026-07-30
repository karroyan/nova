import json
import os
import random
import shutil
import sys
from argparse import ArgumentParser

import torch
import wandb

# 允许加载旧 checkpoint 中的自定义类
try:
    from nanochat.tokenizer import RustBPETokenizer
    torch.serialization.add_safe_globals([RustBPETokenizer])
except Exception:
    pass

# 抑制 CUDA stream 不匹配警告（恢复训练时的已知问题）
# 仅 torch >= 2.6 提供该 API；旧版本静默跳过。
if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.strategies import DDPStrategy
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
except ImportError:
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.strategies import DDPStrategy
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

from openebm.elm import logger as text_logger
from openebm.elm.disk_aware_checkpoint import DiskAwareCheckpoint
from openebm.elm.eval import nlp_eval_acc
from openebm.elm.trainer import ModelTrainer
from openebm.elm.utils import init_wandb_watch, model_sizes

@rank_zero_only # to ensure only one wandb run is created, if didnt do that then each GPU would create its own wandb run
def setup_wandb(args): 
    import wandb
    if wandb.run is None:
        run = wandb.init(dir="logs/", name=f'{args.run_name}', entity=f'{args.wandb_entity}', project=f'{args.wandb_project}', mode = "offline" if args.wandb_offline else "online") # this is solely used to force wandb to start tracking stdout in logs
        wandb.define_metric("__init", hidden=True)
        return run
    return None


def _bytes_to_gib(num_bytes):
    return num_bytes / (1024 ** 3)


def _checkpoint_file_size_gib(path):
    if path and os.path.isfile(path):
        return _bytes_to_gib(os.path.getsize(path))
    return None


def _largest_existing_checkpoint_gib(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None, None

    largest_size = None
    largest_path = None
    for root, _, files in os.walk(ckpt_dir):
        for filename in files:
            if not filename.endswith(".ckpt"):
                continue
            path = os.path.join(root, filename)
            try:
                size_gib = _bytes_to_gib(os.path.getsize(path))
            except OSError:
                continue
            if largest_size is None or size_gib > largest_size:
                largest_size = size_gib
                largest_path = path
    return largest_size, largest_path


def _estimate_checkpoint_size_gib(args, ckpt_dir, model_trainer):
    candidates = []

    existing_size_gib, existing_path = _largest_existing_checkpoint_gib(ckpt_dir)
    if existing_size_gib is not None:
        candidates.append((existing_size_gib, f"existing checkpoint: {existing_path}"))

    for attr_name, label in (
        ("finetuning_model_ckpt", "finetuning checkpoint"),
        ("resume_training_ckpt", "resume checkpoint"),
    ):
        path = getattr(args, attr_name, "")
        size_gib = _checkpoint_file_size_gib(path)
        if size_gib is not None:
            candidates.append((size_gib, f"{label}: {path}"))

    if candidates:
        return max(candidates, key=lambda item: item[0])

    tensor_bytes = 0
    for value in model_trainer.state_dict().values():
        if torch.is_tensor(value):
            tensor_bytes += value.numel() * value.element_size()

    # Lightning checkpoints also carry optimizer/trainer state. Use a conservative
    # multiplier when no real checkpoint file is available as a reference.
    estimated_gib = max(_bytes_to_gib(tensor_bytes) * 4.0, 0.1)
    return estimated_gib, "estimated from model state_dict"


def _planned_checkpoint_slots(args, save_last):
    slots = 0
    details = []
    warnings = []

    if args.save_top_k_ckpts == -1:
        slots += 1
        details.append("top_k=all (preflight lower bound: 1)")
        warnings.append("save_top_k_ckpts=-1 is unbounded, so exact checkpoint capacity cannot be guaranteed.")
    elif args.save_top_k_ckpts > 0:
        slots += args.save_top_k_ckpts
        details.append(f"top_k={args.save_top_k_ckpts}")

    if save_last:
        slots += 1
        details.append("last=1")

    if args.save_periodic_steps > 0:
        slots += 1
        details.append(f"periodic=1 every {args.save_periodic_steps} steps")

    return slots, details, warnings


def preflight_checkpoint_disk(args, ckpt_dir, save_last, model_trainer, is_rank_zero_process):
    if args.only_test:
        return

    checkpoint_slots, slot_details, warnings = _planned_checkpoint_slots(args, save_last)
    if checkpoint_slots <= 0:
        return

    min_free_gib = max(0.0, float(args.checkpoint_min_free_gb))
    os.makedirs(ckpt_dir, exist_ok=True)

    free_gib = _bytes_to_gib(shutil.disk_usage(ckpt_dir).free)
    ref_ckpt_gib, ref_source = _estimate_checkpoint_size_gib(args, ckpt_dir, model_trainer)
    required_gib = checkpoint_slots * ref_ckpt_gib + min_free_gib

    if is_rank_zero_process:
        print("[Checkpoint Preflight]")
        print(f"  dir: {ckpt_dir}")
        print(f"  planned checkpoint slots: {checkpoint_slots} ({', '.join(slot_details)})")
        print(f"  reference checkpoint size: {ref_ckpt_gib:.2f} GiB ({ref_source})")
        print(f"  required free before training: {required_gib:.2f} GiB = {checkpoint_slots} * {ref_ckpt_gib:.2f} + reserve {min_free_gib:.2f}")
        print(f"  current free: {free_gib:.2f} GiB")
        for warning in warnings:
            print(f"  warning: {warning}")

    if free_gib < required_gib:
        raise RuntimeError(
            "Checkpoint disk preflight failed: "
            f"current free space {free_gib:.2f} GiB is less than required {required_gib:.2f} GiB "
            f"for {checkpoint_slots} checkpoint slots plus {min_free_gib:.2f} GiB reserve. "
            f"checkpoint_dir={ckpt_dir}; reference={ref_source}. "
            "Free disk space or reduce --save_top_k_ckpts/--save_periodic_steps before starting training."
        )

def main(args):
    # --disable_wandb is an alias for --no_wandb
    if args.disable_wandb:
        args.no_wandb = True

    if(args.is_random_seed):
        seed_everything(random.randint(0,1000000), workers=True)
    else:
        seed_everything(33, workers=True) #33 is default
        
    if args.debug_mode:
        args.no_wandb = True
        args.detect_anomaly = True
        args.limit_train_batches = 1

    os.makedirs("./logs", exist_ok=True)

    wandb_logger = None
    if not args.no_wandb: # put this early so can capture text logs later
        run = None # need both lines since setup_wandb is a @rank_zero_only function
        run = setup_wandb(args)

        wandb_logger = WandbLogger(save_dir=args.wandb_save_dir, name=f'{args.run_name}', entity=f'{args.wandb_entity}', project=f'{args.wandb_project}', offline = args.wandb_offline, experiment=run)
        if args.wandb_tags != None:
            wandb_logger.experiment.tags = args.wandb_tags
    else:
        console_log_file_path = os.path.join("./logs", args.console_log_filename)
        sys.stdout = text_logger.Tee(sys.stdout, console_log_file_path)
        sys.stderr = text_logger.Tee(sys.stderr, console_log_file_path)
        print("$$$$$$$$$ NOTE THAT NOT ALL STDOUT LOGS (i.e. pytorch lighting logs) ARE CAPTURED THROUGH CONSOLE LOGGER, THIS IS ONLY RECOMMENDED FOR DEBUGGING $$$$$$$$$$")

    if args.is_slurm_run:
        if not args.override_slurm_checks:
            assert args.debug_mode == False and args.detect_anomaly == False and args.limit_train_batches == 1 and args.limit_val_batches == 1 and args.limit_test_batches == 1 and args.find_unused_parameters == False and args.debug_unused_parameters == False, "for slurm run cannot have certain params set to values since am assuming are not debugging, please check values here"

        print("Current Slurm job ID:", os.environ.get('SLURM_JOBID'))
        print("Current Slurm node list:", os.environ.get('SLURM_NODELIST'))
        print("SLURM_NTASKS:", os.environ.get("SLURM_NTASKS"))
        print("SLURM_GPUS_PER_NODE:", os.environ.get("SLURM_GPUS_PER_NODE"))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # txt_logger is no longer supported [deprecated]
    # txt_logger = text_logger.setup_custom_logger(log_filename = args.debug_log_filename, print_console = True) # this used to be set to args.print_logs but is default true now and print_logs is not used
    # args.txt_logger = txt_logger

    if args.model_size != "": # set params based off of model size
        args.num_transformer_blocks = model_sizes[args.model_size]['num_transformer_blocks']
        args.multiheaded_attention_heads = model_sizes[args.model_size]['multiheaded_attention_heads']
        args.embedding_dim = model_sizes[args.model_size]['embedding_dim']
        print("model_size", args.model_size, "args.num_transformer_blocks", args.num_transformer_blocks, "args.multiheaded_attention_heads", args.multiheaded_attention_heads, "args.embedding_dim", args.embedding_dim)

    if args.override_embedding_dim > 0:
        args.embedding_dim = args.override_embedding_dim

    if args.override_transformer_blocks > 0:
        args.num_transformer_blocks = args.override_transformer_blocks
    
    # hparam assertion sanity checks
    # if args.modality == "VID":
    #     if args.backbone_type == "dinov2":
    #         assert args.embedding_dim == 0, "embedding dim defined implicitly by encoder dimensionality"
    #         vit_backbone_embed_dim_map = {"small": 384, "base": 768, "large": 1024, "giant": 1536} # gotten from dinov2 repo
    #         args.vit_backbone_dim = vit_backbone_embed_dim_map[args.vit_backbone_size]
    #         args.embedding_dim = args.vit_backbone_dim
    #     elif args.backbone_type == "vae":
    #         assert args.embedding_dim != 0, "must define embedding dim for vae"
    #     else:
    #         raise NotImplementedError(f"Unspported backbone type: {args.backbone_type}")
    if args.modality == "NLP":
        assert args.embedding_dim != 0, "must define embedding dim for NLP models"
        if args.vocab_to_embed_uses_prob_dist:
            assert args.normalize_initial_condition, "if vocab_to_embed_uses_prob_dist is true must use normalize_initial_condition"
    elif args.modality == "VID":
        # VID is old default, treat as NLP for backward compatibility
        args.modality = "NLP"
        print(f"WARNING: modality=VID is deprecated, treating as NLP")
    # elif args.modality == "IMG":
    #     args.backbone_type = "vae" # always uses vae
    #     assert args.embedding_dim != 0, "must define embedding dim for IMG models"
    else:
        raise ValueError(f"please add support for modality {args.modality}")
    
    # assert not(args.random_num_mcmc_steps == True and args.reconstruct_loss_only_final_step == True), "cannot have both random_num_mcmc_steps and reconstruct_loss_only_final_step set"
    # if args.ramp_up_num_mcmc_steps_every_x_epochs != -1:
    #     assert args.random_num_mcmc_steps, "random_num_mcmc_steps needs to be True"
    #NOTE should uncomment if add above hparams back
    
    # 按优先级推导 num_nodes:
    # 1. SLURM_JOB_NUM_NODES  —— Slurm 环境
    # 2. NODE_COUNT            —— rjob/平台注入的环境变量（本集群使用）
    # 3. WORLD_SIZE / LOCAL_WORLD_SIZE —— torchrun 自动注入
    # 4. 默认为 1
    if os.getenv('SLURM_JOB_NUM_NODES') is not None:
        args.num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
        print(f"num_nodes={args.num_nodes} (from SLURM_JOB_NUM_NODES)")
    elif os.getenv('NODE_COUNT') is not None:
        args.num_nodes = int(os.getenv('NODE_COUNT'))
        print(f"num_nodes={args.num_nodes} (from NODE_COUNT)")
    elif os.getenv('WORLD_SIZE') is not None and os.getenv('LOCAL_WORLD_SIZE') is not None:
        world_size = int(os.getenv('WORLD_SIZE'))
        local_world_size = int(os.getenv('LOCAL_WORLD_SIZE'))
        args.num_nodes = max(1, world_size // local_world_size)
        print(f"num_nodes={args.num_nodes} (inferred from WORLD_SIZE={world_size} / LOCAL_WORLD_SIZE={local_world_size})")
    else:
        args.num_nodes = 1
        print("num_nodes=1 (default, no distributed env vars found)")
    print("torch.cuda.device_count()", torch.cuda.device_count())
    if args.gpus == "-1":
        num_gpus = args.num_nodes * torch.cuda.device_count()
    elif '[' in args.gpus: # is a list
        num_gpus = len(args.gpus.split(","))
        args.gpus = json.loads(args.gpus)
    else:
        num_gpus = int(args.gpus)
    print("devices/args.gpus: ", args.gpus)

    # NOTE: num_workers is NOT configurable — nanochat pretrain DataLoader hardcodes
    # num_workers=0. The IterableDataset owns mutable exact-resume state
    # (parquet cursor + doc_buffer) and materializes CUDA tensors inside __iter__(),
    # so worker processes would duplicate iterator state and break resume semantics.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    assert (device == torch.device('cuda') and num_gpus > 0), "using cpu instead of cuda. if you would like to proceed please remove this line and change code below to not use GPUs, otherwise check packages to ensure torch/others have cuda support"
    print(f'GPU Availability: {device}, gpus: {num_gpus}\n')
    args.num_gpus = num_gpus
    effective_batch_size = args.num_gpus * args.batch_size_per_device * args.accumulate_grad_batches
    print(f"effective_batch_size: {effective_batch_size}", "batch_size_per_device", args.batch_size_per_device)
    if args.lr_scaling_rule:
        scaled_lr = args.peak_learning_rate * effective_batch_size / 256 
        args.peak_learning_rate = scaled_lr
        print(f"Learning Rate rescaled to: {scaled_lr} based off lr_scaling_rule")
    if args.max_scheduling_steps == -1:
        args.max_scheduling_steps = args.max_steps

    model_trainer = ModelTrainer(args)

    # SFT: 加载预训练权重但不恢复训练状态
    if args.finetuning_model_ckpt is not None and args.finetuning_model_ckpt != "":
        print(f"[SFT] 加载预训练权重: {args.finetuning_model_ckpt}")
        ckpt = torch.load(args.finetuning_model_ckpt, map_location='cpu', weights_only=False)
        model_trainer.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"[SFT] 权重加载完成，训练从 step 0 开始")

    # if args.debug_dataloader:
    #     debug_dataloader(args, model_trainer)
    #     return
    is_rank_zero_process = int(os.environ.get("RANK", "0")) == 0
    if args.log_model_archi and is_rank_zero_process:
        print(str(model_trainer.model))
        print(str(args))

    if not args.no_wandb and args.wandb_watch:
        init_wandb_watch(wandb_logger, model_trainer, args.wandb_watch_log_freq, args.wandb_watch_level)

    if args.create_model_viz:
        # BACKLOG use tensorboard for this if active eventually. disabled for now since makes some things challenging
        return
    print(f'pytorch version: {torch.__version__}\n')

    if args.set_matmul_precision is not None: #default is highest
        torch.set_float32_matmul_precision(args.set_matmul_precision)
    
    opt_name = args.optimizer if hasattr(args, 'optimizer') else 'adamw'
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else f"./logs/checkpoints/{args.run_name}"
    checkpoint_filename = f"s={{step}}-{args.model_size}-ctx{args.context_length}-lr{args.peak_learning_rate}-bs{args.batch_size_per_device}x{args.accumulate_grad_batches}-{opt_name}-{args.checkpoint_monitor_string}={{{args.checkpoint_monitor_string}:.4f}}"
    save_last = (args.save_periodic_steps <= 0 and args.save_top_k_ckpts != 0)  # save_top_k=0 means no checkpoint files
    checkpoint_cleanup_on_low_space = not args.disable_checkpoint_cleanup_on_low_space
    if args.checkpoint_preflight_required:
        preflight_checkpoint_disk(args, ckpt_dir, save_last, model_trainer, is_rank_zero_process)
    checkpoint_callback = DiskAwareCheckpoint(monitor=args.checkpoint_monitor_string, mode = args.checkpoint_monitor_mode, save_top_k=args.save_top_k_ckpts, save_last = save_last, dirpath=ckpt_dir, filename=checkpoint_filename, verbose=True, min_free_gb=args.checkpoint_min_free_gb, cleanup_on_low_space=checkpoint_cleanup_on_low_space)

    # 定期保存 checkpoint（不依赖 val_loss），防止 SFT 后期模型丢失
    periodic_checkpoint = None
    if args.save_periodic_steps > 0:
        periodic_checkpoint = DiskAwareCheckpoint(
            save_top_k=1,
            save_last=False,
            every_n_train_steps=args.save_periodic_steps,
            dirpath=ckpt_dir,
            filename=f"periodic-s={{step}}-{args.model_size}-ctx{args.context_length}",
            verbose=True,
            min_free_gb=args.checkpoint_min_free_gb,
            cleanup_on_low_space=checkpoint_cleanup_on_low_space
        )

    for name, param in model_trainer.model.named_parameters():
        if not param.requires_grad:
            print(f"Non-trainable parameters: {name} with shape {param.shape}")
    
    if not args.only_test: #training and testing (if testing selected) as per usual
        print("$$$$$$$$$$  STARTED TRAINING  $$$$$$$$$$")
        trainer = set_trainer(args, wandb_logger, checkpoint_callback, periodic_checkpoint=periodic_checkpoint)
        resume_training_ckpt = None if args.resume_training_ckpt == "" else args.resume_training_ckpt
        trainer.fit(model_trainer, ckpt_path=resume_training_ckpt, weights_only=False)
        
        if args.run_testing_after_training:
            args.only_test_model_ckpt = checkpoint_callback.best_model_path
            best_model = ModelTrainer.load_from_checkpoint(
                checkpoint_callback.best_model_path, 
                hparams=args
            )
            clear_cache()
            print(f"best model path that will be used during testing {checkpoint_callback.best_model_path}")
            print("$$$$$$$$$$  STARTED TESTING AFTER TRAINING  $$$$$$$$$$")
            raise NotImplementedError("need to test this with newer PL")
            # test_trainer = L.Trainer(logger=wandb_logger,devices=1,num_nodes=1) NOTE tested this code does not work gets stuck, see thread, TODO test with newer PL
            #warning reference - https://github.com/Lightning-AI/lightning/issues/12862
            best_model.eval()
            trainer.test(best_model)
        clear_cache()
    else: #only testing
        print("$$$$$$$$$$  ONLY TESTING MODEL ([NO]) TRAINING  $$$$$$$$$$")
        assert args.only_test_model_ckpt != None, "Must supply pretrained model when only testing"
        run_name, log_file_name = args.only_test_model_ckpt.split("/")[-2:]
        args.save_generation_logs_dir = os.path.join(args.infer_output_dir, args.modality, args.dataset_name, run_name, log_file_name.replace(".ckpt", ""))
        if os.path.exists(args.save_generation_logs_dir): # remove any existing logs
            shutil.rmtree(args.save_generation_logs_dir)

        
        checkpoint = torch.load(args.only_test_model_ckpt, weights_only=False)
        pretrained_hparams = checkpoint['hyper_parameters']

        default_args = vars(args).copy() # NOTE this is so can test older models trained with older code as well as use newer hparams (for inference) on pretrained models, may be finicky feel free to tweak
        for key, value in default_args.items():
            if key not in pretrained_hparams:
                pretrained_hparams[key] = value
                print(f"MISSING PARAMETER IN PRETRAINED CHECKPOINT: Using args set value for missing parameter in pretrained checkpoint '{key}': {value}")
            if key.startswith("infer_"):
                pretrained_hparams[key] = value
                print(f"OVERRIDING PARAMETER IN PRETRAINED CHECKPOINT FOR INFERENCE: Using args set value for inference parameter '{key}': {value}")

        # if args.modality == "VID":
        #     pretrained_hparams["modality"] = "VID" # this is just to test old models with a refactor of CV -> VID

        model = ModelTrainer(pretrained_hparams)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model_trainer = ModelTrainer(args, trained_model=model.model) # need to use args as we have to use the model most recently passed in for inference

        trainer = set_trainer(args, wandb_logger, checkpoint_callback, stage = "test")
        model_trainer.model.eval()
        trainer.test(model_trainer)
        if args.modality == "NLP": # can have modality specific logic here for inference
            if args.execution_mode == "inference":
                # Only compute EM/F1 for generation tasks (GSM8K, etc), not PPL tasks (nanochat)
                if args.dataset_name in ["gsm8k", "arc", "humaneval", "mmlu", "smoltalk", "spellingbee"]:
                    em_score, f1_score = nlp_eval_acc(os.path.join(args.save_generation_logs_dir, "results.jsonl"))
                    trainer.logger.experiment.log({"em_score": em_score, "f1_score": f1_score})
                else:
                    print(f"Skipping EM/F1 evaluation for PPL-only dataset: {args.dataset_name}")
        # elif args.modality == "VID":
        #     if args.infer_generate_video:
        #         print("calling style gan FVD code on generated video dataset, NOTE THIS CODE MAY NOT WORK AS EXPECTED or get stuck")
        #         fvd, fid = call_style_gan_fvd(args)
        #         trainer.logger.experiment.log({"test_fvd": fvd, "test_fid": fid})
        # elif args.modality == "IMG":
        #     pass # no post test code for denoising, for t2i could call FID code here if desired
        else:
            raise NotImplementedError(f"no post test evaluation setup for this modality: {args.modality} yet")

def set_trainer(args, wandb_logger, checkpoint_callback, stage = "train", periodic_checkpoint=None):
    torch.autograd.set_detect_anomaly(args.detect_anomaly) #NOTE seems pl detect anomaly is not working so manually set it here

    if args.find_unused_parameters or args.distributed_strategy == "ddp":
            args.distributed_strategy = DDPStrategy(
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
    args.overfit_batches = int(args.overfit_batches) if int(args.overfit_batches) == args.overfit_batches else args.overfit_batches
    profiler = None if args.profiler == "" else args.profiler
    gradient_clip_val = args.gradient_clip_val if args.gradient_clip_val > 0 else None
    limit_val_batches = 0 if args.overfit_batches > 0 else args.limit_val_batches
    if (
        getattr(args, "dataset_name", "") == "nanochat_sft"
        and isinstance(args.val_check_interval, float)
        and args.val_check_interval > 1
        and args.val_check_interval.is_integer()
    ):
        args.val_check_interval = int(args.val_check_interval)
    # val_check_interval = args.val_check_interval if args.val_check_interval == 1.0 else args.val_check_interval * args.accumulate_grad_batches  #NOTE the reason we mult by args.accumulate_grad_batches is because of this bug https://github.com/Lightning-AI/pytorch-lightning/issues/12205
    limit_test_batches = args.limit_test_batches if args.limit_test_batches == 1 else args.limit_test_batches * args.accumulate_grad_batches

    # v3: optional adaptive sudoku ratio + difficulty-bucket schedule callback. Only
    # constructed when training on sudoku_mixed AND a non-fixed schedule is requested,
    # so non-sudoku runs retain their previous Trainer construction byte-for-byte.
    extra_callbacks = []
    if (
        getattr(args, 'dataset_name', '') == 'sudoku_mixed'
        and (
            getattr(args, 'sudoku_ratio_schedule', 'fixed') != 'fixed'
            or getattr(args, 'sudoku_difficulty_schedule', 'fixed') != 'fixed'
        )
    ):
        from openebm.elm.trainer import AdaptiveRatioCallback
        extra_callbacks.append(
            AdaptiveRatioCallback(
                ratio_schedule=getattr(args, 'sudoku_ratio_schedule', 'fixed'),
                difficulty_schedule=getattr(args, 'sudoku_difficulty_schedule', 'fixed'),
            )
        )

    trainer = Trainer(
        accelerator="auto",
        devices = args.gpus,
        num_nodes=args.num_nodes,
        precision=args.float_precision,
        max_steps=args.max_steps,
        logger=wandb_logger,
        enable_model_summary=args.log_model_archi,
        callbacks = [checkpoint_callback] + ([periodic_checkpoint] if periodic_checkpoint else []) + extra_callbacks + [ModelSummary(max_depth=-1)],
        strategy = args.distributed_strategy, 
        enable_checkpointing=True,
        fast_dev_run = args.fast_dev_run,
        num_sanity_val_steps = args.val_sanity,
        limit_train_batches = args.limit_train_batches,
        limit_val_batches = limit_val_batches,
        limit_test_batches = limit_test_batches,
        detect_anomaly=args.detect_anomaly,
        gradient_clip_val=gradient_clip_val,
        overfit_batches=args.overfit_batches,
        profiler=profiler,
        val_check_interval=args.val_check_interval,
        # check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.deterministic,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        inference_mode=False # set inference mode to false to get grad for models like ebt during testing
    )

    # trainer = IterableTrainer(
    #     accelerator="auto",
    #     devices = args.gpus,
    #     num_nodes=args.num_nodes,
    #     precision=args.float_precision,
    #     max_steps=args.max_steps,
    #     logger=wandb_logger,
    #     enable_model_summary=args.log_model_archi,
    #     callbacks = [checkpoint_callback, ModelSummary(max_depth=-1)],
    #     strategy = args.distributed_strategy, 
    #     enable_checkpointing=True,
    #     fast_dev_run = args.fast_dev_run,
    #     num_sanity_val_steps = args.val_sanity,
    #     limit_train_batches = args.limit_train_batches,
    #     limit_val_batches = limit_val_batches,
    #     limit_test_batches = limit_test_batches,
    #     detect_anomaly=args.detect_anomaly,
    #     gradient_clip_val=gradient_clip_val,
    #     overfit_batches=args.overfit_batches,
    #     profiler=profiler,
    #     val_every_n_step=args.val_every_n_step,
    #     val_after_n_step=args.val_after_n_step,
    #     deterministic=args.deterministic,
    #     log_every_n_steps=args.log_every_n_steps,
    #     accumulate_grad_batches=args.accumulate_grad_batches,
    #     inference_mode=False # set inference mode to false to get grad for models like ebt during testing
    # )

    return trainer
    

def clear_cache():
    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    parser = ArgumentParser()

    #SUPER IMPORTANT HPARAMS--always set these properly #############################################

    parser.add_argument("--run_name", help="run name, should include model name in it for important runs (see slurm scripts)", default="test")

    parser.add_argument("--modality", help="is the model being trained for NLP, VID, IMG. default is VID, and most of the code has been designed around VID.", type=str, default="VID")

    parser.add_argument("--model_name", help="model_type/name", default='ebt')

    parser.add_argument("--model_size", help="model size, is used to set num_transformer_blocks, multiheaded_attention_heads, and embedding_dim", choices=model_sizes.keys(), default='xxs')

    #NOTE not all below hparams are implemented for all models, so please check before using one     

    # VID SPECIFIC PARAMS ############################################################################################################

    parser.add_argument("--vit_backbone_size", help="small, base, large or giant", type=str, default="base")

    parser.add_argument("--backbone_type", help="backbone type for VID encoder", choices=["dinov2", "vae"], type=str, default="dinov2")

    parser.add_argument("--time_between_frames", help="time (seconds) between frames of video", type=float, default=1.0)
    
    parser.add_argument("--sampling_rate", help="how many frames to draw a single video frame from. e.g. sampling rate=4 will have three frames in between each sample. overrides time_between_frames", type=float, default=0.0)
    
    parser.add_argument("--use_raw_framerate", help="whether to use dataset default framerate with a sampling rate of 1. this overrides the time_between_frames and sampling_rate parameters.", action="store_true", default=False)

    parser.add_argument("--vae_normalization", help="whether to use the commonly used vae normalization of [-1, 1].", action="store_true", default=False)

    parser.add_argument("--sdxl_vae_standardization", help="whether to standardize vae encodings according to sdxl by multiplying by 0.18215.", action="store_true", default=False)

    parser.add_argument("--weight_tie_vae_proj", help="for baseline transformer, weight ties the projection to vae latent space", action="store_true", default = False)
     
    # NLP SPECIFIC PARAMS ############################################################################################################

    # IMPORTANT: For NanoChat dataset, this parameter is IGNORED!
    # The actual tokenizer is loaded via get_tokenizer() in trainer.py (line 165)
    # which uses NanoChat custom BPE tokenizer from $NANOCHAT_BASE_DIR/tokenizer/
    # This parameter is only used for non-NanoChat datasets.
    parser.add_argument("--tokenizer", help="tokenizer for nlp tasks (IGNORED for nanochat dataset)", type=str, default="EleutherAI/gpt-neox-20b")
    
    parser.add_argument("--pretokenize_dataset", help="whether to pretokenize the dataset and save that or just tokenize as loading the dataset. tokenizing the dataset takes a long time and may need to be done using debug dataloader. due to a bug with HF does not always work reliably, may get stuck. not currently implemented for fine-tuning datasets", action="store_true", default=False)

    parser.add_argument("--normalize_initial_condition", help="makes initial condition a normalized probability distribution if modality is NLP, helps a ton with stability", action="store_true", default = False)

    parser.add_argument("--normalize_initial_condition_only_first_step", help="makes initial condition a normalized probability distribution if modality is NLP ONLY on first MCMC step, by default does for all steps. this didnt work very well", action="store_true", default = False)

    parser.add_argument("--sharpen_predicted_distribution", help="scales the temperature of the distribution, higher temp means less sharp lower temp sharpens distribution. divides by this number. didnt work very well", type=float, default=0.0)

    parser.add_argument("--vocab_to_embed_uses_prob_dist", help="makes initial condition a normalized probability distribution if modality is NLP, this worked decently well and may be more data efficient", action="store_true", default = False)

    parser.add_argument("--soften_target_prob_dist", help="softens the target prob dist differently across different mcmc steps, making it harder during later MCMC steps. didnt work very well", type=float, default=0.0)

    # IMG SPECIFIC PARAMS ########################################################################################################################

    parser.add_argument("--image_task", help="task for image modality, default is text to image (t2i)", choices=["t2i", "denoising"], type=str, default="t2i")

    parser.add_argument("--denoise_images_noise_level", help="when denoising is the image task what std of gaussian noise added should be", type=float, default=1)

    parser.add_argument("--randomize_denoise_images_noise_level", help="(for non diffusion models) denoise the original image to a random extent, rather than just the max", action="store_true", default = False)

    parser.add_argument("--clip_text_encoder_size", help="size for clip text encoder", choices=["small", "base", "large", "xl"], type=str, default="base")

    parser.add_argument("--patch_size", help="patch size from DiT paper, larger means less compute", choices=[2, 4, 8, 16], type=int, default=8)

    parser.add_argument("--log_image_every_n_steps", help="logs a generated image every n training steps", type=int, default=500)

    #EBT MCMC ##################################################################
    #NOTE in the code we broadly refer to MCMC/optimization interchangeably, although in the paper we refrain from the term mcmc

    parser.add_argument("--langevin_dynamics_noise", help="is the standard deviation of noise to use that is mean centered", type=float, default=0.0)
    
    parser.add_argument("--langevin_dynamics_noise_learnable", help="is the std of noise for langevin dynamics learnable", action="store_true", default = False)

    parser.add_argument("--no_langevin_during_eval", help="dont use langevin dynamics during validation", action="store_true", default = False)
    
    parser.add_argument("--mcmc_step_size", help="is size of optimization step, or alpha in the paper, kinda like LR, can be learned param", type=float, default=60.0)
    
    parser.add_argument("--mcmc_step_size_learnable", help="makes mcmc_step_size a learnable param", action="store_true", default = False)

    parser.add_argument("--mcmc_step_size_per_step",
        help="learn a separate mcmc_step_size (alpha) for each MCMC step index",
        action="store_true", default=False)
    parser.add_argument("--mcmc_step_size_per_step_values", type=str, default="",
        help="Comma-separated per-step alpha initial values when --mcmc_step_size_per_step is enabled. "
             "Length must match --mcmc_num_steps, e.g. '50,300'. Empty means repeat --mcmc_step_size.")
    parser.add_argument("--mcmc_step_size_per_step_learnable_mask", type=str, default="",
        help="Comma-separated 0/1 per-step alpha learnable mask when --mcmc_step_size_per_step is enabled. "
             "Length must match --mcmc_num_steps, e.g. '0,1' fixes alpha0 and learns alpha1. "
             "Empty means all per-step alphas follow --mcmc_step_size_learnable.")

    parser.add_argument("--mcmc_step_size_lr_multiplier", help="learning rate multiplier for mcmc step size, so to get lr of mcmc step size take lr multiply by this value", type=float, default=5000.0)

    parser.add_argument("--randomize_mcmc_step_size_scale", help="randomize the value of mcmc_step_size by a factor specified, i.e. if is 2 will mult by 2 and div by 2 and thats the range to sample from uniformly", type=float, default=1)
    
    parser.add_argument("--mcmc_num_steps", help="number of MCMC steps, try 2-5, check data samples as well to see how many we need. NOTE if are using time embed or adaln is the number of energy landscapes", type=int, default=2)
    parser.add_argument("--zero_mcmc_ce_only", action="store_true", default=False,
        help="Strict K=0 ablation: require --mcmc_num_steps 0 and train only the TF-head CE path from the initial corrupted/free-embedding state, without energy-head forward, autograd.grad, or MCMC update.")

    parser.add_argument("--randomize_mcmc_num_steps", help="makes mcmc_num_steps random, each step at each landscape is repeated uniform(1, 1+randomize_mcmc_num_steps) times (unless randomize_mcmc_num_steps_min is set, then thats the min value). if ebt_type is default each landscape is the same, so it effectively just randomized mcmc_num_steps", type=int, default=0)

    parser.add_argument("--randomize_mcmc_num_steps_final_landscape", help="makes it so the randomize_mcmc_num_steps param only applies to the final energy landscape when using non default EBT", action="store_true", default = False)

    parser.add_argument("--randomize_mcmc_num_steps_min", help="makes it so there is a minimum number of randomized mcmc steps", type=int, default=0)
    
    parser.add_argument("--denoising_initial_condition", help="['random_noise', 'most_recent_embedding', or 'zeros'], what condition to start off for denoising. random_noise is most likely the best and what is used for every experiment in the paper, most_recent_embedding may be helpful for VID but probably not for NLP, or can condition on zeros (vector)", type=str, default="random_noise")

    parser.add_argument("--gaussian_random_noise_scaling", help="scales the std of the gaussian sampled from for random noise denoising_initial_condition. is mainly for NLP ebt model (not impl in other modalities for now)", type=float, default=1)

    parser.add_argument("--clamp_futures_grad", help="clamps grad of predictions (used to be called futures when was doing AR). not used anymore but can help with stability", action="store_true", default=False)

    parser.add_argument("--clamp_futures_grad_max_change", help="max total change during mcmc to clamp to (possibly divides by num mcmc steps), used by above. not used anymore", type=float, default=9.0)

    parser.add_argument("--absolute_clamp", help="clamps the absolute value of predicted_tokens to be within range [-val, val]. not used anymore", type=float, default=0.0)

    parser.add_argument("--clamp_max_after_warm_up", help="clamps the absolute value of predicted_tokens to be within range [-val, val], after warming up. not used anymore", type=float, default=0.0)

    parser.add_argument("--ebt_type", help="type of energy based transformer to use, inspired by DiT paper.", choices=["default", "time_embed", "adaln", "adaln_zero", "nanochat_d26"], type=str, default="default")

    # TF (teacher-forced, Gemma-drafter-style) head — sits AFTER MCMC and consumes
    # either trunk pred_hidden or the free-embedding MCMC state for CE logits. Some
    # variants also use embed(input_ids[t]) as a first-layer anchor.
    # See TF_HEAD_ARCHITECTURE.md §9 and openebm/elm/tf_head.py.
    parser.add_argument("--use_tf_head", action="store_true", default=False,
        help="Replace MCMC-derived CE with TF head on top of trunk pred_hidden. Default off.")
    parser.add_argument("--tf_head_type", choices=["linear", "concat_direct_unembed", "transformer", "direct_unembed", "pre_update_hidden_unembed", "post_update_state_unembed"], type=str, default="transformer",
        help="TF head variant. 'transformer' = L-block causal head (Gemma drafter). "
             "'linear' = concat+project. "
             "'concat_direct_unembed' = project concat(post-update trunk pred_hidden, prev_embed) directly. "
             "'direct_unembed' = project post-update trunk pred_hidden directly. "
             "'pre_update_hidden_unembed' = project the energy-forward pre-update trunk pred_hidden directly. "
             "'post_update_state_unembed' = project the post-update free-embedding MCMC state directly.")
    parser.add_argument("--tf_head_layers", type=int, default=1,
        help="Number of causal AR blocks in the transformer head (L=1 is empirical sweet spot).")
    parser.add_argument("--tf_head_n_heads", type=int, default=0,
        help="Attention heads inside TF head block; 0 inherits trunk's multiheaded_attention_heads.")
    parser.add_argument("--tf_head_ffn_mult", type=float, default=4.0,
        help="FFN expansion factor inside TF head transformer block.")
    # Free embedding MCMC: iterate in D-dim embedding space directly (instead of V-dim logit).
    # Skips the softmax + matmul(embeddings.weight) conversion at trunk input. Requires
    # --use_tf_head (TF head provides the discrete CE supervision; without it the D-dim iterate
    # has no decode path). See blockwise_free_embedding_optimizations_2026-06-01.md.
    parser.add_argument("--free_embedding_mcmc", action="store_true", default=False,
        help="Run MCMC in D-dim free embedding space instead of V-dim logit space. Requires --use_tf_head.")
    parser.add_argument("--free_embed_noise_scale", type=float, default=1.0,
        help="Multiplier on initial corruption noise when free_embedding_mcmc=True (doc §4 noise rescaling). "
             "Higher = harder denoising = forces model to use context.")
    parser.add_argument("--post_update_state_use_rmsnorm", action="store_true", default=False,
        help="For tf_head_type=post_update_state_unembed only: apply RMSNorm to z_{i+1} before Linear(D,V).")
    parser.add_argument("--post_update_state_concat_zi", action="store_true", default=False,
        help="For tf_head_type=post_update_state_unembed only: decode concat(z_i, z_{i+1}) with Linear(2D,V).")
    parser.add_argument("--post_update_state_concat_prev_embed", action="store_true", default=False,
        help="For tf_head_type=post_update_state_unembed only: decode concat(z_{i+1}, embed(input_ids)) with Linear(2D,V).")
    parser.add_argument("--post_update_state_detach_prev_embed", action="store_true", default=False,
        help="For post_update_state_unembed + concat_prev_embed only: detach embed(input_ids) before TF-head CE.")
    parser.add_argument("--post_update_state_tf_head_adamw", action="store_true", default=False,
        help="For tf_head_type=post_update_state_unembed with --optimizer muon_adamw: put TF-head matrix params in AdamW instead of Muon.")

    parser.add_argument("--use_ve", help="启用 Value Embedding (VE)，为交替层添加可学习的值嵌入", action="store_true", default=False)

    parser.add_argument("--use_mcmc_time_embed", action="store_true", default=False,
        help="Enable MCMC step time embedding (only for ebt_type=time_embed). When False, all steps share the same transition kernel, enabling arbitrary step count at inference.")

    parser.add_argument("--ebt_norm", help="type of norm to use for energy based transformer, NOTE is only supported for ebt_time_embed. not used anymore didnt work better than default rms from llama2", choices=["rms", "none", "layer", "ebm_backwards_norm", "dyt"], type=str, default="rms")

    parser.add_argument("--dyt_alpha_init", help="initial value for alpha in dyt layer norm, from paper https://jiachenzhu.github.io/DyT/. didnt work well increased instability", type=float, default=0.5)

    parser.add_argument("--ebt_act_func", help="activation function to use for energy based transformer, NOTE is only supported for ebt_time_embed. silu (default from llama2) worked best", type=str, default="silu")

    parser.add_argument("--truncate_mcmc", help="truncate mcmc and only use final step of loss to calculate, for S2 models", action="store_true", default=False)
    parser.add_argument("--truncate_mcmc_per_step_ce", action="store_true", default=False,
        help="Use S2-style truncated MCMC gradients while still averaging CE over all supervised MCMC steps. "
             "Requires --truncate_mcmc; useful to isolate final-only CE from create_graph truncation.")

    parser.add_argument("--mcmc_replay_buffer", help="enables a replay buffer for MCMC, particularly S2 models", action="store_true", default=False)

    parser.add_argument("--mcmc_replay_buffer_sample_bs_percent", help="number of samples to retrieve in replay buffer, as a percentage of batch size per device", type=float, default=0.5)

    parser.add_argument("--mcmc_replay_buffer_size", help="number of total samples to store in replay buffer", type=int, default=192) # for 40 GB A100 192 is the max

    parser.add_argument("--no_mcmc_detach", help="dont detach between mcmc steps, probably need to use for S2 models but can increase instability due to longer gradient computation graphs", action="store_true", default=False)

    parser.add_argument("--contrastive_loss", help="uses a contrastive loss to shape the landscape of EBM, idea from IRED paper https://arxiv.org/abs/2406.11179", action="store_true", default=False)

    parser.add_argument("--contrastive_loss_coeff", help="coefficient for contrastive loss, didnt work well not used", type=float, default=0.0005)

    parser.add_argument("--discrete_contrastive_loss_true_logit_val", help="value for contrastive loss discrete logits, didnt work well not used", type=float, default=0.0)

    parser.add_argument("--learnable_process_memory", help="allows for a learnable process memory, which will be mapped to the same embed as the vocab dist using an MLP. didnt explore enough but worth trying out :) feel free to reach out to discuss", action="store_true", default=False)

    parser.add_argument("--process_memory_type", help="type of learnable process memory", choices=['add', 'gate', 'residual_gate'], type=str, default=None)
    
    parser.add_argument("--process_memory_linear_layer", help="allows for the process memory to have an extra linear layer to process it", action="store_true", default=False)

    # removed hparams that could be useful to try again (if enable add impl (ask alexi) and uncomment above lines with NOTE that use these params)

    # parser.add_argument("--reconstruct_loss_only_final_step", help="this will make it so the reconstruction loss will only be based on the final step. this is useful to allow for more MCMC to occur without lowering bs.", action="store_true", default = False)

    # parser.add_argument("--random_num_mcmc_steps", help="makes it so out of the given num or mcmc steps, has random chance to do 1-N (where N is num provided). Only calcs rec loss on last 2 steps.", action="store_true", default = False)
    
    # parser.add_argument("--ramp_up_num_mcmc_steps_every_x_epochs", help="every X epochs ramps up num mcmc steps, should use with random_num_mcmc_steps", type=int, default=-1) # shuld prob replace epochs for steps to be more modality agnostic

    #EBT [mostly deprecated] THINGS from a past life ################################################################
        
    parser.add_argument("--reconstruction_coeff", help="coefficient for reconstruction loss, was for when doing multi objective but not just is always 1", type=float, default=1.0)
    
    parser.add_argument("--out_of_bounds_loss_coeff", help="coefficient for oob loss for mcmc, was for when was trying to regularize energy landscape, didnt work well not used", type=float, default=0.0)

    # all below are for energy loss (which is no longer used, just rec loss is used)

    parser.add_argument("--energy_loss_coeff", help="coefficient for energy prediction loss, was for when was trying to regularize energy landscape, didnt work well not used", type=float, default=0.0)
    
    parser.add_argument("--energy_loss_hinge", help="hinge loss margin, so only predicts energy being close up to this amount, helps with inherent randomness in cosine sim", type=float, default=0.0)
    
    parser.add_argument("--energy_loss_fn", help="l1_loss, MSE, smooth_l1_loss", type=str, default="MSE")
        
    parser.add_argument("--embeddings_distance_fn", help="for comparing embeddings of predictions (energy loss coeff) - euclidean, normalized_euclidean, cosine, manhattan", type=str, default="cosine")
    
    parser.add_argument("--scale_cosine_sim_decay", help="scale cosine sim so has more variation < 0.5 and less above 0.5, try 0 (for none) 2, 4, 6, 8 for energy_loss_coeff", type = int, default=7)

    # DIFFUSION ################################################################
    parser.add_argument("--diffusion_steps", help="number of steps in diffusion", type = int, default=1000)

    parser.add_argument("--use_deterministic_reverse", help="makes it so the reverse process is deterministic for generation and uses ddim instead of ddpm with learned sigma", action="store_true", default = False)

    parser.add_argument("--infer_increase_steps", help="inference number of steps to increase to", type = int, default=0)

    parser.add_argument("--infer_recurse_diffusion_n_times", help="inference number of times to recurse diffusion on its own outputs, helpful for denoising OOD data. did work better than increase steps", type = int, default=0)

    #MODEL AND ARCHITECTURE ##################################################################

    # transformer specific ############################################

    parser.add_argument("--context_length", help="context length for AR models, i.e. for language model (commonly 256) or video model (commonly 16)", type=int, default=0)
          
    parser.add_argument("--num_transformer_blocks", help="number of transformer blocks, uses default from model size specified", type=int, default=12)
    
    parser.add_argument("--multiheaded_attention_heads", help="number of attention heads for transformer, uses default from model size specified", type=int, default=2)

    ######################################################################

    parser.add_argument("--embedding_dim", help="embedding dimension for transformers, if are using a model size this is automatically set", type=int, default=384)

    parser.add_argument("--ffn_dim_multiplier", help="how much wider than the embedding dim the transformer FFN dim should be", type=float, default=None)

    parser.add_argument("--override_embedding_dim", help="override the embedding dimension for a transformer. do not use unless you know what you are doing since it will likely cause issues and mess up scaling trends", type=int, default=0)

    parser.add_argument("--override_transformer_blocks", help="override the number of blocks for a transformer. do not use unless you know what you are doing since it will likely cause issues and mess up scaling trends", type=int, default=0)

    parser.add_argument("--num_modality_processing_mlp_layers", help="number of linear layers in the MLP for processing each modality, need at least 1", type=int, default=1) # NOTE not currently used or implemented

    parser.add_argument("--weight_initialization_method", help="xavier or he", type=str, default="xavier")

    parser.add_argument("--weight_initialization_gain", help="gain of the weight init, see https://pytorch.org/docs/stable/nn.init.html, default is equiavalent to linear gain which is 1. tried tweaking this for EBT didnt help", type=float, default=1.0)

    # seperate MLP specific #####################################################################
    # NOTE, these are not for transformer MLP are for seperate MLPs
    parser.add_argument("--mlp_dropout", help="dropout in basic MLP", type=float, default=0.0)
                
    parser.add_argument("--mlp_layer_norm", help="layer normalization", action="store_true", default=False)

    parser.add_argument("--mlp_dim_multiplier", help="how much wider than the embedding dims the final layer projector should be", type=float, default=2.0)

    parser.add_argument("--mlp_hidden_layers", help="number of linear layers in MLP after encoder, if is 1 is just a linear layer", type=int, default=0)

    ################
    
    parser.add_argument("--encoder_lr_divider", help="how much to divide lr for encoder", type=float, default=100.0)  

    #HARDWARE###################################################################

    parser.add_argument("--gpus", help="number of gpus or gpus list, -1 uses all GPUs. use -1 for multinode, if want to specify which GPUs to use specify as comma seperated str with brackets e.g. [0, 1]", default="-1")
    
    parser.add_argument("--distributed_strategy", help="distributed strategy - ddp_spawn, ddp, fsdp_native, or None", default='ddp')


    #TRAINING#########################################################

    parser.add_argument("--peak_learning_rate", help="peak learning rate after warm up", type=float, default=0.02)

    parser.add_argument("--batch_size_per_device", help="batch size (PER DEVICE!!!, not effective). to get effective_batch_size is num_gpus * batch_size_per_device * accumulate_grad_batches, see above)", type=int, default=2)

    parser.add_argument("--accumulate_grad_batches", help="number of batches to accumulate before stepping optimizer, 1 is regular", type=int, default=1)

    parser.add_argument("--gradient_clip_val", help="maximum value for gradient clipping", type=float, default=1.0)
    
    parser.add_argument("--is_random_seed", help="is_random_seed", action="store_true", default=False)
    
    parser.add_argument("--deterministic", help="ensures results are determnistic, may run a bit slower, sets flag on pl trainer and sets workers for dataloader", action="store_true", default=False)

    parser.add_argument("--execution_mode", type=str, choices=["pretrain", "finetune", "inference"], default="pretrain")
    
    parser.add_argument("--finetuning_model_ckpt", help="model ckpt when finetuning", type=str, default=None)

    #METRICS#########################################################
    #NOTE reported metrics in wandb will be averages over the time span they are being computed, so the final one in an epoch will be the most accurate

    parser.add_argument("--metrics_list", help = "list of metrics to use", nargs='+', default = [])
    
    parser.add_argument("--metrics_average_type", help="type of average to use for accuracy - macro, micro, weighted - see torchmetrics docs for more info", type=str, default="micro")

    parser.add_argument("--num_classes", help="the number of classes, must set if using metrics, see torchmetrics docs for more info", type=int, default=-1)

    parser.add_argument("--metrics_task", help="the type of task being done ['binary', 'multiclass' or 'multilabel'], see torchmetrics docs for more info, must set if using metrics_list", type=str, default="")
    
    #OPTIMIZER AND LR SCHEDULER#########################################################
    
    parser.add_argument("--weight_decay", help="weight decay to use", type=float, default=0.01)
    
    parser.add_argument("--beta1", help="exponential decay rate for first moment estimate", type=float, default=0.9)
    
    parser.add_argument("--beta2", help="exponential decay rate for second movement estimate", type=float, default=0.999)

    parser.add_argument("--lr_scaling_rule", help="the LR will be scaled according to the rule LR = base_lr * effective_batch_size / 256. is useful for prototyping and is popular in vision SSL. effective_batch_size is based off bs * num_gpus * accumulate_grad_batches", action="store_true", default=False)

    parser.add_argument("--min_lr_scale", help="the most the lr will be scaled down during cosine decay", type=int, default=10)

    parser.add_argument("--max_steps", help="max number of steps for training", type=int, default=1000000)

    parser.add_argument("--max_scheduling_steps", help="max number of steps used for lr/other hparam scheduling. in general should be the same as max_steps but can be different if dont want to do a full run but want the lr and other things to be scheduled the same. similar to V jepa paper. if is not set will default to max_steps value", type=int, default=-1)
        
    parser.add_argument("--warm_up_steps", help="number of steps to increase the LR linearly over before hitting specified peak LR, starts from 0 unless warm_up_base_lr_divider is set and does linear warm up", type=int, default=10000)
    
    parser.add_argument("--warm_up_base_lr_divider", help="lr divider for when doing linear learning rate warm up. if is set to -1 then does warm up from 0", type=float, default=-1)
    
    parser.add_argument("--optimizer", help="used to turn on different optimizers. current options include adamw (default), lars, stableadamw, muon_adamw", type=str, default="adamw")

    parser.add_argument("--lars_trust_coeff", help="exponential decay rate for second movement estimate", type=float, default=0.001)

    parser.add_argument("--lars_exclude_bias_bn_wd", help="excludes bias and batch norm from Lars adaptation and weight decay", action="store_true", default=False)

    # Muon + AdamW 混合优化器参数 (--optimizer muon_adamw 时生效)
    parser.add_argument("--muon_lr", help="[Muon] 矩阵参数学习率 (Muon optimizer)", type=float, default=0.02)
    parser.add_argument("--muon_momentum", help="[Muon] Nesterov 动量系数", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", help="[Muon] Polar Express 迭代次数", type=int, default=5)
    parser.add_argument("--muon_beta2", help="[Muon] 二阶矩 beta2 (variance reduction)", type=float, default=0.95)
    # Muon 模式下 AdamW 参数的独立绝对 LR (不再从 peak_lr 派生)
    # 参考 NanoChat gpt.py:setup_optimizer 和 base_train.py 的默认值
    # 设为 -1 表示 fallback 到 peak_lr × mult 的旧行为
    parser.add_argument("--adamw_embedding_lr", help="[Muon] embedding 绝对 LR (NanoChat 默认 0.3, 会自动 dmodel scaling)", type=float, default=-1)
    parser.add_argument("--adamw_vocab_to_embed_lr", help="[Muon] vocab_to_embed 绝对 LR (EBT 特有, 建议保守 0.01)", type=float, default=-1)
    parser.add_argument("--adamw_scalar_lr", help="[Muon] transformer scalar 绝对 LR (NanoChat 默认 0.04)", type=float, default=-1)
    parser.add_argument("--adamw_dmodel_lr_scaling", help="[Muon] 是否对 AdamW LR 做 dmodel scaling: lr × (dim/768)^-0.5", action="store_true", default=False)
    parser.add_argument("--muon_momentum_warmup_steps", help="[Muon] Momentum 从 0.85 线性预热到 muon_momentum 的步数 (NanoChat 默认 300, 设 0 禁用)", type=int, default=300)

    # Option 1: 分层学习率参数
    parser.add_argument("--layered_lr", help="[Option 1] 启用分层学习率，不同参数类型使用不同学习率", action="store_true", default=False)
    parser.add_argument("--embedding_lr_mult", help="[Option 1] embedding 层学习率倍数 (相对于 peak_lr)", type=float, default=0.3)
    parser.add_argument("--vocab_to_embed_lr_mult", help="[Option 1] vocab_to_embed 层学习率倍数", type=float, default=0.1)
    parser.add_argument("--scalar_lr_mult", help="[Option 1] transformer 标量参数学习率倍数", type=float, default=0.5)

    # Option 2: 动态 Weight Decay 参数
    parser.add_argument("--dynamic_wd", help="[Option 2] 启用动态 Weight Decay，线性衰减到 0", action="store_true", default=False)

    # Option 3: Linear Warmdown LR 调度参数
    parser.add_argument("--linear_warmdown", help="[Option 3] 启用 NanoChat 风格的 Linear Warmdown LR 调度", action="store_true", default=False)
    parser.add_argument("--warmup_ratio", help="[Option 3] warmup 占总步数的比例 (NanoChat 默认 0.0)", type=float, default=0.0)
    parser.add_argument("--warmdown_ratio", help="[Option 3] warmdown 占总步数的比例 (NanoChat 默认 0.5)", type=float, default=0.5)
    parser.add_argument("--final_lr_frac", help="[Option 3] 最终 LR 占 peak_lr 的比例 (NanoChat 默认 0.0)", type=float, default=0.0)

    #DATASET AND DATALOADER #########################################################

    # NOTE: --num_workers and --prefetch_factor have been removed.
    # The nanochat pretrain DataLoader hardcodes num_workers=0 because it is a
    # stateful IterableDataset with exact-resume cursor/doc_buffer ownership and
    # it also copies CUDA tensors inside __iter__(). pin_memory=False because
    # yielded pretrain batches are already on GPU.
    
    parser.add_argument("--dataset_name", help="dataset name", default="ucf101")

    parser.add_argument("--sudoku_ratio", help="Mixing ratio for sudoku_mixed dataset: P(sample from Sudoku v2) per step. The remainder samples from nanochat SFT.", type=float, default=0.6)

    # ── v3 sudoku training extensions (no-CoT scope) ────────────────────────────
    parser.add_argument(
        "--sudoku_ratio_schedule",
        help=(
            "v3 P3+P4: how `sudoku_ratio` evolves during training. "
            "`fixed` keeps the value passed to --sudoku_ratio constant (v2 behavior). "
            "`three_phase` uses warmup 0.6 / focus 0.85 / consolidate 0.7. "
            "`three_phase_with_guard` adds a retention guard that pulls the ratio back "
            "if `valid_loss_sft` drifts > 0.03 above the v2 baseline (1.121)."
        ),
        type=str,
        choices=["fixed", "three_phase", "three_phase_with_guard"],
        default="fixed",
    )
    parser.add_argument(
        "--sudoku_blank_loss_weight",
        help=(
            "v3 P1: blank-position loss weighting K. Targets corresponding to blank "
            "cells in the input puzzle get CE × K; givens & non-cell tokens get × 1. "
            "K=1.0 (default) disables the weighting."
        ),
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sudoku_difficulty_schedule",
        help=(
            "v3 P5: difficulty-aware sampling over RRN-train. `fixed` = uniform "
            "(legacy v2). `three_phase` upweights the 17-22 hint bucket to 0.55 in "
            "the focus phase, then relaxes to 0.40 in consolidation."
        ),
        type=str,
        choices=["fixed", "three_phase"],
        default="fixed",
    )
    parser.add_argument(
        "--base_train_dataset",
        help="local base pretraining corpus used by the nanochat parquet dataloader",
        choices=["fineweb", "climbmix", "dclm"],
        default=os.environ.get("BASE_TRAIN_DATASET", os.environ.get("NANOCHAT_BASE_TRAIN_DATASET", "fineweb")),
    )
    parser.add_argument(
        "--base_train_data_dir",
        help="optional override for the local parquet directory of --base_train_dataset",
        default=os.environ.get("BASE_TRAIN_DATA_DIR", ""),
    )
    
    parser.add_argument("--dataset_dir", help="dataset base directory", default="")

    # NanoChat shard evaluation specific parameters
    parser.add_argument("--eval_shard_indices", help="Comma-separated list of shard indices to evaluate (e.g., '0,15' for first and last)", type=str, default="0,15")
    parser.add_argument("--max_samples_per_shard", help="Maximum number of samples to evaluate per shard", type=int, default=50)
    parser.add_argument("--enable_nanochat_generation", help="Enable text generation evaluation for nanochat (in addition to PPL)", action="store_true", default=True)
    parser.add_argument("--generation_split_ratio", help="Ratio to split sequence for generation (0.5 = half prompt, half target)", type=float, default=0.5)
    parser.add_argument("--min_generation_length", help="Minimum token length for generation samples", type=int, default=64)

    parser.add_argument("--image_dims", help="List of image dimensions", nargs='+', type=int, default = [224, 224])
    
    parser.add_argument("--no_randomness_dataloader", help="makes dataloader have no randomness by only sampling from start", action="store_true", default=False)
        
    parser.add_argument("--preprocess_data", help="center crops images - helps to avoid black bars on sides which cause issues with unstable gradient. not used anymore", action="store_true", default=False)
    
    parser.add_argument("--crop_all_samples", help="if preprocess data is enabled this will crop all samples. not used anymore", action="store_true", default=False)
    
    parser.add_argument("--custom_image_normalization", help="whether to normalize data according to each dataset's std and mean", action="store_true", default=False)

    parser.add_argument("--use_rand_caption", help="whether to use a random caption for image generation rather than always the first caption", action="store_true", default=False)

    parser.add_argument("--ffprobe_path", help="path to ffprobe binary", default="")

    # IMPORTANT: For NanoChat dataset, this parameter is DOCUMENTATION ONLY!
    # The actual train/val split is HARDCODED in nanochat/dataloader.py:
    # - Training: parquet_paths[:-1] (all files except the last one)
    # - Validation: parquet_paths[-1:] (only the last file)
    # With 370 total shards, this gives 369 train + 1 val = 0.27% validation
    # This parameter does NOT control the split for NanoChat!
    parser.add_argument("--validation_split_pct", help="if training and validation set are being split - which percent is validation (IGNORED for nanochat dataset)", type=float, default=0.1)
    
    parser.add_argument("--test_split_pct", help="if training, validation, and test set are being split - which percent is test. by default this is not used", type=float, default=0)
    
    parser.add_argument("--limit_train_batches", help="percent of training dataset to use, or if > 1 is num batches", type=float, default=1.0)
    
    parser.add_argument("--limit_val_batches", help="percent of validation dataset to use, or if > 1 is num batches", type=float, default=1.0)
    
    parser.add_argument("--limit_test_batches", help="percent of testing dataset to use, or if > 1 is num batches", type=float, default=1.0)
    
    parser.add_argument("--val_sanity", help="number of sanity validation steps", type=int, default=0)
    
    parser.add_argument("--val_check_interval", help="interval to do validation per training epoch, 1 means once per epoch. useful if epochs are too large to wait to do validation", type=float, default=1.0)

    parser.add_argument("--val_every_n_step", help="interval to do validation in steps", type=int, default=15000)

    parser.add_argument("--val_after_n_step", help="how many steps to wait before doing validation. is useful when you have a small dataset and dont want to do validation every time", type=int, default=0)

    parser.add_argument("--val_steps", help="how many steps of validation", type=int, default=1000)

    #TESTING / INFERENCE#########################################################

    parser.add_argument("--run_testing_after_training", help="evaluate on test dataset. by default is off since repo is mostly pre-training, and may not work :/", action="store_true", default=False)

    parser.add_argument("--only_test", help="just testing no training, used passed in model checkpoint", action="store_true", default=False)

    parser.add_argument("--only_test_model_ckpt", help="model ckpt when only testing", type=str, default=None)
    
    parser.add_argument("--infer_max_gen_len", help="[Inference] Maximum number of tokens/frames/sequence to generate", type=int, default=256)

    parser.add_argument("--infer_echo", help="[Inference] Include input prompt in the output", action="store_true", default=False)

    parser.add_argument("--infer_output_dir", type=str, default="./logs/inference")

    # NLP INFERENCE ########################################################################
    
    parser.add_argument("--infer_temp", help="[Inference] Sampling temperature (higher = more random)", type=float, default=0.6)
    
    parser.add_argument("--infer_topp", help="[Inference] Nucleus sampling probability threshold", type=float, default=0.9)
    
    parser.add_argument("--infer_topk", help="[Inference] Limit sampling to top k most likely tokens", type=int, default=None)
    
    parser.add_argument("--infer_logprobs", help="[Inference] Return log probabilities of generated tokens", type=bool, default=False)

    # VID INFERENCE ########################################################################

    parser.add_argument("--infer_video_condition_frames", help="[Inference] number of frames to condition generation on", type=int, default=3)

    parser.add_argument("--infer_generate_video", help="[Inference] whether to get FVD and FID metrics", action="store_true", default=False)

    # EBT SPECIFIC INFERENCE/TESTING PARAMS FOR ADVANCED INFERENCE ##################################################

    parser.add_argument("--infer_ebt_advanced", help="[Inference] Does advanced inference for EBT, including hparams for more mcmc steps, override alpha, generating many samples and choosing the best, langevin dynamics, etc", action="store_true", default=False)

    parser.add_argument("--infer_ebt_num_steps", help="[EBT Inference] Number of MCMC steps to take, if is time embed/adaln will do this number of steps for each mcmc discrete time step", type=int, default=1)

    parser.add_argument("--infer_ebt_override_alpha", help="[EBT Inference] Step size for MCMC, default is 0, if is nonzero will override the trained value", type=float, default=0)

    parser.add_argument("--infer_generated_samples", help="[EBT Inference] Number of generated samples to create per sample, and then choose using infer_energy_sampling_technique", type=int, default=1)

    parser.add_argument("--infer_debug_sample_distances", help="[Inference] Print the distances between samples when generating many samples when doing infer_generated_samples > 1", action="store_true", default=False)

    parser.add_argument("--infer_plot_energy_landscape", help="[Inference] Plot the energy landscape and the corresponding solution distances. didnt work very well unsure if is due to bug", action="store_true", default=False)
    
    parser.add_argument("--infer_langevin_dynamics_noise", help="[EBT Inference] Langevin dynamics noise during inference, default is 0", type=float, default=0)

    parser.add_argument("--infer_langevin_first_step", help="[Inference] Makes it so the langevin dynamics only applies to the first step", action="store_true", default=False)

    parser.add_argument("--infer_energy_sampling_technique", help="the way to sample from EBM, default is min which means choose the sample with the lowest energy, other options are max_gap which chooses the sample with highest gap from starting energy and lowest energy", choices = ["min", "max_gap", "max"], type=str, default="min")

    parser.add_argument("--infer_accept_lower_energies", help="[Inference] Makes it so only accept a step if it has lower energy", action="store_true", default=False)

    parser.add_argument("--infer_steps_final_landscape", help="[Inference] Makes it so infer_ebt_num_steps only affects the final energy landscape in the case of EBT with time_embed/adaln", action="store_true", default=False)

    parser.add_argument("--infer_alpha_final_landscape", help="[Inference] Makes it so infer_ebt_override_alpha only affects the final energy landscape in the case of EBT with time_embed/adaln", action="store_true", default=False)

    #WANDB##################################################################

    parser.add_argument("--no_wandb", help="no wandb", action="store_true", default=False)
    parser.add_argument("--disable_wandb", help="alias for --no_wandb, completely disables wandb", action="store_true", default=False)

    parser.add_argument("--wandb_entity", help="wandb entity", default='')

    parser.add_argument("--wandb_project", help="wandb project name", default='EBT')

    parser.add_argument("--wandb_tags", help="wandb tags to add", nargs='+', default=None)

    # parser.add_argument("--wandb_offline", help="set wandb to offline mode", action="store_true", default=False)
    parser.add_argument("--wandb_offline", help="set wandb to offline mode", action="store_true", default=True) # TODO

    parser.add_argument("--wandb_watch", help="turns on watch mode for wandb - expensive so only use for debugging", action="store_true", default=False)

    parser.add_argument("--wandb_watch_level", help="wandb watch log level: 'parameters' (default when --wandb_watch is on), 'gradients', or 'all' (most expensive)", type=str, default="parameters", choices=["parameters", "gradients", "all"])

    parser.add_argument("--wandb_watch_log_freq", help="number of steps to log for wandb watch. is higher since is a bit expensive", type = int, default=1000)

    #LOGGING##################################################################

    parser.add_argument("--debug_log_filename", help="filename of log, deprecated", default='debug.log')

    parser.add_argument("--console_log_filename", help="filename of log, used when no_wandb is active", default='console.log')

    parser.add_argument("--print_logs", help="[deprecated] print to console, default false", action="store_true", default=False)

    parser.add_argument("--log_model_archi", help="log model architecture", action="store_true", default=False)

    parser.add_argument("--log_gradients", help="logs gradients at every step to wandb to debug them", action="store_true", default=False)

    parser.add_argument("--log_every_n_steps", help="turns on logger freq via pl, not advised to use this", type=int, default=50)

    # CHECKPOINTING ##################################################################

    parser.add_argument("--resume_training_ckpt", help="checkpoint to resume training from, use absolute",type=str, default="")

    parser.add_argument("--resume_warmup_steps", type=int, default=0,
        help="Resume 后的 warmup 步数，LR 从 0 线性升到 schedule 值。0=不启用")     

    parser.add_argument("--checkpoint_monitor_string", help="string to use to monitor for saving checkpoint. supported by PL callback", type=str, default="valid_loss")

    parser.add_argument("--checkpoint_monitor_mode", help="monitoring mode for checkpoint_monitor_string, either ['min', 'max']. if is loss do min, if is a metric like accuracy do max", type=str, default="min")

    parser.add_argument("--save_top_k_ckpts", help="number of ckpts to save when doing val (saves the ones with best metrics using checkpoint monitor string and mode defined). -1 means save all", type=int, default=10)

    parser.add_argument("--checkpoint_dir", type=str, default="",
        help="Override checkpoint directory (default: ./logs/checkpoints/{run_name})")
    parser.add_argument("--wandb_save_dir", type=str, default="logs/",
        help="Override WandB save directory")

    parser.add_argument("--save_periodic_steps", type=int, default=0,
        help="Save checkpoint every N training steps regardless of val_loss (0=disabled). Useful for SFT where val_loss may rise while task performance improves.")

    parser.add_argument("--checkpoint_min_free_gb", type=float, default=10.0,
        help="Minimum free disk space in GiB to reserve before saving a checkpoint.")

    parser.add_argument("--checkpoint_preflight_required", action="store_true", default=False,
        help="Before training, fail if the checkpoint directory cannot hold the configured checkpoint count plus the free-space reserve.")

    parser.add_argument("--disable_checkpoint_cleanup_on_low_space", action="store_true", default=False,
        help="Disable automatic deletion of old checkpoints when free disk space is below --checkpoint_min_free_gb.")

    #PRECISION#########################################################################

    parser.add_argument("--set_matmul_precision", help="set math mult precision - \"medium\", \"high\", or \"highest\" ", default=None)

    parser.add_argument("--float_precision", help="float precision, pl recommends 16-mixed/bf16-mixed, also has by default 32-true", type=str, default="32-true")

    # SPEED ##################################################################

    parser.add_argument("--compile_model", help="compiles the model using torch.compile", action="store_true", default=False)
    parser.add_argument("--compile_mode", help="torch.compile 模式: full (编译整个模型), transformer_only (仅编译 transformer，推荐), disabled", type=str, default="transformer_only", choices=["full", "transformer_only", "disabled"])
    parser.add_argument("--compile_backend", help="torch.compile 后端: inductor (默认), eager, aot_eager", type=str, default="inductor")
    parser.add_argument("--compile_dynamic", help="允许动态形状 (可能降低加速效果)", action="store_true", default=False)

    parser.add_argument("--gradient_checkpointing", help="启用 gradient checkpointing 以节省显存 (用计算换显存)", action="store_true", default=False)
    parser.add_argument("--use_sdpa_attention", help="使用 SDPA 注意力变体 (ar_ebt_time_embed_sdpa_math.py), 大幅减少 Path B 中间张量显存占用", action="store_true", default=False)

    #SLURM#########################################################################

    parser.add_argument("--is_slurm_run", help="please set to true if doing slurm run, as of now just stops capturing console logs", action="store_true", default=False)
    
    parser.add_argument("--override_slurm_checks", help="dont use slurm checks to assert that certain conditions are true (i.e. train/test limit_batches is default value)", action="store_true", default=False)
    
    #DEBUGGING#######################################################################

    parser.add_argument("--debug_mode", help="turns debug mode on where dataset returned is very small, no_wandb is on and detect anomaly is on", action="store_true", default=False)

    parser.add_argument("--fast_dev_run", help="turns fast_dev_run for trainer on, makes it just do one training epoch and one val epoch", action="store_true", default=False)

    # parser.add_argument("--debug_dataloader", help="makes mode to just debug dataloader", action="store_true", default=False)

    parser.add_argument("--overfit_batches", help="if nonzero will overfit to specified num/percent of batches", type=float, default=0.0)

    parser.add_argument("--profiler", choices=["simple", "advanced", "pytorch"], type=str, default="")

    parser.add_argument("--no_shuffle", help="stops shuffling - helpful for debugging", action="store_true", default=False)

    parser.add_argument("--create_model_viz", help="creates model for visualization", action="store_true", default=False)

    parser.add_argument("--detect_anomaly", help="turns on anomaly detection mode", action="store_true", default=False)

    parser.add_argument("--find_unused_parameters", help="turns on pl find unused params mode - DO NOT KEEP ON for actual training, helpful if want to debug. this uses DDPStrategy and ignores distributed_strategy", action="store_true", default=False)

    parser.add_argument("--debug_unused_parameters", help="makes it so it tracks which params are used to find the params that are causing the unused params issue. need to do some things in base_model_trainer so ctrl f this hparam to see the NOTEs", action="store_true", default=False)

    parser.add_argument("--manual_gc_collect_every_n_steps", help="manually call gc collect every n steps, can be done to prevent CPU RAM memory 'leak'", type = int, default=-1)

    parser.add_argument("--debug_videos", help="debug generated videos in a grid", action="store_true", default=False)


    ########################################################################

    args = parser.parse_args()
    main(args)
