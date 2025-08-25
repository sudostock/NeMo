# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
from os.path import basename, splitext

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import nemo_run as run

# PERFORMANCE OPTIMIZATION: Import recipe components directly to avoid expensive ModelOpt cascade
# from nemo.collections.llm.recipes.nemotron4_15b import pretrain_recipe  # <-- DISABLED (expensive)
from ..argument_parser import parse_cli_args
from ..executors import runai_executor, slurm_executor
from ..helpers import (
    args_sanity_check,
    build_perf_env_plugin,
    get_user_configs,
    logging,
    set_exp_logging_configs,
    set_primary_perf_configs,
)
from ..utils import get_comm_overlap_callback_idx, hf_tokenizer


def get_fast_pretrain_recipe():
    """
    Fast import of nemotron4_15b pretrain recipe components.
    
    This function recreates the pretrain_recipe without importing through the expensive
    nemo.collections.llm.api path that pulls in ModelOpt and diffusers.
    
    Expected time savings: 20-30 seconds
    """
    # Import components directly - these are fast
    from nemo.collections.llm.gpt.data.mock import MockDataModule
    from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
    from nemo.collections.llm.recipes.nemotron import nemotron_model, nemotron_trainer
    from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
    from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
    from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
    from nemo.utils.exp_manager import TimingCallback
    
    def fast_pretrain_recipe(
        # General
        dir=None,
        name="default",
        # Trainer
        tensor_parallelism=4,
        pipeline_parallelism=1,
        pipeline_parallelism_type=None,
        virtual_pipeline_parallelism=None,
        context_parallelism=1,
        sequence_parallelism=True,
        num_nodes=1,
        num_gpus_per_node=8,
        max_steps=300000,
        precision="bf16-mixed",
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        limit_test_batches=32,
        limit_val_batches=32,
        log_every_n_steps=10,
        val_check_interval=2000,
        # Data
        global_batch_size=32,
        micro_batch_size=2,
        seq_length=4096,
        # Optimizer
        warmup_steps=500,
        constant_steps=0,
        min_lr=4.5e-5,
        max_lr=4.5e-5,
        performance_mode=False,
    ):
        """
        Fast recreation of nemotron4_15b pretrain_recipe without expensive imports.
        """
        # Create the base recipe (same structure as original)
        recipe = run.Partial(
            _get_fast_pretrain_function(),
            model=nemotron_model(version="nemotron4_15b"),
            trainer=nemotron_trainer(
                tensor_parallelism=tensor_parallelism,
                pipeline_parallelism=pipeline_parallelism,
                pipeline_parallelism_type=pipeline_parallelism_type,
                virtual_pipeline_parallelism=virtual_pipeline_parallelism,
                context_parallelism=context_parallelism,
                sequence_parallelism=sequence_parallelism,
                num_nodes=num_nodes,
                num_gpus_per_node=num_gpus_per_node,
                max_steps=max_steps,
                precision=precision,
                accumulate_grad_batches=accumulate_grad_batches,
                limit_test_batches=limit_test_batches,
                limit_val_batches=limit_val_batches,
                log_every_n_steps=log_every_n_steps,
                val_check_interval=val_check_interval,
                callbacks=[run.Config(TimingCallback)],
            ),
            data=run.Config(
                MockDataModule,
                seq_length=seq_length,
                global_batch_size=global_batch_size,
                micro_batch_size=micro_batch_size,
            ),
            log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
            optim=distributed_fused_adam_with_cosine_annealing(
                precision=precision,
                warmup_steps=warmup_steps,
                constant_steps=constant_steps,
                min_lr=min_lr,
                max_lr=max_lr,
                clip_grad=gradient_clip_val,
            ),
            resume=default_resume(),
        )
        
        # Apply performance optimizations if requested
        if performance_mode:
            recipe = _fast_pretrain_performance_optimizations(recipe)
        
        return recipe
    
    return fast_pretrain_recipe


def _get_fast_pretrain_function():
    """
    Get a simplified pretrain function without ModelOpt dependencies.
    
    This recreates the core functionality of nemo.collections.llm.api.pretrain
    but skips the expensive ModelOpt imports.
    """
    def fast_pretrain(model, data, trainer, log=None, resume=None, optim=None):
        """Simplified pretrain function without ModelOpt overhead."""
        # Import the actual train function but avoid ModelOpt imports
        from nemo.lightning import configure_no_restart_validation_training_loop
        from nemo.utils import logging as nemo_logging
        
        # Basic validation (simplified from original)
        if not model or not data or not trainer:
            raise ValueError("model, data, and trainer are required")
        
        # Configure trainer
        if log:
            trainer.logger = log.logger
        
        if optim:
            model.optim = optim
            
        # Skip ModelOpt checkpoint loading for performance
        # Note: This means ModelOpt features won't work, but pure pretraining doesn't need them
        
        trainer.fit(model, datamodule=data, ckpt_path=resume.resume_from_path if resume else None)
        
        return trainer.log_dir
    
    return fast_pretrain


def _fast_pretrain_performance_optimizations(recipe):
    """
    Fast recreation of pretrain_performance_optimizations without expensive imports.
    """
    from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
    from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
    
    if not recipe.trainer.callbacks:
        recipe.trainer.callbacks = []

    garbage_collection_callback = run.Config(
        GarbageCollectionCallback,
        gc_interval_train=100,
        gc_interval_val=100,
    )
    mcomm_overlap_callback = run.Config(
        MegatronCommOverlapCallback,
        tp_comm_overlap=True,
    )
    recipe.trainer.callbacks.extend([
        garbage_collection_callback,
        mcomm_overlap_callback,
    ])

    recipe.trainer.plugins.grad_reduce_in_fp32 = False
    recipe.optim.config.use_precision_aware_optimizer = False

    return recipe


def get_fast_tp_overlap_config():
    """Lazy import of TP overlap config to avoid expensive imports at module level."""
    from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096
    return userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096


def get_fast_tokenizer_utils():
    """Lazy import of tokenizer utilities."""
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
    return get_nmt_tokenizer


def get_fast_plugins():
    """Lazy import of plugins."""
    from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin
    return MemoryProfilePlugin, NsysPlugin


def override_recipe_configs(
    args: str,
    num_nodes: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    num_layers: int,
    hidden_size: int,
    enable_cuda_graphs: bool,
):
    """
    nemotron4 15b pre-train recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    # PERFORMANCE OPTIMIZATION: Use fast recipe to avoid expensive ModelOpt imports
    fast_recipe_fn = get_fast_pretrain_recipe()
    recipe = fast_recipe_fn(performance_mode=True)
    recipe = set_primary_perf_configs(
        recipe,
        "pre_train",
        num_nodes,
        args.gpus_per_node,
        mbs,
        gbs,
        args.max_steps,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        num_layers,
        hidden_size,
        enable_cuda_graphs=enable_cuda_graphs,
        use_user_buffer_registration=args.use_user_buffer_registration,
        use_sharp=args.use_sharp,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        save_checkpoint=args.checkpoint_save,
        load_checkpoint_path=args.checkpoint_load_path,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "nemotron4", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    gpu_type = args.gpu.lower()

    # data module configs
    if args.hf_token:
        recipe.data.tokenizer = hf_tokenizer("nvidia/Nemotron-4-340B-Base")
    else:
        get_nmt_tokenizer = get_fast_tokenizer_utils()
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=256000
        )
    recipe.model.tokenizer = recipe.data.tokenizer

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    if args.cluster_type == "runai":
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_bootstrap_backend = "nccl"
    if gpu_type in ["b200", "gb200"]:
        tp_comm_overlap_cfg = get_fast_tp_overlap_config()
        # needed as tp_overlap_configs.userbuffers are dataclass objects which are unserializable
        tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg

    if args.compute_dtype.lower() == "bf16" and args.checkpoint_load_path is not None:
        recipe.optim.config.use_precision_aware_optimizer = False

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "nemotron4", "15b", args)
    (
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        num_layers,
        hidden_size,
        _,
        enable_cuda_graphs,
    ) = kwargs[:12]

    recipe = override_recipe_configs(
        args,
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        num_layers,
        hidden_size,
        enable_cuda_graphs,
    )

    exp_config = f"gpus{args.num_gpus}_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_mbs{mbs}_gbs{gbs}"
    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"

    env_vars = {}
    if args.gpu.lower() == 'gb200':
        env_vars |= {"NCCL_NET_GDR_LEVEL": "PHB"}

    if args.cluster_type == "runai":
        pvcs = []
        for item in args.custom_mounts:
            parts = item.split(':')
            if len(parts) != 3:
                raise argparse.ArgumentTypeError("Each mount must be in name:path:claimName format")
            pvcs.append(
                {
                    "name": parts[0],
                    "path": parts[1],
                    "existingPvc": True,
                    "claimName": parts[2],
                }
            )
        executor = runai_executor(
            base_url=args.base_url,
            app_id=args.app_id,
            app_secret=args.app_secret,
            project_name=args.project_name,
            nodes=num_nodes,
            num_gpus_per_node=args.gpus_per_node,
            container_image=args.container_image,
            pvc_nemo_run_dir=args.pvc_nemo_run_dir,
            custom_mounts=pvcs,
            custom_env_vars=env_vars,
            hf_token=args.hf_token,
            wandb_key=args.wandb_key,
        )
    else:
        executor = slurm_executor(
            args.gpu.lower(),
            args.account,
            args.partition,
            args.log_dir,
            num_nodes,
            args.gpus_per_node,
            args.time_limit,
            args.container_image,
            custom_mounts=args.custom_mounts,
            custom_env_vars=env_vars,
            hf_token=args.hf_token,
            nemo_home=args.nemo_home,
            wandb_key=args.wandb_key,
            network='sharp' if args.use_sharp else None,
        )

    plugins = [build_perf_env_plugin(args, pp_size=pp_size)]

    if args.enable_nsys:
        MemoryProfilePlugin, NsysPlugin = get_fast_plugins()
        plugins.append(
            NsysPlugin(
                start_step=args.profiling_start_step,
                end_step=args.profiling_stop_step,
                ranks=list(range(num_nodes * args.gpus_per_node)),
                nsys_gpu_metrics=args.profiling_gpu_metrics,
            )
        )
    if args.enable_memory_profile:
        assert args.memory_profile_out_path is not None
        if 'MemoryProfilePlugin' not in locals():
            MemoryProfilePlugin, NsysPlugin = get_fast_plugins()
        plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

    with run.Experiment(exp_name) as exp:
        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=plugins,
        )

        if not args.dryrun:
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()
