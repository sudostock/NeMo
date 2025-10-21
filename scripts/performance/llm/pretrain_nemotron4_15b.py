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

# TEMPORARY WORKAROUND - Remove in next release when upstream srun issue is fixed
from ..slurm_exit_code_override import *  # Monkey-patch for false-positive job failures

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import nemo_run as run

from nemo.collections.llm.recipes.nemotron4_15b import pretrain_recipe
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin

from ..argument_parser import parse_additional_slurm_params, parse_cli_args
from ..executors import runai_executor, slurm_executor
from ..helpers import (
    args_sanity_check,
    build_perf_env_plugin,
    build_torch_profiler_plugin,
    get_user_configs,
    logging,
    set_exp_logging_configs,
    set_primary_perf_configs,
)
from ..utils import get_comm_overlap_callback_idx, hf_tokenizer


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
    recipe = pretrain_recipe(performance_mode=True)
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
        use_mcore_fsdp=args.use_mcore_fsdp,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
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
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=256000
        )
    recipe.model.tokenizer = recipe.data.tokenizer

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    if args.cluster_type == "runai":
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_bootstrap_backend = "nccl"
    if gpu_type in ["b200", "gb200"]:
        tp_comm_overlap_cfg = userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096
        # needed as tp_overlap_configs.userbuffers are dataclass objects which are unserializable
        tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg

    if args.compute_dtype.lower() == "bf16" and args.checkpoint_load_path is not None:
        recipe.optim.config.use_precision_aware_optimizer = False

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    # Parse additional SLURM parameters if provided
    additional_slurm_params = None
    if hasattr(args, 'additional_slurm_params') and args.additional_slurm_params:
        additional_slurm_params = parse_additional_slurm_params(args.additional_slurm_params)

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
            additional_slurm_params=additional_slurm_params,
        )

    plugins = [build_perf_env_plugin(args, pp_size=pp_size)]

    if args.enable_nsys:
        plugins.append(
            NsysPlugin(
                start_step=args.profiling_start_step,
                end_step=args.profiling_stop_step,
                ranks=list(range(num_nodes * args.gpus_per_node)),
                nsys_gpu_metrics=args.profiling_gpu_metrics,
            )
        )

    if torch_profiler_plugin := build_torch_profiler_plugin(args):
        plugins.append(torch_profiler_plugin)

    if args.enable_memory_profile:
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
