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

from nemo_run.config import get_nemorun_home

from .utils import DEFAULT_NEMO_HOME


def parse_cli_args():
    """
    Command line arguments correspong to Slurm cluster and NeMo2.0 for running pre-training and
    fine-tuning experiments.
    """
    parser = argparse.ArgumentParser(description="NeMo2.0 Performance Pretraining and Fine-Tuning")

    subparsers = parser.add_subparsers(dest="cluster_type", help='Type of cluster: slurm or runai')

    slurm_parser = subparsers.add_parser('slurm', help="define variables for slurm launcher")
    runai_parser = subparsers.add_parser('runai', help="define variables for runai launcher")

    slurm_parser.add_argument(
        "-a",
        "--account",
        type=str,
        help="Slurm account to use for experiment",
        required=True,
    )
    slurm_parser.add_argument(
        "-p",
        "--partition",
        type=str,
        help="Slurm partition to use for experiment",
        required=True,
    )
    slurm_parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        help=f"Directory for logging experiment results. Defaults to {get_nemorun_home()}",
        required=False,
        default=get_nemorun_home(),
    )
    slurm_parser.add_argument(
        "-t",
        "--time_limit",
        type=str,
        help="Maximum time limit to run experiment for. Defaults to 30 minutes (format- 'HH:MM:SS')",
        required=False,
        default="00:30:00",
    )
    runai_parser.add_argument(
        "-b",
        "--base_url",
        help="NVIDIA Run:ai API url to use for experiment. Should look like https://<base-url>/api/v1",
        type=str,
        required=True,
    )

    runai_parser.add_argument(
        "-id",
        "--app_id",
        help="Name of NVIDIA Run:ai Application",
        type=str,
        required=True,
    )
    runai_parser.add_argument(
        "-s",
        "--app_secret",
        help="NVIDIA Run:ai Application secret",
        type=str,
        required=True,
    )
    runai_parser.add_argument(
        "-p",
        "--project_name",
        help="NVIDIA Run:ai Project to run the experiment in",
        type=str,
        required=True,
    )
    runai_parser.add_argument(
        "-pd",
        "--pvc_nemo_run_dir",
        help="Directory path of your nemo-run home in Run:ai PVC",
        type=str,
        required=True,
    )
    container_img_msg = [
        "NeMo container to use for experiment. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'",
        "Make sure your NGC credentials are accessible in your environment.",
    ]
    parser.add_argument(
        "-i",
        "--container_image",
        type=str,
        help=" ".join(container_img_msg),
        required=False,
        default="nvcr.io/nvidia/nemo:dev",
    )
    parser.add_argument(
        "-c",
        "--compute_dtype",
        type=str,
        choices=["bf16", "fp8"],
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    fp8_recipe_msg = (
        "FP8 recipe. Options- ds (per-tensor delayed scaling), cs (per-tensor current scaling), mxfp8. Defaults to ds"
    )
    parser.add_argument(
        "-fr",
        "--fp8_recipe",
        type=str,
        choices=["ds", "cs", "mxfp8"],
        help=fp8_recipe_msg,
        required=False,
        default="ds",
    )
    parser.add_argument(
        "-en",
        "--enable_nsys",
        help="Enable Nsys profiling. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-nt",
        "--enable_nccltrace",
        help="Enable NCCL tracing. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard",
        help="Enable tensorboard logging. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-wd",
        "--wandb",
        help="Enable wandb logging. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-wdk",
        "--wandb_key",
        type=str,
        help="wandb key. Needed for wandb logger projetion to server",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-wdp",
        "--wandb_prj_name",
        type=str,
        help="wandb project name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-wdj",
        "--wandb_job_name",
        type=str,
        help="wandb job name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-f",
        "--finetuning",
       help="Finetuning scheme. Only SFT (Supervised Fine-Tuning) is supported.",
        default='sft',
        const='sft',
    )
    parser.add_argument(
        "-hf",
        "--hf_token",
        type=str,
        help="HuggingFace token. Defaults to None. Required for accessing tokenizers and checkpoints.",
        default=None,
    )
    nemo_home_msg = [
        "Sets env var `NEMO_HOME` (on compute node using sbatch script)- directory where NeMo searches",
        "for models and checkpoints. This saves a lot of time (especially for bigger models) if checkpoints already",
        f"exist here. Missing files will be downloaded here from HuggingFace. Defaults to {DEFAULT_NEMO_HOME}",
    ]
    parser.add_argument(
        "-nh",
        "--nemo_home",
        type=str,
        help=" ".join(nemo_home_msg),
        default=DEFAULT_NEMO_HOME,
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        help="If true, prints sbatch script to terminal without launching experiment.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-tp",
        "--tensor_parallel_size",
        type=int,
        help="Intra-layer model parallelism. Splits tensors across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-pp",
        "--pipeline_parallel_size",
        type=int,
        help="Inter-layer model parallelism. Splits transformer layers across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cp",
        "--context_parallel_size",
        type=int,
        help="Splits network input along sequence dimension across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-vp",
        "--virtual_pipeline_parallel_size",
        type=int,
        help="Number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ep",
        "--expert_parallel_size",
        type=int,
        help="Distributes Moe Experts across sub data parallel dimension.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-et",
        "--expert_tensor_parallel_size",
        type=lambda x: int(x) if x is not None else None,
        nargs="?",
        const=None,
        help="Intra-layer tensor model parallelsm for expert layer. Splits tensors across GPU ranks.\
            Use -et/--expert_tensor_parallel_size <space> for None or -et/--expert_tensor_parallel_size <int>",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-mb",
        "--micro_batch_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-gb",
        "--global_batch_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        help="Target gpu type. Defaults to 'h100'.",
        required=False,
        default="h100",
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        help="Number of gpus.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-gn",
        "--gpus_per_node",
        type=int,
        help="Number of gpus per node. Defaults to 8",
        required=False,
        default=8,
    )
    parser.add_argument(
        "-ms",
        "--max_steps",
        type=int,
        help="Number of train steps. Defaults to 100",
        required=False,
        default=50,
    )

    def bool_arg(arg):
        if arg.lower() in ['true', '1', 't', 'yes', 'y']:
            return True
        elif arg.lower() in ['false', '0', 'f', 'no', 'n']:
            return False
        else:
            raise ValueError(f"Invalid value for boolean argument: {arg}")

    parser.add_argument(
        "-cg",
        "--cuda_graphs",
        help="Enable CUDA graphs. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,  # NOTE: DO NOT SET DEFAULT TO FALSE, IT WILL BE OVERRIDDEN BY THE RECOMMENDED MODEL CONFIGS
    )
    parser.add_argument(
        "-fsdp",
        "--use_mcore_fsdp",
        help="Enable Megatron Core (Mcore) FSDP. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-rl",
        "--recompute_layers",
        type=int,
        help="Number of Transformer layers to recompute, where all the intermediate "
        "activations of a Transformer layer are computed. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ol",
        "--activation_offload_layers",
        type=int,
        help="Number of Transformer layers to offload to the CPU memory. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--nccl_communicator_config_path",
        type=str,
        help="Path to NCCL communicator config yaml file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-nlay",
        "--num_layers",
        type=int,
        help="Sets number of model layers.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-hs",
        "--hidden_size",
        type=int,
        help="Sets hidden model size",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-pss", "--profiling_start_step", type=int, help="Defines start step for profiling", required=False, default=46
    )
    parser.add_argument(
        "-pso", "--profiling_stop_step", type=int, help="Defines start step for profiling", required=False, default=50
    )

    parser.add_argument(
        "-pgm",
        "--profiling_gpu_metrics",
        help="Enable nsys gpu metrics. Disabled by default.",
        action="store_true",
    )

    parser.add_argument(
        "-cps",
        "--checkpoint_save",
        type=bool_arg,
        help="When enabled will trigger checkpoint save operation at the end of training",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cpl",
        "--checkpoint_load_path",
        type=str,
        help="Path to checkpoint to load prior to training start",
        required=False,
        default=None,
    )

    def list_of_strings(arg):
        return arg.split(',')

    parser.add_argument(
        "-rm",
        "--recompute_modules",
        type=list_of_strings,
        help="Comma-separated string of modules to recompute. Defaults to None",
        required=False,
        default=None,
    )

    parser.add_argument(
        "-cm",
        "--custom_mounts",
        type=list_of_strings,
        help="Comma separated string of mounts. For Run:ai, each mount must be in name:path:k8s-claimName format",
        required=False,
        default=[],
    )


    parser.add_argument(
        "-ev",
        "--custom_env_vars",
        type=str,
        required=False,
        default={},
    )

    parser.add_argument(
        "-cpin",
        "--cpu_pinning",
        type=int,
        help="Enable CPU pinning to improve performance on some clusters by setting numbers of CPUs per task. Disabled by default",
        required=False,
        default=0,
    )

    return parser
