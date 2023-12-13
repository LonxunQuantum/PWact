import argparse
import warnings
import os
import sys
import torch

def set_train_param(params):
    param_list = params.split()
    res = []
    for i in param_list:
        if ' ' in res:
            continue
        else:
            res.append(i)
    return res

def parser_input_args():
    parser = argparse.ArgumentParser(description="PyTorch MLFF Training")
    parser.add_argument(
        "--datatype",
        default="float64",
        type=str,
        help="Datatype and Modeltype default float64",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs", default=30, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        type=int,
        metavar="N",
        help="mini-batch size (default: 1), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume",
        action="store_true",
        help="resume the latest checkpoint",
    )
    parser.add_argument(
        "-s" "--store-path",
        default="default",
        type=str,
        metavar="STOREPATH",
        dest="store_path",
        help="path to store checkpoints (default: 'default')",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://localhost:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "-wt",
        "--work-type",
        default="train",
        type=str,
        help="work type: train/valid/kpu",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--magic", default=2022, type=int, help="Magic number. ")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--dp", dest="dp", action="store_true", help="Whether to use DP, default False."
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    # parser.add_argument("--dummy", action="store_true", help="use fake data to benchmark")
    parser.add_argument(
        "-n", "--net-cfg", default="DeepMD_cfg_dp_kf", type=str, help="Net Arch"
    )
    parser.add_argument("--act", default="sigmoid", type=str, help="activation kind")
    parser.add_argument(
        "--opt", default="ADAM", type=str, help="optimizer type: LKF, GKF, ADAM, SGD"
    )
    parser.add_argument(
        "--Lambda", default=0.98, type=float, help="KFOptimizer parameter: Lambda."
    )
    parser.add_argument(
        "--nue", default=0.99870, type=float, help="KFOptimizer parameter: Nue."
    )
    parser.add_argument(
        "--blocksize", default=10240, type=int, help="KFOptimizer parameter: Blocksize."
    )
    parser.add_argument(
        "--nselect", default=24, type=int, help="KFOptimizer parameter: Nselect."
    )
    parser.add_argument(
        "--groupsize", default=6, type=int, help="KFOptimizer parameter: Groupsize."
    )
    return parser

def set_args(parser, train_config = None):
    pram_list = set_train_param(train_config) if train_config is not None else None
    # sys.argv.extend(pram_list)
    args = parser.parse_args(pram_list)
    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely \
            disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # if not os.path.exists(args.store_path): create when at the using step
    #     os.mkdir(args.store_path)
    #     print(args.store_path)
    
    if torch.cuda.is_available():
        args.ngpus_per_node = torch.cuda.device_count()
    else:
        args.ngpus_per_node = 1

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size

    args.davg_dstd_dir = None  #davg_dstd_dir is used for dpkf data scaling
    args.percent_kpu = 1 #1:100% 2:50% 4:25%-> img_idx % precent_kpu == 0: do kpu caculating.
    return args
