import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from src.pre_data.data_loader_2type_multidirs import MovementDataset, get_ener_shift_multi_dirs
from src.model.dp import DP
from src.model.MLFF import MLFFNet

from src.optimizer.LKF import LKFOptimizer
from src.optimizer.GKF import GKFOptimizer

from active_learning.util import get_recent_model

def get_model(device, ener_shift_path, args, ngpus_per_node, work_dir, \
        training_type = torch.float32, load_model_sign = False, dropout = False):

    if args.dp:
        ener_shift = get_ener_shift_multi_dirs(ener_shift_path)
        # model = DP(device, opts.opt_net_cfg, opts.opt_act, ener_shift, opts.opt_magic, dropout)
        model = DP(device, args.net_cfg, args.act, ener_shift, args.magic, dropout)
        model = model.to(training_type)
    else:
        model = MLFFNet(device, training_type, dropout = dropout)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("using CPU, this will be slow")
        raise(Exception("GPU use error, no gpu no work, go to check?"))

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], find_unused_parameters=False
                )
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = model.cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.MSELoss().to(device)

    if args.opt == "LKF":
        optimizer = LKFOptimizer(
            model.parameters(),
            args.Lambda,
            args.nue,
            args.blocksize,
            device,
            training_type,
        )
    elif args.opt == "GKF":
        optimizer = GKFOptimizer(
            model.parameters(), args.Lambda, args.nue, device, training_type
        )
    elif args.opt == "ADAM":
        optimizer = optim.Adam(model.parameters(), args.lr)
    elif args.opt == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        print("Unsupported optimizer!")

    # optionally resume from a checkpoint
    if args.resume:
        load_p = None
        # file_name = os.path.join(work_dir.model_dir, "best.pth.tar")
        reuse, model_path, p_path = get_recent_model(work_dir.model_dir)
        if reuse:
            print("=> loading checkpoint '{}'".format(model_path))
            if args.gpu is None:
                checkpoint = torch.load(model_path)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(model_path, map_location=loc)

            args.start_epoch = checkpoint["epoch"]+1
            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            # scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    model_path, checkpoint["epoch"]
                )
            )
            load_p = torch.load(p_path)
            optimizer.set_kalman_P(load_p, checkpoint["kalman_lambda"], checkpoint["kalman_nue"])
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
    
    return model, optimizer, criterion
