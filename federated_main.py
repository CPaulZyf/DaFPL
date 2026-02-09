import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from Dassl.dassl.utils import setup_logger, set_random_seed, collect_env_info
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer

import copy
from prettytable import PrettyTable
import numpy as np
import torch.optim as optim
import random

from fed_utils import average_weights, count_parameters

from clip.discriminator import DomainDiscriminator

import open_clip
print(open_clip.list_pretrained())

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.PROMPTFL.CSC = False  # class-specific context
    cfg.TRAINER.PROMPTFL.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.PROMPTFL.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PROMPTFL.FEATURE = args.feature

    ctx_vectors = torch.empty(args.n_ctx, 512, dtype=torch.float32)
    nn.init.normal_(ctx_vectors, std=0.02)
    cfg.ctx_vectors = [ctx_vectors]
    cfg.num_domain = args.num_domain
    cfg.domain_tokens = args.domain_tokens
    cfg.lambda_dom = args.lambda_dom
    cfg.lambda_unify = args.lambda_unify

    cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample  # all, base or new
    cfg.DATASET.USERS = args.num_users
    cfg.DATASET.DOMAINUSERS = args.domain_num_users# number of clients
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.USEALL = args.useall # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.NAME = args.dataset
    cfg.DATASET.REPEATRATE = 0.0 # repeat rate on each client
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = args.num_domain # number of domain
    cfg.DATASET.IMBALANCE_TRAIN = args.imbalance_train # is adding label skew to feature skew datasets
    cfg.DATASET.SPLIT_CLIENT = args.split_client # is adding label skew to feature skew datasets and split one domain to multi clients
    cfg.OPTIM.ROUND = args.round # global round
    cfg.OPTIM.MAX_EPOCH = args.local_epoch # local epoch
    cfg.OPTIM.GAMMA = args.gamma # gamma of single-step
    cfg.OPTIM.LR = args.lr #learning rate
    cfg.OPTIM.GPU = args.gpu

    cfg.MODEL.BACKBONE.PRETRAINED = True

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size


    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    print_args(args, cfg)
    client_acc = [[] for _ in range(args.num_users)]

    local_trainers = defaultdict(list)
    global_weights = defaultdict(list)
    for i, j  in  enumerate(args.models):
        cfg.MODEL.BACKBONE.defrost()
        cfg.MODEL.BACKBONE.NAME = j
        cfg.TRAINER.PROMPTFL.freeze()
        local_trainer_ = build_trainer(cfg)
        count_parameters(local_trainer_.model, "prompt_learner")
        local_trainers[i] = local_trainer_
        global_weights[i] = copy.deepcopy(local_trainer_.model.state_dict())
        global_weights[i]['prompt_learner.ctx'] = copy.deepcopy(global_weights[0]['prompt_learner.ctx'])

    datanumber_client = []

    for net_i in range(cfg.DATASET.USERS):
        datanumber_client.append(len(local_trainers[0].fed_train_loader_x_dict[net_i].dataset))

    local_weights_prompt = defaultdict(list)

    for net_i in range(cfg.DATASET.USERS):
        local_weights_prompt[net_i] = copy.deepcopy(global_weights[0]['prompt_learner.ctx'])

    global_weights_prompt_domains = [copy.deepcopy(global_weights[0]['prompt_learner.ctx'][args.domain_tokens:]) for _ in
                                     range(args.num_domain)]
    global_weights_prompt_shared = copy.deepcopy(global_weights[0]['prompt_learner.ctx'][:args.domain_tokens])

    discriminator = DomainDiscriminator(global_weights[0]['prompt_learner.ctx'].shape[1], args.num_domain)
    discriminator.to(local_trainers[0].device)
    discriminator.eval()
    for p in discriminator.parameters():
        p.requires_grad_(False)
    optim_discriminator = optimizer = torch.optim.SGD(
        discriminator.parameters(),
        lr=args.dis_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    local_weights_prompt_domains = defaultdict(list)
    local_weights_prompt_shared = defaultdict(list)

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    global_test_acc_list = []
    global_epoch_list = []
    clientid2domain = local_trainers[0].dm.dataset.clientid2domain
    for epoch in range(start_epoch, max_epoch):
        idxs_users = []
        for i in range(0, args.num_users, args.domain_num_users):
            idxs_users.extend(range(i, min(i + args.num_participant, args.num_users)))
        print("idxs_users", idxs_users)
        print("------------local train start epoch:", epoch, "-------------")
        domain_users = [[] for _ in range(args.num_domain)]
        for idx in idxs_users:
            global_weights[clientid2domain[idx]]['prompt_learner.ctx'] = torch.cat([global_weights_prompt_shared,
                                                              global_weights_prompt_domains[clientid2domain[idx]]], dim=0)
            domain_users[clientid2domain[idx]].append(idx)
            local_trainers[clientid2domain[idx]].model.load_state_dict(copy.deepcopy(global_weights[clientid2domain[idx]]))
            local_trainers[clientid2domain[idx]].train(idx=idx, global_epoch=epoch, is_fed=False,
                                global_weight=global_weights_prompt_shared, fedprox=True,
                                mu=0.5, discriminator=discriminator)
            local_weight = local_trainers[clientid2domain[idx]].model.state_dict()
            local_weights_prompt[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'])
            local_weights_prompt_shared[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][:args.domain_tokens])
            local_weights_prompt_domains[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][args.domain_tokens:])
        print("------------local train finish epoch:", epoch, "-------------")

        #### Classifier in Server
        discriminator.train()
        for p in discriminator.parameters():
            p.requires_grad_(True)

        def build_batch(local_weights_prompt_domains, clientid2domain, client_ids):
            prompt_domains = []
            labels = []
            for idx in client_ids:
                if idx not in local_weights_prompt_domains:
                    continue
                if idx not in clientid2domain:
                    continue
                prompt_domains.append(local_weights_prompt_domains[idx].mean(dim=0))
                labels.append(int(clientid2domain[idx]))
            X = torch.stack(prompt_domains, dim=0)  # [B, in_dim]
            y = torch.tensor(labels, device=X.device, dtype=torch.long)  # [B]
            return X, y
        total_loss = 0.0
        for _ in range(args.dis_epoch):
            batch_cids = random.sample(idxs_users, k=len(idxs_users) // 2)
            X, y = build_batch(local_weights_prompt_domains, clientid2domain, batch_cids)

            logits = discriminator(X)
            loss = F.cross_entropy(logits, y)

            optim_discriminator.zero_grad(set_to_none=True)
            loss.backward()
            optim_discriminator.step()

            total_loss += float(loss.detach().cpu())

        discriminator.eval()
        for p in discriminator.parameters():
            p.requires_grad_(False)

        for domain_id, users in enumerate(domain_users):
            if len(users) > 0:
                global_weights_prompt_domains[domain_id] = average_weights(local_weights_prompt_domains,users, datanumber_client, islist=True)

        global_weights_prompt_shared = average_weights(local_weights_prompt_shared, idxs_users, datanumber_client, islist=True)

        print("------------local test start-------------")
        results = []
        for idx in idxs_users:
            local_weights = copy.deepcopy(global_weights[clientid2domain[idx]])
            local_weights['prompt_learner.ctx'] = local_weights_prompt[idx]
            local_trainers[clientid2domain[idx]].model.load_state_dict(local_weights,strict=False)
            acc_idx = local_trainers[clientid2domain[idx]].test(idx=idx)
            client_acc[idx].append(acc_idx[0])
            results.append(acc_idx)
        global_test_acc = []
        for k in range(len(results)):
            global_test_acc.append(results[k][0])
        global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
        global_epoch_list.append(epoch)
        print("------------local test finish-------------")
        for i in idxs_users:
            print('client', i, 'local acc', client_acc[i])
            print('client', i, 'max acc', max(client_acc[i]))
        print("Global test acc:", global_test_acc_list)
        print('Global max acc', max(global_test_acc_list))
        print("Epoch on server :", epoch)

    print("global_test_acc_list:",global_test_acc_list)
    print("maximum test acc:", max(global_test_acc_list))
    return global_test_acc_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument("--model", type=str, default="DaFPL", help="model")
    parser.add_argument('--partition', type=str, default='hom',
                        help='the data partitioning strategy, select from "homo, dir"')
    parser.add_argument("--dataset", type=str, default="PACS",
                        help="[Office, PACS, OfficeHome, DomainNet]")
    parser.add_argument('--domain_num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--domain_tokens', type=int, default=8, help="number of domain token")
    parser.add_argument("--num_participant", type=int, default=5,
                        help="load model weights at this epoch for evaluation")
    parser.add_argument("--lambda_dom", type=float, default=0.1)
    parser.add_argument("--lambda_unify", type=float, default=0.01)

    parser.add_argument("--tau", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument('--dis_lr', type=float, default=0.01, help='learning rate for discriminator')
    parser.add_argument('--dis_epoch', type=int, default=50, help='epoch for discriminator')

    parser.add_argument("--trainer", type=str, default="DaFPL")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--n_ctx', type=int, default=16
                        , help="number of text encoder of text prompts")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--gamma', type=float, default=1, help='gamma of single_step')
    parser.add_argument('--iid', default=False, help="is iid")
    parser.add_argument('--subsample', type=str, default='base', help="all,base,new")
    parser.add_argument('--feature', default=False, help="is compute similarity between text feature and image feature map")
    parser.add_argument('--round', type=int, default=50, help="number of communication round")
    parser.add_argument('--beta', type=float, default=0.3,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--mu', type=float, default=0.1, help='The parameter for fedprox')
    parser.add_argument('--temp', type=float, default=0.5, help='The tempuature')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--num_prompt', type=int, default=1, help="number of prompts")
    parser.add_argument('--avg_prompt', type=int, default=1, help="number of prompts to average")
    parser.add_argument('--thresh', type=float, default=1e-3, help='the thresh of sinkhorn distance')
    parser.add_argument('--eps', type=float, default=0.1, help='the lambada of sinkhorn distance')
    parser.add_argument('--logits2', default=False, help="is caculate the similarity between text feature and image class token")
    parser.add_argument('--OT', type=str, default='COT', help="type of OT used: Sinkhorn, COT")
    parser.add_argument('--top_percent', type=float, default=1, help='the top_percent of COT')
    parser.add_argument('--max_iter', type=int, default=100, help="max iteration of COT")
    parser.add_argument('--imbalance_train', default=False, help="is adding label skew to feature skew datasets")
    parser.add_argument('--split_client', default=False, help="is adding label skew to feature skew datasets and split one domain to multi clients")
    parser.add_argument('--num_domain', type=int, default=4, help="number of domain")
    parser.add_argument('--ctx_init', default=False, help="is using the ctx init")

    parser.add_argument('--num_shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--bottleneck', type=int, default=4, help="number of middle in reparameter")
    parser.add_argument('--local_epoch', type=int, default=1, help="number of local epoch")
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")

    parser.add_argument('--train_batch_size', type=int, default=64, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=64, help="number of test batch size")

    parser.add_argument("--root", type=str, default="DATA/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="outputtest/", help="output directory")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="configs/model.yaml", help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/oxford_pets.yaml", help="path to config file for dataset setup")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    args = parser.parse_args()
    args.models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-B/32', 'ViT-B/16']

    if args.dataset == 'Office':
        args.num_domain = 4
    elif args.dataset == 'PACS':
        args.num_domain = 4
    elif args.dataset == 'OfficeHome':
        args.num_domain = 4
    elif args.dataset == 'Office31':
        args.num_domain = 3
    elif args.dataset == 'DomainNet':
        args.num_domain = 6
    args.models = args.models[:args.num_domain]
    args.num_users = args.domain_num_users * args.num_domain
    main(args)




