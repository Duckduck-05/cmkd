import os
import sys
import torch
import argparse
import numpy as np
import random
import wandb
from utils.helper import gen_data, train_network_distill, train_network_distill2, train_network_distill21, train_network_distill22, train_network_distill3, train_network_distill4, train_network_distill5, pre_train
# from utils.model import ImageNet, AudioNet
from utils.model_res import ImageNet, AudioNet
from utils.module import Tea, Stu
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def eval_overlap_tag(loader, device, args):
    stu_type = args.stu_type
    tea_type = 1 - stu_type
    # load teacher model
    tea_model = ImageNet(args).to(device) if tea_type == 0 else AudioNet(args).to(device)
    if stu_type == 1:
        arch = args.image_arch
        print(f'teacher:image ({args.image_arch}); student:audio({args.audio_arch})')
    elif stu_type == 0:
        arch = args.audio_arch
        print(f'teacher:audio ({args.audio_arch}); student:image({args.image_arch})')
    tea_model.load_state_dict(
        #change
        torch.load('./results/teacher_mod_' + str(tea_type) + '_' + arch + '_' + str(args.num_frame) + '_overlap.pkl',
                   map_location={"cuda:0": "cpu"}), strict=False)
    print(f'Finish Loading teacher model')

    net = ImageNet(args).to(device) if stu_type == 0 else AudioNet(args).to(device)
    tea = Tea().cuda()
    stu = Stu().cuda()
    optimizer = torch.optim.SGD([
            {'params': net.parameters()},
            {'params': tea_model.parameters()},
            {'params': tea.parameters()},
            {'params': stu.parameters()},
        ], lr=args.lr, momentum=0.9)

    net.fc
    if args.distill_type == 1:
        acc = train_network_distill(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, args, tea, stu)
    if args.distill_type == 2:
        acc = train_network_distill2(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, args, tea, stu)
    if args.distill_type == 3:
        acc = train_network_distill3(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, args, tea, stu)
    if args.distill_type == 4:
        acc = train_network_distill4(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, args, tea, stu)
    if args.distill_type == 5:
        acc = train_network_distill5(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, args, tea, stu)
    if args.distill_type == 21:
        acc = train_network_distill21(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, args, tea, stu)
    if args.distill_type == 22:
        acc = train_network_distill22(stu_type, tea_model, args.num_epochs, loader, net, device, optimizer, args, tea, stu)
    return acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # the parameters you might need to change
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--stu-type', type=int, default=0, help='the modality of student unimodal network, 0 for image, 1 for audio')
    parser.add_argument('--num-runs', type=int, default=1, help='num runs')
    parser.add_argument('--num-epochs', type=int, default=00, help='num epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--batch-size2', type=int, default=512, help='batch size for calculating the overlap tag')
    parser.add_argument('--num-workers', type=int, default=16, help='dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='lr')
    parser.add_argument('--num_frame', type=int, default=1)
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--audio_arch', type=str, default='resnet18')
    parser.add_argument('--image_arch', type=str, default='resnet18')
    parser.add_argument('--krc', type=float, default=0.0)
    parser.add_argument('--pre_train', default=0, help='pre_train student and teacher models')
    parser.add_argument('--cmkd', default=1, help='crossmodal knowledge distillation')
    parser.add_argument('--group', type=str, default='c2kd', help='group of experiments')
    parser.add_argument('--run_name', type=str, default='a2v', help='prefix run name of the experiment')
    parser.add_argument('--distill_type', type=int, default=2)

    args = parser.parse_args()

    wandb.login(key="365a2332ad390479c5a6bb01365f47f0f427f47f")
    wandb.init(entity= "cmkd" ,project="c2kd-ours",
                name=f"{args.run_name}_lr_{args.lr}_bs_{args.batch_size}_numepochs_{args.num_epochs}_stutype_{args.stu_type}_distill_type_{args.distill_type}",
                config=vars(args), group=args.group)

    print(args)

    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))
    loader = gen_data('./data', args.batch_size, args.num_workers, args)
    if args.pre_train:
        loader_fb = gen_data('./data', args.batch_size2, args.num_workers, args)
        pre_train(args.stu_type, loader, args.num_epochs, args.lr, device, args)

    if args.cmkd:
        log_np = np.zeros((args.num_runs, 4))
        for run in range(args.num_runs):
            set_random_seed(seed=run)
            print(f'Seed {run}')
            log_np[run, :] = eval_overlap_tag(loader, device, args)
        log_mean = np.mean(log_np, axis=0)
        log_std = np.std(log_np, axis=0)
        print(f'Finish {args.num_runs} runs')
        print(f'Student Val Acc {log_mean[0]:.3f} ± {log_std[0]:.3f} | Test Acc {log_mean[1]:.3f} ± {log_std[1]:.3f}')
        print(f'Teacher Val Acc {log_mean[2]:.3f} ± {log_std[2]:.3f} | Test Acc {log_mean[3]:.3f} ± {log_std[3]:.3f}')
        if args.stu_type == 1:
            arch = args.image_arch
            print(f'teacher:image ({args.image_arch}); student:audio({args.audio_arch})')
        elif args.stu_type == 0:
            arch = args.audio_arch
            print(f'teacher:audio ({args.audio_arch}); student:image({args.image_arch})')