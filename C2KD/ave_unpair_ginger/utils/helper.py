import random
import numpy as np
import torch
from utils.dataloader import get_loader
import os
import codecs
import csv
from copy import deepcopy
# from utils.model import ImageNet, AudioNet
from utils.model_res import ImageNet, AudioNet
from utils.dist_utils import *
import time
import torch.nn.functional as F
import geomloss
from utils.AVEDataset import AVEDataset, AVEDataset_aud, AVEDataset_img
# from utils.VGGSoundDataset import VGGSound
from torch.utils.data import DataLoader
import math
import wandb
from scipy.stats.stats import kendalltau
# from scipy.stats import wasserstein_distance
import ot

from functools import reduce
import operator

def count_trainable_params(model):
    c = 0
    for p in model.parameters():
        try:
            if p.requires_grad:
                c += reduce(operator.mul, list(p.size()))
        except:
            pass
    return c

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask



def adjust_lr(lr=1e-2, iter=None, max_iter=100, power=0.9, optimizer=None):
    cur_lr = 1e-4 + (lr - 1e-4)*((1-float(iter)/max_iter)**(power))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr



def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def evaluate(loader, device, net, in_type):
    # type == 0: image input; type == 1: audio input; type == 2: both input
    correct, v_loss, total, logits = 0, 0, 0, 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img_inputs, aud_inputs, labels = data['image'].to(device), data['audio'].to(device), data['label'].to(
                device)
            total += labels.size(0)
            if in_type == 0:
                outputs, _, _ = net(img_inputs)
            elif in_type == 1:
                outputs, _, _ = net(aud_inputs)
            elif in_type == 2:
                outputs, _, _ = net(img_inputs, aud_inputs)
            else:
                raise ValueError('the value of element in in_type_list should be 0,1,2\n')

            logits = torch.Tensor([F.softmax(outputs, dim=-1)[i][labels[i]] for i in range(outputs.size(0))]).sum()
            _, predicted = torch.max(outputs.detach(), 1)
            correct += (predicted == labels).sum().item()
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)  # fix to CE loss
            v_loss += loss.item()
    logits = logits / len(loader)
    v_loss = v_loss / len(loader)
    val_acc = 100 * correct / total
    return v_loss, val_acc

def evaluate_unpair(loader, device, net, in_type):
    # type == 0: image input; type == 1: audio input; type == 2: both input
    correct, v_loss, total, logits = 0, 0, 0, 0
    net.eval()
    with torch.no_grad():
        for i, (data_img, data_aud) in enumerate(zip(*loader)):
            img_inputs_cln, labels_img = data_img['image'], data_img['label']
            aud_inputs_cln, labels_aud = data_aud['audio'], data_aud['label']
            img_inputs, aud_inputs, labels_img, labels_aud = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels_img.to(
                device), labels_aud.to(device)

            if in_type == 0:
                labels = labels_img
                outputs, _, _ = net(img_inputs)
            elif in_type == 1:
                outputs, _, _ = net(aud_inputs)
                labels = labels_aud
            elif in_type == 2:
                outputs, _, _ = net(img_inputs, aud_inputs)
            else:
                raise ValueError('the value of element in in_type_list should be 0,1,2\n')
            total += labels.size(0)

            logits = torch.Tensor([F.softmax(outputs, dim=-1)[i][labels[i]] for i in range(outputs.size(0))]).sum()
            _, predicted = torch.max(outputs.detach(), 1)
            correct += (predicted == labels).sum().item()
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)  # fix to CE loss
            v_loss += loss.item()
    logits = logits / len(loader)
    v_loss = v_loss / len(loader)
    val_acc = 100 * correct / total
    return v_loss, val_acc


def evaluate_allacc(loader, device, net, in_type):
    _, train_acc = evaluate(loader['train'], device, net, in_type)
    _, val_acc = evaluate(loader['val'], device, net, in_type)
    _, test_acc = evaluate(loader['test'], device, net, in_type)
    return train_acc, val_acc, test_acc


def gen_data(data_dir, batch_size, num_workers, args):

    train_dataset_img = AVEDataset_img(args, mode='train')
    train_dataset_aud = AVEDataset_aud(args, mode='train')
    val_dataset_img = AVEDataset_img(args, mode='val')
    val_dataset_aud = AVEDataset_aud(args, mode='val')
    test_dataset_img = AVEDataset_img(args, mode='test')
    test_dataset_aud = AVEDataset_aud(args, mode='test')
    train_dataloader_img = DataLoader(train_dataset_img, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)  # 计算机的内存充足的时候，可以设置pin_memory=True
    train_dataloader_aud = DataLoader(train_dataset_aud, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)  # 计算机的内存充足的时候，可以设置pin_memory=True
    test_dataloader_img = DataLoader(test_dataset_img, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader_aud = DataLoader(test_dataset_aud, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_dataloader_img = DataLoader(val_dataset_img, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_dataloader_aud = DataLoader(val_dataset_aud, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


    train_dataset = AVEDataset(args, mode='train')
    test_dataset = AVEDataset(args, mode='test')
    val_dataset = AVEDataset(args, mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)  # 计算机的内存充足的时候，可以设置pin_memory=True
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    loader = {'train': (train_dataloader_img, train_dataloader_aud),
              'val': (val_dataloader_img, val_dataloader_aud),
              'test': (test_dataloader_img, test_dataloader_aud)}

    return loader


def ntkl(logits_student, logits_teacher, target, mask=None, criterion4=None, temperature=1):

    gt_mask = _get_gt_mask(logits_student, target)
    logits_teacher = logits_teacher * (~gt_mask)
    pred_teacher_part2 = F.softmax(logits_teacher / temperature, dim=1)
    logits_student = logits_student * (~gt_mask)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature, dim=1)
    if mask.sum() == 0:
        temp = torch.tensor(0)
    else:
        temp = ((mask * (criterion4(log_pred_student_part2, pred_teacher_part2.detach()).sum(1)))).mean()
    return temp



def train_network_distill(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    wandb.login(key="365a2332ad390479c5a6bb01365f47f0f427f47f")
    wandb.init(entity= "cmkd" ,project="c2kd")
    
    # wandb.init(project='Difformer',
    #            group=group,
    #            name=f"Diffusion {args.run_name}: seed_{args.seed}_lr_{args.lr}_pretrained_seed_{args.pretrained_seed}_ksvd_layers_{args.ksvd_layers}_lambda_mean_{args.lambda_mean}_var_{args.lambda_var}_ce_{args.lambda_ce}_batchsize_{args.batch_size}_epochs_{args.nb_epochs}",
    #         #    name=f"Diffusion {args.run_name}: seed_{args.seed}_lr_{args.lr}_clip_{args.clip}_pretrained_seed_{args.pretrained_seed}_mlp_dropout_{args.mlp_dropout}_ksvd_layers_{args.ksvd_layers}_lambda_mean_{args.lambda_mean}_var_{args.lambda_var}_ce_{args.lambda_ce}_batchsize_{args.batch_size}_architecture_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_epochs_{args.nb_epochs}_adversarial_noise_{args.adversarial_noise}_adversarial_samples_{args.adversarial_samples}_rnn_hidden_{args.rnn_hidden}_rnn_num_layers_{args.rnn_num_layers}_rnn_dropout_{args.rnn_dropout}",
    #            config=vars(args))

    iter = 0
    net.train()
    tea.train()
    stu.train()
    tea_model.train()
    # tea_model.eval()
    # for name, param in tea_model.named_parameters():
    #     if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
    #     # if 'fc' not in name:
    #         param.requires_grad = False
    
    for epoch in range(epochs):

        train_loss = 0.0
        corr_mean = 0.0
        corr_num = 0.0
        loss1, loss2, loss_s2t, loss_t2s, loss_ang = 0, 0, 0, 0, 0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()
            XE_s = F.cross_entropy(outputs, labels, reduction='none')
            XE_t = F.cross_entropy(pseu_label, labels, reduction='none')

            kl_loss_t_t_im = criterion3(F.log_softmax(tea_logits, -1), F.softmax(pseu_label.detach(), dim=-1))
            kl_loss_t_im_t = criterion3(F.log_softmax(pseu_label, -1), F.softmax(tea_logits.detach(), dim=-1))


            kl_loss_s_s_im = criterion3(F.log_softmax(stu_logits, -1), F.softmax(outputs.detach(), dim=-1))
            kl_loss_s_im_s = criterion3(F.log_softmax(outputs, -1), F.softmax(stu_logits.detach(), dim=-1))

            kendall = np.empty(tea_logits.size(0))
            for i in range(tea_logits.size(0)):
                temp = \
                kendalltau(tea_logits[i, :].cpu().detach().numpy(), stu_logits[i, :].cpu().detach().numpy())[0]
                kendall[i] = temp
            kendall = torch.from_numpy(kendall).cuda()
            kendall_mean = kendall.mean()
            mask = torch.where(kendall > args.krc, 1, 0)
            mask_num = kendall.sum()
            del kendall


            if mask.sum()==0:
                kl_loss_st_s_t = 0
                kl_loss_st_t_s = 0
            else:
                kl_loss_st_s_t = ntkl(stu_logits, tea_logits.detach(), labels, mask, criterion4)
                kl_loss_st_t_s = ntkl(tea_logits, stu_logits.detach(), labels, mask, criterion4)

            kendall = ntkl(tea_logits, stu_logits, labels, mask, criterion4)
            mask_num = kendall.sum()

            tmp1 = XE_t.mean() + kl_loss_t_t_im + kl_loss_st_t_s + kl_loss_t_im_t
            tmp2 = XE_s.mean() + kl_loss_s_s_im + kl_loss_st_s_t + kl_loss_s_im_s
            loss = tmp1 + tmp2
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            loss1 += tmp1.item()
            loss2 += tmp2.item()
            corr_mean += kendall_mean.item()
            corr_num += mask_num.item()

        if epoch >= 80:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'loss1': loss1 / len(loader['train']), 'loss2': loss2 / len(loader['train']), 'Train Loss': train_loss / len(loader['train'])})
        wandb.log({'KRC': corr_num / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(f"corr {corr_mean / len(loader['train']):.3f}| num: {corr_num / len(loader['train']):.3f}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f} | temp1 Loss {loss1 / len(loader['train']):.3f} | temp2 Loss {loss2 / len(loader['train']):.3f} | PL_pseu_label Loss {loss_s2t / len(loader['train']):.3f} | PL_output Loss {loss_t2s / len(loader['train']):.3f}| cosine Loss {loss_ang / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results', exist_ok=True)
        model_path = os.path.join('results', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

def train_network_distill2(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)

                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            # print(FA_loss)
            # LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax(pseu_label, dim=-1))
            # LA_loss = criterion3(F.log_softmax(pseu_label, -1), F.softmax(outputs.detach(), dim=-1))
            LA_loss = criterion3(F.log_softmax(pseu_label, -1), F.softmax(outputs, dim=-1))

            loss = CE_loss.mean() + FA_loss + LA_loss
            # loss = CE_loss.mean() + FA_loss
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean()
            FA_loss_total += FA_loss
            LA_loss_total += LA_loss



        if epoch >= 1:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE_loss_total': CE_loss_total / len(loader['train']), \
                    'FA_loss_total': FA_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

#just la + CE
def train_network_distill21(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)

                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            # FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            # print(FA_loss)
            LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax(pseu_label, dim=-1))
            # LA_loss = criterion3(F.log_softmax(pseu_label, -1), F.softmax(outputs.detach(), dim=-1))

            # loss = CE_loss.mean() + FA_loss + LA_loss
            loss = CE_loss.mean() + LA_loss
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean()
            # FA_loss_total += FA_loss
            LA_loss_total += LA_loss



        if epoch >= 1:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE_loss_total': CE_loss_total / len(loader['train']), \
                    'FA_loss_total': FA_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

# just ce + fa
def train_network_distill22(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)

                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            # print(FA_loss)
            # LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax(pseu_label, dim=-1))
            # LA_loss = criterion3(F.log_softmax(pseu_label, -1), F.softmax(outputs.detach(), dim=-1))

            # loss = CE_loss.mean() + FA_loss + LA_loss
            loss = CE_loss.mean() + FA_loss
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean()
            FA_loss_total += FA_loss
            # LA_loss_total += LA_loss



        if epoch >= 1:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE_loss_total': CE_loss_total / len(loader['train']), \
                    'FA_loss_total': FA_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

# just ce
def train_network_distill23(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)

                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            # FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            # print(FA_loss)
            # LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax(pseu_label, dim=-1))
            # LA_loss = criterion3(F.log_softmax(pseu_label, -1), F.softmax(outputs.detach(), dim=-1))

            # loss = CE_loss.mean() + FA_loss + LA_loss
            loss = CE_loss.mean()
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean()
            # FA_loss_total += FA_loss
            # LA_loss_total += LA_loss



        if epoch >= 1:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE_loss_total': CE_loss_total / len(loader['train']), \
                    'FA_loss_total': FA_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

# unpaired data
def train_network_distill3(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)

                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            # print(FA_loss)

            # feature from teacher come to classification head of student
            # ví dụ: match mean/std theo student feature trong batch
            with torch.no_grad():
                m_s = stu_f.mean(dim=0, keepdim=True)
                s_s = stu_f.std(dim=0, keepdim=True).clamp_min(1e-6)
                m_t = pseu_label_128.mean(dim=0, keepdim=True)
                s_t = pseu_label_128.std(dim=0, keepdim=True).clamp_min(1e-6)

            pseu_label_128_norm = (pseu_label_128 - m_t) / s_t
            pseu_label_128_aligned = pseu_label_128_norm * s_s + m_s  # đưa về scale giống student

            # nguoc lai
            # tea_ch_label = net.fc(pseu_label_128_aligned)
            # LA_loss = criterion3(F.log_softmax(tea_ch_label, -1), F.softmax( pseu_label, dim=-1))

            stu_ch_label = tea_model.fc(stu_f)
            LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax( stu_ch_label, dim=-1))

            loss = CE_loss.mean() + FA_loss + LA_loss
            # print(loss.item(), CE_loss.mean(), FA_loss, LA_loss)
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean()
            FA_loss_total += FA_loss
            LA_loss_total += LA_loss



        if epoch >= 1:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE_loss_total': CE_loss_total / len(loader['train']), \
                    'FA_loss_total': FA_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


#real unpair
def train_network_distill31(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, (data_img, data_aud) in enumerate(zip(*loader['train'])):
            iter = iter + 1
            img_inputs_cln, labels_img = data_img['image'], data_img['label']
            aud_inputs_cln, labels_aud = data_aud['audio'], data_aud['label']
            img_inputs_cln, aud_inputs_cln, labels_img, labels_aud = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels_img.to(
                device), labels_aud.to(device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)
                labels = labels_img
                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                labels = labels_aud
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            # print(FA_loss)

            # feature from teacher come to classification head of student
            # ví dụ: match mean/std theo student feature trong batch
            with torch.no_grad():
                m_s = stu_f.mean(dim=0, keepdim=True)
                s_s = stu_f.std(dim=0, keepdim=True).clamp_min(1e-6)
                m_t = pseu_label_128.mean(dim=0, keepdim=True)
                s_t = pseu_label_128.std(dim=0, keepdim=True).clamp_min(1e-6)

            pseu_label_128_norm = (pseu_label_128 - m_t) / s_t
            pseu_label_128_aligned = pseu_label_128_norm * s_s + m_s  # đưa về scale giống student

            # nguoc lai
            # tea_ch_label = net.fc(pseu_label_128_aligned)
            # LA_loss = criterion3(F.log_softmax(tea_ch_label, -1), F.softmax( pseu_label, dim=-1))

            with torch.no_grad():
                m_s = stu_f.mean(dim=0, keepdim=True)
                s_s = stu_f.std(dim=0, keepdim=True).clamp_min(1e-6)
                m_t = pseu_label_128.mean(dim=0, keepdim=True)
                s_t = pseu_label_128.std(dim=0, keepdim=True).clamp_min(1e-6)

            # pseu_label_128_norm = (pseu_label_128 - m_t) / s_t
            # pseu_label_128_aligned = pseu_label_128_norm * s_s + m_s  # đưa về scale giống student
            stu_f_norm = (stu_f - m_s) / s_s
            stu_f_aligned = stu_f_norm * s_t + m_t

            stu_ch_label = tea_model.fc(stu_f_aligned)
            # LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax( stu_ch_label, dim=-1))

            # loss = CE_loss.mean() + FA_loss + LA_loss
            # loss = CE_loss.mean() + LA_loss
            loss = CE_loss.mean() + FA_loss
            # print(loss.item(), CE_loss.mean(), FA_loss, LA_loss)
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean()
            FA_loss_total += FA_loss
            # LA_loss_total += LA_loss



        if epoch >= 1:
            _, train_acc = evaluate_unpair(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate_unpair(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate_unpair(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate_unpair(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate_unpair(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate_unpair(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE_loss_total': CE_loss_total / len(loader['train']), \
                    'FA_loss_total': FA_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

# online 
def train_network_distill4(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    # wandb.login(key="365a2332ad390479c5a6bb01365f47f0f427f47f")
    # wandb.init(entity= "cmkd" ,project="c2kd-ours")
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE1_loss_total = 0.0
        CE2_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0
        kl_loss_px_t_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)

                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE1_loss = F.cross_entropy(outputs, labels, reduction='none')
            CE2_loss = F.cross_entropy(tea_logits, labels, reduction='none')
            FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            kl_loss_px_t = criterion3(F.log_softmax(tea_logits, -1), F.softmax(pseu_label, -1))
            # print(FA_loss)

            # feature from teacher come to classification head of student
            # ví dụ: match mean/std theo student feature trong batch
            
            # with torch.no_grad():
            #     m_s = stu_f.mean(dim=0, keepdim=True)
            #     s_s = stu_f.std(dim=0, keepdim=True).clamp_min(1e-6)
            #     m_t = pseu_label_128.mean(dim=0, keepdim=True)
            #     s_t = pseu_label_128.std(dim=0, keepdim=True).clamp_min(1e-6)

            # pseu_label_128_norm = (pseu_label_128 - m_t) / s_t
            # pseu_label_128_aligned = pseu_label_128_norm * s_s + m_s  # đưa về scale giống student

            # tea_ch_label = net.fc(pseu_label_128_aligned)

            # LA_loss = criterion3(F.log_softmax(tea_ch_label, -1), F.softmax( pseu_label, dim=-1))

            # LA_loss = criterion3(F.log_softmax(pseu_label, -1), F.softmax(tea_ch_label, dim=-1))
            LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax(pseu_label, dim=-1))


            loss = CE1_loss.mean() + CE2_loss.mean() + FA_loss + LA_loss + kl_loss_px_t
            # print(loss.item(), CE_loss.mean(), FA_loss, LA_loss)
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            CE1_loss_total += CE1_loss.mean()
            CE2_loss_total += CE2_loss.mean()
            FA_loss_total += FA_loss
            LA_loss_total += LA_loss
            kl_loss_px_t_total += kl_loss_px_t



        if epoch >= 1:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE1_loss_total': CE1_loss_total / len(loader['train']), 'CE2_loss_total': CE2_loss_total / len(loader['train']), \
                    'FA_loss_total': FA_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train']), "kl_loss_px_t_total": kl_loss_px_t_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)
        print(f"trainable param: teacher model: {count_trainable_params(tea_model)} | student model: {count_trainable_params(net)} | proxy teacher model: {count_trainable_params(tea)} | proxy student model: {count_trainable_params(stu)}" )

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


def train_network_distill5(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        MSE_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)

                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            # FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            MSE_loss = F.mse_loss(stu_f.reshape(-1), tea_f.reshape(-1))
            # print(FA_loss)
            # LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax(pseu_label, dim=-1))
            # LA_loss = criterion3(F.log_softmax(pseu_label, -1), F.softmax(outputs.detach(), dim=-1))

            # loss = CE_loss.mean() + FA_loss + LA_loss
            loss = CE_loss.mean() + MSE_loss
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean()
            MSE_loss_total += MSE_loss
            # LA_loss_total += LA_loss



        if epoch >= 1:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE_loss_total': CE_loss_total / len(loader['train']), \
                    'MSE_loss_total': MSE_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t


# up to 31.59 don't change
def train_network_distill_highest(stu_type, tea_model, epochs, loader, net, device, optimizer, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)

                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            # print(FA_loss)
            # LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax(pseu_label, dim=-1))
            LA_loss = criterion3(F.log_softmax(pseu_label, -1), F.softmax(outputs.detach(), dim=-1))

            # loss = CE_loss.mean() + FA_loss + LA_loss
            loss = CE_loss.mean() + FA_loss
            loss.backward()
            optimizer.step()
            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean()
            FA_loss_total += FA_loss
            # LA_loss_total += LA_loss



        if epoch >= 1:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE_loss_total': CE_loss_total / len(loader['train']), \
                    'FA_loss_total': FA_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

# up to 31.59 don't change but change sth
def train_network_distill_highest2(stu_type, tea_model, epochs, loader, net, device, optimizer, warmup_lr_scheduler, main_lr_scheduler, lr_scheduler, args, tea, stu):
    save_model = True
    val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t = 0, 0, 0, 0
    model_best = net
    criterion3 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion4 = torch.nn.KLDivLoss(reduction='none')
    iter = 0
    net.train()
    tea.train()
    stu.train()
    # tea_model.train()
    tea_model.eval()
    for name, param in tea_model.named_parameters():
        # if 'layer4' not in name and 'layer4' not in name and 'fc' not in name:
        # if 'fc' not in name:
            param.requires_grad = False


    for epoch in range(epochs):

        train_loss = 0.0
        CE_loss_total = 0.0
        FA_loss_total = 0.0
        LA_loss_total = 0.0

        for i, data in enumerate(loader['train']):
            iter = iter + 1
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            # output and net == student
            if stu_type == 0:
                outputs, outputs_128, stu_fit = net(img_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(aud_inputs_cln)

                # Proxy
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            elif stu_type == 1:
                outputs, outputs_128, stu_fit = net(aud_inputs_cln)
                pseu_label, pseu_label_128, tea_fit = tea_model(img_inputs_cln)
                tea_logits = tea(tea_fit)
                stu_logits = stu(stu_fit)
            else:
                raise ValueError("Undefined training type in distilled training")

            optimizer.zero_grad()

            stu_f = outputs_128
            tea_f = pseu_label_128

            # print(stu_f.shape)
            CE_loss = F.cross_entropy(outputs, labels, reduction='none')
            FA_loss = ot.wasserstein_1d(stu_f.reshape(-1), tea_f.reshape(-1))
            # print(FA_loss)
            # LA_loss = criterion3(F.log_softmax(outputs, -1), F.softmax(pseu_label, dim=-1))
            LA_loss = criterion3(F.log_softmax(pseu_label, -1), F.softmax(outputs.detach(), dim=-1))

            loss = CE_loss.mean() + FA_loss + LA_loss
            # loss = CE_loss.mean() + FA_loss
            loss.backward()
            optimizer.step()
            # lr = adjust_lr(iter=epoch, optimizer=optimizer)
            lr_scheduler.step()
            train_loss += loss.item()
            CE_loss_total += CE_loss.mean()
            FA_loss_total += FA_loss
            # LA_loss_total += LA_loss



        if epoch >= 1:
            _, train_acc = evaluate(loader['train'], device, net, stu_type)
            val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
            test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)
            _, train_acc_t = evaluate(loader['train'], device, tea_model, int(1-stu_type))
            val_loss_t, val_acc_t = evaluate(loader['val'], device, tea_model, int(1-stu_type))
            test_loss_t, test_acc_t = evaluate(loader['test'], device, tea_model, int(1-stu_type))

            wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,\
                       'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        else:
            _, train_acc = 0, 0
            val_loss, val_acc = 0, 0
            test_loss, test_acc = 0, 0
            _, train_acc_t = 0, 0
            val_loss_t, val_acc_t = 0, 0
            test_loss_t, test_acc_t = 0, 0

        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)
        if val_acc_t >= val_best_acc_t:
            val_best_acc_t = val_acc_t
            test_best_acc_t = test_acc_t

        wandb.log({'trainloss': train_loss / len(loader['train']), 'CE_loss_total': CE_loss_total / len(loader['train']), \
                    'FA_loss_total': FA_loss_total / len(loader['train']), 'LA_loss_total': LA_loss_total / len(loader['train'])})
        wandb.log({'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, \
                   'train_acc_t': train_acc_t, 'val_acc_t': val_acc_t, 'test_acc_t': test_acc_t})
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(
            f"Train Loss: {train_loss / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"teacher: Train | Val | Test Accuracy {train_acc_t:.3f} | {val_acc_t:.3f} | {test_acc_t:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}")
        print(f"Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}\n", '-' * 70)


    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')
    print(f'Training finish! Best Teacher Val | Test Accuracy | {val_best_acc_t:.3f} | {test_best_acc_t:.3f}')

    if save_model:
        os.makedirs('results/our', exist_ok=True)
        model_path = os.path.join('results', 'our', 'distillednet_mod_' + str(stu_type) + '_' + str(
            args.num_frame) + '_kdweight' + str(args.weight) + '_stu_acc_' + str(round(test_best_acc, 2)) + '_tea_acc_' \
                                  + str(round(test_best_acc_t, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(
            f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc, val_best_acc_t, test_best_acc_t

def pre_train_models(stu_type, tea_type, loader, epochs, learning_rate, device, args, save_model=False):
    criterion = torch.nn.CrossEntropyLoss()
    tea_model = ImageNet(args).to(device) if tea_type == 0 else AudioNet(args).to(device)
    stu_model = ImageNet(args).to(device) if stu_type == 0 else AudioNet(args).to(device)
    optimizer = torch.optim.SGD(list(tea_model.parameters()) + list(stu_model.parameters()), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.SGD(list(tea_model.parameters()), lr=learning_rate, momentum=0.9)
    val_best_acc, test_best_acc, val_best_acc_s, test_best_acc_s, tea_model_best, stu_model_best = 0, 0, 0, 0, 0, 0
    for epoch in range(epochs):
        tea_model.train()
        stu_model.train()
        # train_loss = 0.0
        loss1, loss2, loss3 = 0, 0, 0
        for i, data in enumerate(loader['train']):
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(device)
            if stu_type == 0:
                outputs1 = stu_model(img_inputs_cln)
                outputs2 = tea_model(aud_inputs_cln)
            elif stu_type == 1:
                outputs2 = tea_model(img_inputs_cln)
                outputs1 = stu_model(aud_inputs_cln)
            else:
                raise ValueError("Undefined training type")

            lr = adjust_lr(iter=epoch, optimizer=optimizer)
            optimizer.zero_grad()
            tmp1 = criterion(outputs1[0], labels)
            tmp2 = criterion(outputs2[0], labels)
            tmp3 = 0
            loss = tmp2
            loss.backward()
            optimizer.step()

            loss1 += tmp1.item()
            loss2 += tmp2.item()
            loss3 += tmp3
        print(f"Epoch {epoch} | loss1 {loss1 / len(loader['train']):.4f} | loss2 {loss2 / len(loader['train']):.4f} ")

        if epoch > 80:
            _, train_acc = evaluate(loader['train'], device, tea_model, tea_type)
            _, train_acc_s = evaluate(loader['train'], device, stu_model, stu_type)
            _, val_acc = evaluate(loader['val'], device, tea_model, tea_type)
            _, val_acc_s = evaluate(loader['val'], device, stu_model, stu_type)
            _, test_acc = evaluate(loader['test'], device, tea_model, tea_type)
            _, test_acc_s = evaluate(loader['test'], device, stu_model, stu_type)
            print(f'Teacher train|val|test acc {val_acc:.2f}|{val_acc:.2f}|{test_acc:.2f}')
            print(f'Student train|val|test acc {train_acc_s:.2f}|{val_acc_s:.2f}|{test_acc_s:.2f}')
            if val_acc > val_best_acc:
                val_best_acc = val_acc
                test_best_acc = test_acc
                tea_model_best = deepcopy(tea_model)
            if val_acc_s > val_best_acc_s:
                val_best_acc_s = val_acc_s
                test_best_acc_s = test_acc_s
                stu_model_best = deepcopy(stu_model)
            print(f"Best Teacher Val|Test acc: {val_best_acc:.2f}|{test_best_acc:.2f}")
            print(f"Best Student Val|Test acc: {val_best_acc_s:.2f}|{test_best_acc_s:.2f}")
            print('-' * 60)

    print('Finish training for single modality')

    if save_model:
        os.makedirs('./results', exist_ok=True)
        model_path_t = os.path.join('./results', 'teacher_mod_' + str(tea_type) + '_' + 'resnet18' + '_' + str(args.num_frame) + '_overlap.pkl')
        torch.save(tea_model_best.state_dict(), model_path_t)
        model_path_s = os.path.join('./results', 'student_mod_' + str(stu_type) + '_' + 'resnet18' + '_'+ str(args.num_frame) + '_overlap.pkl')
        torch.save(stu_model_best.state_dict(), model_path_s)
        print(f"Saving teacher and student model to {model_path_t} and {model_path_s}")
        print(f"Best Test acc: teacher: {test_best_acc:.2f} student: {test_best_acc_s:.2f}")


def pre_train(stu_type, loader, epochs, learning_rate, device, args):
    tea_type = 1 - stu_type
    pre_train_models(stu_type, tea_type, loader, epochs, learning_rate, device, args, save_model=True)



if __name__ == '__main__':
    pass