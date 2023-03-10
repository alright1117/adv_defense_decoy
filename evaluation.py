import numpy as np
import random
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wilcoxon
from tqdm import tqdm
import torch
import torch.nn as nn

def evaluation(model, loader, mode, size, device, return_label=False):
    """
        mode: SP (Softmax probability),
              WSRT (Wilcoxon signed-rank test).

    """

    assert (mode is "SP") | (mode is "WSRT"), "Evaluaiton mode error."

    pred_labels = []
    target_labels = []
    labels_name = []
    prob_list = []

    model.eval()
    if mode == "SP":

        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for data, targets, label_name in tqdm(loader):
                
                data = data.view(-1, 3, size, size).to(device)
                target = targets.view(-1)[0].cpu().tolist()

                outputs = softmax(model(data)).detach().cpu()
                prob = torch.mean(outputs[:, 1]).item()
                preds = torch.argmax(outputs, dim=1).tolist()

                #pred = 1 if (sum(preds) >= (len(preds) // 2)) else 0
                pred = 1 if prob > 0.5 else 0

                pred_labels.append(pred)
                target_labels.append(target)
                labels_name.append(label_name[0])
                prob_list.append(prob)

    else:
        dis={"real_dis0":[], "fake_dis0":[], "real_dis1":[], "fake_dis1":[]}
        with torch.no_grad():
            for data, targets, label_name in tqdm(loader):

                data = data.view(-1, 3, size, size).to(device)
                target = targets.view(-1)[0].cpu().tolist()

                samples = model(data)[:, 0].detach().cpu().tolist()
                samples_dim1 = model(data)[:, 1].detach().cpu().tolist()

                if target == 1:
                    dis["fake_dis0"].extend(samples)
                    dis["fake_dis1"].extend(samples_dim1)
                else:
                    dis["real_dis0"].extend(samples)
                    dis["real_dis1"].extend(samples_dim1)

                w, p = wilcoxon(samples, alternative='greater')

                pred = 1 if p < 0.05 else 0
            
                pred_labels.append(pred)
                target_labels.append(target)
                labels_name.append(label_name[0])
                prob_list.append(1 - p)

    f1 = f1_score(target_labels, pred_labels, average='macro')
    accuracy = np.sum(np.array(target_labels) == np.array(pred_labels)) / len(target_labels)
    auc = roc_auc_score(target_labels, prob_list)

    if return_label:
        acc_dict = get_label_accuracy(pred_labels, target_labels, labels_name)
        return f1, accuracy, auc, acc_dict
    else:
        if mode == "SP":
            return f1, accuracy, auc
        else:
            return f1, accuracy, auc, dis


def evaluation_PGD(model, loader, mode, size, device, attacker, nes_batch=10, nes_iters=10, black=False):
    """
        mode: SP (Softmax probability),
              WSRT (Wilcoxon signed-rank test).

    """
    assert (mode is "SP") | (mode is "WSRT"), "Evaluaiton mode error."

    pred_labels = []
    target_labels = []
    prob_list = []

    model.eval()
    if mode == "SP":

        softmax = nn.Softmax(dim=1)

        for data, targets, label_name in tqdm(loader):
        
            data = data.view(-1, 3, size, size).to(device)
            targets = targets.view(-1).to(device)
            target = targets.view(-1)[0].cpu().tolist()

            if black:
                images = attacker.nes_attack(model, data, targets)
            else:
                images = attacker.attack(model, data, targets)

            outputs = softmax(model(images)).detach().cpu()

            prob = torch.mean(outputs[:, 1]).item()
            preds = torch.argmax(outputs, dim=1).tolist()

           # pred = 1 if (sum(preds) >= (len(preds) // 2)) else 0
            pred = 1 if prob > 0.5 else 0

            pred_labels.append(pred)
            target_labels.append(target)
            prob_list.append(prob)
    
    else:
        for data, targets, label_name in tqdm(loader):

            data = data.view(-1, 3, size, size).to(device)
            targets = targets.view(-1).to(device)

            if black:
                images = attacker.nes_attack(model, data, targets)
            else:
                images = attacker.attack(model, data, targets)

            target = targets.view(-1)[0].cpu().tolist()

            samples = model(images)[:, 0].cpu().tolist()
            w, p = wilcoxon(samples, alternative='greater')

            pred = 1 if p < 0.05 else 0
        
            pred_labels.append(pred)
            target_labels.append(target)
            prob_list.append(1 - p)

    
    f1 = f1_score(target_labels, pred_labels, average='macro')
    accuracy = np.sum(np.array(target_labels) == np.array(pred_labels)) / len(target_labels)
    auc = roc_auc_score(target_labels, prob_list)

    return f1, accuracy, auc


def evaluation_detection(model, loader, size, device, attacker, eta, black=False):

    random.seed(0)

    attack_idx = [True] * (len(loader) // 2) + [False] * ((len(loader) - len(loader) // 2))
    random.shuffle(attack_idx)

    pred_labels = []
    target_labels = []
    prob_list = []

    for i, (data, targets, label_name) in tqdm(enumerate(loader)):

        if attack_idx[i]:
            data = data.view(-1, 3, size, size).to(device)
            targets = targets.view(-1).to(device)
            data2 = data.clone()
            if black:
                attacker.eps = random.uniform(0.016, 0.032) / 0.5
                images = attacker.nes_attack(model, data, targets)
            else:
                attacker.eps = random.uniform(0.008, 0.024) / 0.5
                images = attacker.attack(model, data, targets)

            target_labels.append(1)
        else:
            images = data.view(-1, 3, size, size).to(device).to(device)
            target_labels.append(0)

        samples_dim0 = model(images)[:, 0].cpu().tolist()
        samples_dim1 = model(images)[:, 1].cpu().tolist()
        
        diff = np.abs(np.array(samples_dim0)- np.array(samples_dim1))
        w, p = wilcoxon(diff - eta, alternative='greater')

        if p < 0.05:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

        prob_list.append(1 - p)

    accuracy = np.sum(np.array(target_labels) == np.array(pred_labels)) / len(target_labels)
    auc = roc_auc_score(target_labels, prob_list)

    return accuracy, auc


def get_label_accuracy(pred_labels, target_labels, labels_name):

    acc_dict = {name:[] for name in set(labels_name)}
    index_dict = acc_dict.copy()

    for i in range(len(labels_name)):
        index_dict[labels_name[i]].append(i)
    
    for name in index_dict.keys():
        index = index_dict[name]
        acc = np.sum(np.array(target_labels)[index] == np.array(pred_labels)[index]) / len(index)
        acc_dict[name] = acc
    
    return acc_dict