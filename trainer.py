from tqdm import tqdm
import random
import torch
import torch.nn as nn
from evaluation import evaluation
import json

def train(model, train_loader, val_loader, optimizer, scheduler, device, logger, args):

    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(range(1, args.epoch + 1))

    best_epoch = 0
    best_acc = 0.0

    for i in pbar:
    
        loss_list = []

        model.train()

        for data, target in train_loader:

            optimizer.zero_grad()

            data = data.view(-1, 3, args.input_size, args.input_size).to(device)
            target = target.view(-1).to(device)

            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            torch.cuda.empty_cache()

        Loss = sum(loss_list) / len(loss_list)
        scheduler.step()
        
        val_f1, val_acc, val_auc = evaluation(model, val_loader, "SP", args.input_size, device)

        if val_acc >= best_acc:
            best_epoch = i
            best_acc = val_acc
            torch.save(model.module.state_dict(), args.save_path)  

        info = "Epoch: %s, Training loss %.4f, val_f1 %.4f, val_acc %.4f, val_auc %.4f, best_acc %.4f" % (i, Loss, val_f1, val_acc, val_auc, best_acc)
        pbar.set_description(info)
        logger.info(info)

    return model, best_epoch, best_acc


def trainDeception(model, train_loader, val_loader, optimizer, scheduler, device, logger, args):
    
    # setting distribution of real and fake scores 
    real_dis = torch.distributions.Normal(-1, 0.2)
    fake_dis = torch.distributions.Normal(1, 0.2)
    criterion =  nn.MSELoss()
    pbar = tqdm(range(1, args.epoch + 1))

    best_epoch = 0
    best_avg_acc = 0.0

    dicts = {"adl":[], "margin":[], "auc_sp":[], "auc_wsrt":[], "dis":{}}

    model.train()
    loss_dict = {"adl":[], "margin":[]}

    for data, target in train_loader:

        optimizer.zero_grad()

        data = data.view(-1, 3, args.input_size, args.input_size).to(device)
        target = target.view(-1).to(device)

        samples = real_dis.sample(target.shape).to(device)
        fake_idx = torch.nonzero(target)
        samples[fake_idx] = fake_dis.sample(fake_idx.shape).to(device)

        outputs = model(data).detach()

        margin_loss = 0.0
        count = 0
        for param1, param2 in zip(model.module.model1.parameters(), model.module.model2.parameters()):
            if len(param1.shape) == 4:
                margin_loss += criterion(param1.flatten(), param2.flatten()).detach()
                count += 1
            else:
                continue

        margin_loss = args.margin - margin_loss / count
        adl_loss = criterion(outputs[:, 0], samples) + criterion(outputs[:, 1], samples * args.eta)

        if margin_loss.item() > 0:
            total_loss = adl_loss + margin_loss * args.lamb
            loss_dict["margin"].append(margin_loss.detach().item())
        else:
            total_loss = adl_loss
            loss_dict["margin"].append(0)

        loss_dict["adl"].append(adl_loss.detach().item())

    val_f1_SP, val_acc_SP, val_auc_SP = evaluation(model, val_loader, "SP", args.input_size, device)
    val_f1_WSRT, val_acc_WSRT, val_auc_WSRT, dis = evaluation(model, val_loader, "WSRT", args.input_size, device)

    info = "Epoch: %s, ADL Loss: %.4f, Margin Loss: %.4f, val_F1_SP: %.4f, val_acc_SP: %.4f, val_F1_WSRT: %.4f, val_acc_WSRT: %.4f, Best Avg acc %.4f" % \
            (0, sum(loss_dict["adl"])/len(loss_dict["adl"]), sum(loss_dict["margin"])/len(loss_dict["margin"]), val_f1_SP, val_acc_SP, val_f1_WSRT, val_acc_WSRT, best_avg_acc)

    pbar.set_description(info)
    logger.info(info)

    dicts["adl"].append(sum(loss_dict["adl"])/len(loss_dict["adl"]))
    dicts["margin"].append(sum(loss_dict["margin"])/len(loss_dict["margin"]))
    dicts["auc_sp"].append(val_auc_SP)
    dicts["auc_wsrt"].append(val_auc_WSRT)
    dicts["dis"][0] = dis

    for i in pbar:
    
        model.train()
        loss_dict = {"adl":[], "margin":[]}

        for data, target in train_loader:

            optimizer.zero_grad()

            data = data.view(-1, 3, args.input_size, args.input_size).to(device)
            target = target.view(-1).to(device)

            # samples from setting distribution of real scores  
            samples = real_dis.sample(target.shape).to(device)
            fake_idx = torch.nonzero(target)
            samples[fake_idx] = fake_dis.sample(fake_idx.shape).to(device)

            outputs = model(data)

            margin_loss = 0.0
            count = 0
            for param1, param2 in zip(model.module.model1.parameters(), model.module.model2.parameters()):
                if len(param1.shape) == 4:
                    margin_loss += criterion(param1.flatten(), param2.flatten())
                    count += 1
                else:
                    continue

            margin_loss = args.margin - margin_loss / count
            adl_loss = criterion(outputs[:, 0], samples) + criterion(outputs[:, 1], samples * args.eta)

            if margin_loss.item() > 0:
                total_loss = adl_loss + margin_loss * args.lamb
                loss_dict["margin"].append(margin_loss.detach().item())
            else:
                total_loss = adl_loss
                loss_dict["margin"].append(0)

            total_loss.backward()
            optimizer.step()

            loss_dict["adl"].append(adl_loss.detach().item())

        scheduler.step()

        val_f1_SP, val_acc_SP, val_auc_SP = evaluation(model, val_loader, "SP", args.input_size, device)
        val_f1_WSRT, val_acc_WSRT, val_auc_WSRT, dis = evaluation(model, val_loader, "WSRT", args.input_size, device)

        # Saving weight if the result is better
        if ((val_acc_SP + val_acc_WSRT) / 2 >= best_avg_acc):
            best_epoch = i
            best_avg_acc = (val_acc_SP + val_acc_WSRT) / 2
            torch.save(model.module.state_dict(), args.save_path)
        
        info = "Epoch: %s, ADL Loss: %.4f, Margin Loss: %.4f, val_F1_SP: %.4f, val_acc_SP: %.4f, val_F1_WSRT: %.4f, val_acc_WSRT: %.4f, Best Avg acc %.4f" % \
                (i, sum(loss_dict["adl"])/len(loss_dict["adl"]), sum(loss_dict["margin"])/len(loss_dict["margin"]), val_f1_SP, val_acc_SP, val_f1_WSRT, val_acc_WSRT, best_avg_acc)

        pbar.set_description(info)
        logger.info(info)

        dicts["adl"].append(sum(loss_dict["adl"])/len(loss_dict["adl"]))
        dicts["margin"].append(sum(loss_dict["margin"])/len(loss_dict["margin"]))
        dicts["auc_sp"].append(val_auc_SP)
        dicts["auc_wsrt"].append(val_auc_WSRT)
        dicts["dis"][i] = dis

    # file = open("data_MD.json", "w")
    # json.dump(dicts, file)
    # file.close()

    return model, best_epoch, best_avg_acc
