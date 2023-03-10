import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import logging

from dataset import get_loader
from network import load_model
from trainer import train, trainDeception
from evaluation import evaluation


def main(args):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(args.log_path, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    model, input_size = load_model(args.model_name, None, logger)
    args.input_size = input_size
    model = nn.DataParallel(model)
    device = torch.device('cuda')
    model.to(device)

    train_data, val_data, test_data, train_loader, val_loader, test_loader = get_loader(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # if args.deception:
    #     model, best_epoch, best_f1 = trainDeception( model, 
    #                                                 train_loader, 
    #                                                 val_loader, 
    #                                                 optimizer,
    #                                                 scheduler, 
    #                                                 device,
    #                                                 logger, 
    #                                                 args )
    # else:
    #     model, best_epoch, best_f1 = train( model, 
    #                                         train_loader, 
    #                                         val_loader, 
    #                                         optimizer, 
    #                                         scheduler, 
    #                                         device,
    #                                         logger,
    #                                         args )


    # logger.info("Best Epoch: %s, Best Val F1 %.4f." % (best_epoch, best_f1))

    if args.deception:
        test_f1_SP, test_acc_SP, test_auc_SP = evaluation(model, test_loader, "SP", args.input_size, device)
        test_f1_WSRT, test_acc_WSRT, test_auc_WSRT, _ = evaluation(model, test_loader, "WSRT", args.input_size, device)

        logger.info("Testing: F1_WSRT %.4f, Acc_WSRT %.4f, AUC_WSRT %.4f, F1_SP %.4f, Acc_SP %.4f, AUC_SP %.4f." % (test_f1_WSRT, test_acc_WSRT, test_auc_WSRT, test_f1_SP, test_acc_SP, test_auc_SP))
    else:
        test_f1, test_acc, test_auc = evaluation(model, test_loader, "SP", args.input_size, device)
        logger.info("Testing: F1 Score %.4f, Accuracy %.4f, AUC %.4f." % (test_f1, test_acc, test_auc))
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Training Setting
    # Dataset
    parser.add_argument("--root_dir", type = str, default = "data/FF++/raw")
    parser.add_argument("--test_root_dir", type = str, default = "data/FF++/raw")
    parser.add_argument("--train_file_path", type = str, default = "./file/FF++_train.txt")
    parser.add_argument("--val_file_path", type = str, default = "./file/FF++_val.txt")
    parser.add_argument("--train_video_batch", type = int, default = 10)
    parser.add_argument("--train_img_batch", type = int, default = 8)
    parser.add_argument("--val_img_batch", type = int, default = 10)
    parser.add_argument("--workers", type = int, default = 8)

    # Optimizer and scheduler
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--step_size", type = int, default = 30)
    parser.add_argument("--gamma", type = float, default = 0.1)
    parser.add_argument("--epoch", type = int, default = 100)

    # Model 
    parser.set_defaults(deception=False)
    parser.add_argument('--deception', dest='deception', action="store_true")
    parser.add_argument("--model_name", type = str, default = "Xception")

    # ADL and Margin loss
    parser.add_argument("--eta", type = float, default = 1.5)
    parser.add_argument("--margin", type = float, default = 0.8)
    parser.add_argument("--lamb", type = float, default = 1)

    # Testing setting
    parser.add_argument("--test_file_path", type = str, default = "./file/FF++_test.txt")
    parser.add_argument("--test_img_batch", type = int, default = 10)

    # Save path
    parser.add_argument("--log_path", type = str, default = "./log/Xception.log")
    parser.add_argument("--save_path", type = str, default = "./weight/Xception.pt")

    args = parser.parse_args()

    main(args)