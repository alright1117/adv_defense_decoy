import argparse
import logging
import torch
import torch.nn as nn

from dataset import get_loader
from network import load_model
from pgd import PGD
from evaluation import evaluation, evaluation_PGD


def main(args):

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler(args.log_path, mode='a')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("Testing file: %s." % (args.test_file_path))
    logger.info("PGD attack with eps %.3f, alpha %.4f, and iteration %d." % (args.eps, args.alpha, args.iters))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, input_size = load_model(args.model_name, args.pretrained_weight, logger)
    model.to(device)
    args.input_size = input_size

    train_data, val_data, test_data, train_loader, val_loader, test_loader = get_loader(args)

    attacker = PGD(args.eps, args.alpha, args.iters, nes_batch = args.nes_batch, nes_iters = args.nes_iters)

    if args.test_clean:

        logger.info("Before Attack")

        if args.deception:
            test_f1_SP, test_acc_SP, test_auc_SP = evaluation(model, test_loader, "SP", args.input_size, device)
            test_f1_WSRT, test_acc_WSRT, test_auc_WSRT, _ = evaluation(model, test_loader, "WSRT", args.input_size, device)

            logger.info("Testing: F1_WSRT %.4f, Acc_WSRT %.4f, AUC_WSRT %.4f, F1_SP %.4f, Acc_SP %.4f, AUC_SP %.4f." % (test_f1_WSRT, test_acc_WSRT, test_auc_WSRT, test_f1_SP, test_acc_SP, test_auc_SP))
        else:
            test_f1, test_acc, test_auc = evaluation(model, test_loader, "SP", args.input_size, device)
            logger.info("Testing: F1 Score %.4f, Accuracy %.4f, AUC %.4f." % (test_f1, test_acc, test_auc))
    
    # Attack

    logger.info("After Attack")

    if args.deception:

        test_f1_WSRT_PGD, test_acc_WSRT_PGD, test_auc_PGD = evaluation_PGD(model = model, 
                                                                           loader = test_loader, 
                                                                           mode = "WSRT", 
                                                                           size = args.input_size,
                                                                           device = device,
                                                                           attacker = attacker,
                                                                           black=args.black)

        logger.info("Testing: F1_WSRT %.4f, Acc_WSRT %.4f, AUC_WSRT %.4f." % (test_f1_WSRT_PGD, test_acc_WSRT_PGD, test_auc_PGD))

    else:
        
        test_f1_PGD, test_acc_PGD, test_auc_PGD = evaluation_PGD(model = model, 
                                                                 loader = test_loader, 
                                                                 mode = "SP", 
                                                                 size = args.input_size, 
                                                                 device = device, 
                                                                 attacker = attacker,
                                                                 black=args.black)

        logger.info("Testing: F1 Score %.4f, Accuracy %.4f, AUC %.4f." % (test_f1_PGD, test_acc_PGD, test_auc_PGD))                               


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--root_dir", type = str, default = "data/FF++/raw/")
    parser.add_argument("--train_file_path", type = str, default = "./file/FF++_train.txt")
    parser.add_argument("--val_file_path", type = str, default = "./file/FF++_val.txt")
    parser.add_argument("--train_video_batch", type = int, default = 10)
    parser.add_argument("--train_img_batch", type = int, default = 8)
    parser.add_argument("--val_img_batch", type = int, default = 10)
    parser.add_argument("--workers", type = int, default = 8)

    # Model 
    parser.set_defaults(deception=False)
    parser.add_argument('--deception', dest='deception', action="store_true")
    parser.add_argument("--model_name", type = str, default = "Xception")
    parser.add_argument("--pretrained_weight", type = str, default = "./weight/Xception.pt")

    # Testing setting
    parser.add_argument("--test_root_dir", type = str, default = "/ssd1/FF++/raw/")
    parser.add_argument("--test_file_path", type = str, default = "./file/FF++_test.txt")
    parser.add_argument("--test_img_batch", type = int, default = 10)
    parser.set_defaults(test_clean=False)
    parser.add_argument('--test_clean', dest='test_clean', action="store_true")
    
    # PGD
    parser.add_argument("--eps", type = float, default = 8/255)
    parser.add_argument("--alpha", type = float, default = 1/255)
    parser.add_argument("--iters", type = int, default = 10)

    # NES+PGD
    parser.set_defaults(black=False)
    parser.add_argument('--black', dest='black', action="store_true")
    parser.add_argument("--nes_iters", type = int, default = 11)
    parser.add_argument("--nes_batch", type = int, default = 9)

    # Save path
    parser.add_argument("--log_path", type = str, default = "./log/Xception_attack.log")

    args = parser.parse_args()

    main(args)