import argparse
import logging
import torch

from dataset import get_loader
from network import load_model
from evaluation import evaluation_detection
from pgd import PGD

def main(args):

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler(args.log_path, mode='a')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("Testing file: %s." % (args.test_file_path))
    logger.info("PGD attack detection with eps %.3f, alpha %.3f, and iteration %d." % (args.eps, args.alpha, args.iters))
    logger.info("Eta in hypothesis testing is %.3f." % (args.eta))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, input_size = load_model(args.model_name, args.pretrained_weight, logger)
    model.to(device)
    args.input_size = input_size

    train_data, val_data, test_data, train_loader, val_loader, test_loader = get_loader(args)

    attacker = PGD(args.eps, args.alpha, args.iters, nes_batch = args.nes_batch, nes_iters = args.nes_iters)

    detect_accuracy, auc = evaluation_detection(model = model, 
                                           loader = test_loader, 
                                           size = args.input_size, 
                                           device = device, 
                                           attacker = attacker,
                                           eta = args.eta,
                                           black = args.black)

    logger.info("Accuracy %.4f, AUC %.4f" % (detect_accuracy, auc))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--root_dir", type = str, default = "data/FF++/raw")
    parser.add_argument("--train_file_path", type = str, default = "./file/FF++_train.txt")
    parser.add_argument("--val_file_path", type = str, default = "./file/FF++_val.txt")
    parser.add_argument("--train_video_batch", type = int, default = 10)
    parser.add_argument("--train_img_batch", type = int, default = 8)
    parser.add_argument("--val_img_batch", type = int, default = 10)
    parser.add_argument("--workers", type = int, default = 1)

    # Model 
    parser.set_defaults(deception=True)
    parser.add_argument("--model_name", type = str, default = "Xception")
    parser.add_argument("--pretrained_weight", type = str, default = "./weight/Xdeception.pt")

    # Testing setting
    parser.add_argument("--test_root_dir", type = str, default = "data/FF++/raw")
    parser.add_argument("--test_file_path", type = str, default = "./file/FF++_test.txt")
    parser.add_argument("--test_img_batch", type = int, default = 10)
    
    # PGD
    parser.add_argument("--eps", type = float, default = 8/255)
    parser.add_argument("--alpha", type = float, default = 1/255)
    parser.add_argument("--iters", type = int, default = 10)

    # NES+PGD
    parser.set_defaults(black=False)
    parser.add_argument('--black', dest='black', action="store_true")
    parser.add_argument("--nes_iters", type = int, default = 11)
    parser.add_argument("--nes_batch", type = int, default = 9)

    # Wilcoxon signed-rank test
    parser.add_argument("--eta", type = float, default = 0.7)

    # Save path
    parser.add_argument("--log_path", type = str, default = "./log/Xdeception_detection.log")

    args = parser.parse_args()

    main(args)