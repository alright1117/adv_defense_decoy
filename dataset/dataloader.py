import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .test_dataset import TestFaceForensics
from .train_dataset import TrainFaceForensics, TrainFaceForensicsDeception

def get_loader(args):

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((args.input_size, args.input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # if args.deception:
    #     train_data = TrainFaceForensicsDeception(args.root_dir, args.train_file_path, args.train_img_batch, args.input_size, transform = transform)
    # else:
    #     train_data = TrainFaceForensics("train", args.root_dir, args.train_img_batch, args.input_size, transform = transform)

    train_data = TrainFaceForensics("train", args.root_dir, args.train_img_batch, args.input_size, transform = transform)
    train_loader = DataLoader(dataset = train_data, batch_size = args.train_video_batch, shuffle=True, num_workers=args.workers, pin_memory=True)

    val_data = TestFaceForensics(args.root_dir, args.val_file_path, args.val_img_batch, args.input_size, transform = transform)
    val_loader = DataLoader(dataset = val_data, batch_size = 1, shuffle=False, num_workers=1, pin_memory=True)

    test_data = TestFaceForensics(args.test_root_dir, args.test_file_path, args.test_img_batch, args.input_size, transform = transform)
    test_loader = DataLoader(dataset = test_data, batch_size = 1, shuffle=False, num_workers=1, pin_memory=True)

    return train_data, val_data, test_data, train_loader, val_loader, test_loader