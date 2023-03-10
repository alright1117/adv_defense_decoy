import torch
import torch.nn as nn

class PGD:
    def __init__(self, eps, alpha, iters, clamp_min=0, clamp_max=255, init=True, nes_batch=10, nes_iters=10):
        # normalizaed alpha, eps, and clamp values
        self.eps = eps / 0.5
        self.alpha = alpha / 0.5
        self.clamp_min = (clamp_min / 255 - 0.5) / 0.5
        self.clamp_max = (clamp_max / 255 - 0.5) / 0.5

        self.iters = iters
        self.init = init
        self.nes_batch = nes_batch
        self.nes_iters = nes_iters

        self.loss = nn.CrossEntropyLoss()

    def attack(self, model, images, labels):

        ori_images = images.clone()
    
        if self.init:
            images += torch.FloatTensor(images.size()).uniform_(-self.eps, self.eps).to(images.device)

        for i in range(self.iters):  

            images.requires_grad = True
            outputs = model(images)

            model.zero_grad()
            cost = self.loss(outputs, labels)
            cost.backward()

            adv_images = images + self.alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=self.clamp_min, max=self.clamp_max).detach_()
                
        return images

    def nes_attack(self, model, images, labels):

        ori_images = images.clone()
        target_labels = (1 - labels)[0].detach().cpu().tolist()

        if self.init:
            images += torch.FloatTensor(images.size()).uniform_(-self.eps, self.eps).to(images.device)

        for _ in range(self.iters):

            perb = self._nes(model, images, target_labels)

            adv_images = images + self.alpha * perb.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = (ori_images + eta).detach_()
                
        return images
    
    def _nes(self, model, images, target_labels):
        b, c, h, w = images.shape
        perb = torch.zeros(b, c, h, w).to(images.device)
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for _ in range(self.nes_iters):
                u = (torch.randn(self.nes_batch, b, c, h, w) / 255).to(images.device)

                outputs = softmax(model((images + u).reshape(-1, c, h, w))).detach_()
                p = outputs[:, 1].reshape(self.nes_batch, b, 1, 1, 1)
                p = 1 - p if target_labels == 0 else p
                perb += torch.sum(p * u, axis=0)

                outputs = softmax(model((images - u).reshape(-1, c, h, w))).detach_()
                p = outputs[:, 1].reshape(self.nes_batch, b, 1, 1, 1)
                p = 1 - p if target_labels == 0 else p
                perb -= torch.sum(p * u, axis=0)

        return perb
