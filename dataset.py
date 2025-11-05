import os
from PIL import Image
from torch.utils.data import Dataset
from FindXY import findXY

class MiHouDataset(Dataset):
    def __init__(self, root, transform=None, augmentation_factor=1):
        super(MiHouDataset, self).__init__()
        self.root = root
        self.styleroot = root.split("images")[0]
        self.paths = os.listdir(self.root)
        self.stylepaths = os.listdir(self.styleroot)
        self.transform = transform
        self.auaugmentation_factor = augmentation_factor
        self.image_filenames = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'MiHouDataset'

    def __getitem__(self, idx):
        path = self.paths[idx]
        sid = idx % 104
        stylepath = self.stylepaths[sid]

        if self.auaugmentation_factor == 1:
            content_mask = findXY(path)
            pil_img = Image.fromarray(content_mask)
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = self.transform(img)
            mask_img = self.transform(pil_img).repeat(3, 1, 1)

            sty_img = Image.open(os.path.join(self.styleroot, stylepath))
            sty_img = self.transform(sty_img)
            return img, mask_img, sty_img
        else:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = self.transform(img)

            sty_img = Image.open(os.path.join(self.styleroot, stylepath))
            sty_img = self.transform(sty_img)
            return img, path, sty_img

