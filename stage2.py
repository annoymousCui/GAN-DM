import torch
import torch.optim as optim
import argparse
import os
import dataset
from torch.utils.data import DataLoader
from UNet import unet
from Diffusion import diffusion
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transform_list)

def test_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transform_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Start to fine-tune the generated image')
    parser.add_argument('--data_dir', type=str, default="./stylized/stage1",
                         help='Path to the directory containing the dataset.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=700, help='Number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for the optimizer.')
    parser.add_argument('--save_model', type=str, default='./experiments/stage2', help='Path to save the trained model.')
    parser.add_argument('--start_iter', type=float, default=0)
    parser.add_argument('--sample_path', type=str, default='./samples/stage2', help='Derectory to save the intermediate samples')
    parser.add_argument('--output_path', type=str, default='./stylized/stage2', help='Derectory to save the output samples')
    parser.add_argument('--model_path', type=str, default='./experiments/stage2/diffusion_model.pth', help='Derectory to save the intermediate samples')
    parser.add_argument('--mode', type=str, default='train', help='choose train or test')
    return parser.parse_args()

# main function
def main():
    args = parse_arguments()
    print(args.batch_size)

    train_dataset = dataset.MiHouDataset(root=args.data_dir, transform=train_transform(), augmentation_factor=1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)


    test_dataset = dataset.MiHouDataset(root=args.data_dir, transform=test_transform(), augmentation_factor=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                   num_workers=0)

    image_size = 128
    channels = 3
    dim_mults = (1, 2, 4,)

    denoise_model = unet.Unet(
        dim=image_size,
        channels=channels,
        dim_mults=dim_mults
    )

    schedule_name = "linear_beta_schedule"
    timesteps = 1000

    model = diffusion.DiffusionModel(schedule_name=schedule_name,
                          timesteps=timesteps,
                          beta_start=0.0001,
                          beta_end=0.02,
                          denoise_model=denoise_model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "train":
        # start train
        for epoch in tqdm(range(args.num_epochs)):
            for contents, masks, styles in train_loader:
                if args.mode == "train":
                    loss, pred_noise, noise, x_noise, pred_out = model(contents, masks, styles, args.mode)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss.item():.4f}')

            # save weight
            if (epoch % 300 == 0):
                save_model = Path(args.save_model)
                torch.save(model.state_dict(), save_model / f'diffusion_model_{epoch}.pth')

            sample_dir = Path(args.sample_path)
            if (epoch % 100 == 0):
                sampleset = torch.cat([contents.to("cuda:0"), noise, x_noise, pred_noise, pred_out], 2)
                sample_name = sample_dir / 'output{:d}.png'.format(epoch + 1)
                save_image(sampleset, str(sample_name))

    else:
        model.load_state_dict(torch.load(args.model_path))
        for i, (images, images_name, _) in enumerate(test_loader):
            print("--------------------{:d}---------------".format(i))
            print(images_name)
            print("---------------------end---------------")

            masks = None
            pred_out = model(_, masks, images, args.mode)

            output_dir = Path(args.output_path)

            contents = []
            for k in range(int(args.batch_size)):
                content_name = os.path.join(args.data_dir, images_name[k].split(".")[0]+".png")
                image = Image.open(content_name).convert('RGB')
                tensor = test_transform()(image)

                contents.append(tensor)

            temp_out = F.interpolate(pred_out[-1], scale_factor=2, mode='bilinear', align_corners=True)

            out_list = []
            for j in range(temp_out.size(0)):
                out_list.append(temp_out[j, :, :, :])

            for j,aout_put in enumerate(out_list):
                aoutput_name = output_dir / images_name[j]
                save_image(aout_put, str(aoutput_name))


if __name__ == '__main__':
    main()