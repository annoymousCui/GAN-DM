import argparse
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import net

def test_transform():
    transform_list = []
    transform_list.append(transforms.Resize(size=(512, 512)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default=['./input/content'],
                    nargs='+', help='Directory paths to content images')
parser.add_argument('--style_dir', type=str, default='./input/style', help='Directory path to style images')
parser.add_argument('--steps', type=str, default=1)
parser.add_argument('--vgg', type=str, default='./model/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='./experiments/stage1/decoder.pth')
parser.add_argument('--transform', type=str, default='./experiments/stage1/transformer.pth')
# Additional options
parser.add_argument('--save_ext', default='.png',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='./stylized/stage1',
                    help='Directory to save the output image(s)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
transform = net.Transform(in_planes=512)
vgg = net.vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform), False)
vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

up = nn.Upsample(scale_factor=2, mode='nearest')

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()

# Pair content directories with style images
content_dirs = args.content_dir
style_files = [os.path.join(args.style_dir, f) for f in os.listdir(args.style_dir)
               if os.path.isfile(os.path.join(args.style_dir, f))]

for content_dir in content_dirs:
    # Get all content images in the directory
    content_files = [os.path.join(content_dir, f) for f in os.listdir(content_dir)
                     if os.path.isfile(os.path.join(content_dir, f))]

    for content_file in content_files:
        # Randomly select a style image
        style_image = random.choice(style_files)
        print(fr"\nProcessing: Content={content_file}, Style={style_image}")

        # Load images
        content = content_tf(Image.open(content_file))
        style = style_tf(Image.open(style_image))

        # Prepare tensors
        content = content.repeat(3, 1, 1)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        ori_content = content.squeeze(0)
        ori_style = style.squeeze(0)

        with torch.no_grad():
            for x in range(int(args.steps)):
                print('iteration ' + str(x))

                Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
                Content5_1 = enc_5(Content4_1)

                Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
                Style5_1 = enc_5(Style4_1)

                content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))
                content.clamp(0, 255)

        style_content = content.squeeze(0)
        content = content.cpu()

        # Generate output filename
        content_basename = splitext(basename(content_file))[0]
        style_basename = splitext(basename(style_image))[0]
        output_name = f"{args.output}\\{content_basename}_stylized_{style_basename}{args.save_ext}"
        save_image(style_content, output_name)

