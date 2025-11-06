# Our Work GAN-DM
  
## Requirements  
We recommend the following configurations:  
- Python 3.9
- PyTorch 2.3.0
- CUDA 12.1

## Stage One ： Coarse Image Generation
- Generate the coarse pseudo-neuron images
- For training GAN-DM stage one, your directory tree should be look like this:
```
$ROOT/input
├── content
│   ├── xxx.png
│   ├── xxx.png
│   └── ...
├── style
│   ├── xxx.png
│   ├── xxx.png
│   └── ...
└── recard
    ├── xxx.csv
    ├── xxx.csv
    └── ...
```

## Training
- In the folder recard, each CSV file stores the pixel coordinates of the centroid of synthetic neuron cells within the corresponding synthetic content image.
- Download the pre-trained [VGG-19](https://drive.google.com/file/d/11uddn7sfe8DurHMXa0_tPZkZtYmumRNH/view?usp=sharing) model.
- Run the following command:
```
python stage1Train.py --content_dir /input/content --style_dir /input/style
```

## Testing
- Put your trained model to *./experiments/stage1/* folder.
- Use same content images to *./input/content/* folder.
- Use same style images to *./input/style/* folder.
- Run the following command:
```
python stage1Test.py --content /input/content --style /input/style
```


## Stage Two ： Image Fine-Tuning
- Generate the fine-tuned pseudo-neuron images
- For training GAN-DM stage two, your input folder should be the output folder for Stage One *./stylized/stage1/* folder.

## Training
- Run the following command:
```
python stage2.py --data_dir /stylized/stage1 --mode train
```

## Testing
- Put your trained model to *./experiments/stage2/* folder.
- Run the following command:
```
python stage2.py --data_dir /stylized/stage1 --mode test
```

