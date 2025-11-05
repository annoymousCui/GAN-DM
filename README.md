# Our Work PSD
  
## Requirements  
We recommend the following configurations:  
- python 3.9
- PyTorch 2.3.0
- CUDA 12.1

## Stage One
- Generate the coarse stylized images
- For training PSD stage one, your directory tree should be look like this:
$ROOT/input
├── content
│   ├── xxx.png
│   ├── xxx.png
│   ├── ...
├── style
│   ├── xxx.png
│   ├── xxx.png
│   ├── ...
├── recard
│   ├── xxx.csv
│   ├── xxx.csv
│   ├── ...

## Stage One Training
- In The Record, Each CSV File Stores The Pixel Coordinates Of The Cytoplasmic Center Of The Synthetic Neuron Cells Within The Corresponding PNG Image's Content.
- Download the pre-trained [VGG-19](https://drive.google.com/file/d/11uddn7sfe8DurHMXa0_tPZkZtYmumRNH/view?usp=sharing) model.
- Run the following command:
```
python stage1.py --content_dir /input/content --style_dir /input/style
```

## Stage One Testing
- Put your trained model to *./experiments/stage1/* folder.
- Use same content images to *./input/content/* folder.
- Use same style images to *./input/style/* folder.
- Run the following command:
```
python Eval.py --content /input/content --style /input/style
```
We provide the stage-one pre-trained model in *./experiments/stage1/* folder. 


## Stage Two
- Generate the refine stylized images
- For training PSD stage two, your input folder should be the output folder for Stage One *./stylized/stage1/* folder.

## Stage Two Train
- Run the following command:
```
python stage2.py --data_dir /stylized/stage1 --mode train
```

## Stage Two Testing
- Put your trained model to *./experiments/stage2/* folder.
- Run the following command:
```
python stage2.py --data_dir /stylized/stage1 --mode test
```
We provide the stage-two pre-trained model in *./experiments/stage2/* folder. 

