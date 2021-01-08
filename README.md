# mitochondria_segmentation
segment mitochondria in microscope images.

## Data

There is 141 images, each has a resolution of 1024\*768 pixels. I use 126 of them as training set, and leave the rest for testing. For each picture, I extract 64\*64 patch, then apply data augmentation to get more training samples. With stride of 64, I had 192 patches from each picture, then by rotating 3 times, and applying horizontal and vertical mirroring, I can get 6 times more, which is 1152. Hence the total amount of training set is 145152.

## Method

### Model

The model is almost U-Net, one big difference is, pooling layers and transposed convolution layers have both been reduced to 3 times, different from the original version, which has it 4 times. And the number of channels of all inner layer has been halved. please read the [source code](./src/model.py) for the details.

### Post Processing

I use opening and closing to reduce small noises. Opening is defined as erosion followed by dilation, while closing is the opposite, defined as dilation then erosion. I found that doing several times of closing then opening would make the result a bit better. 

## Training

Training is performed on Google Colaboratory. Batch size is fixed at 64, after about 500 epochs, it reaches probably the best result. [mito.ipynb](./src/mito.ipynb) shows the process.

## Result

I assume mitochondria as positive, and other area is negative. Below is probably the best result I did.

| Accuracy | Specificity | Dice   | IOU    |
| -------- | ----------- | ------ | ------ |
| 0.9885   | 0.9901      | 0.8984 | 0.8156 |

