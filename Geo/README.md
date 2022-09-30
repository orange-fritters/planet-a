# Planet A : Geological Disaster

The final submitted pytorch code for the **Planet A** Challenge

## Abstract
Geological disasters such as earthquakes, volcanoes, or landslides make buildings or roads in downtown areas lost, 
or buildings are newly built in the process of disaster recovery. 
The purpose of the ***Planet A: Geological Disaster Section*** is to develop a deep learning model that can detect 
changes in urban areas using satellite images
Specifically, The geological disaster problem is to detect changes in five cities with unknown answers through  15 types
of channel data (various satellite bandwith data) taken from 19 cities. [[A1]](##Reference) To solve the problem, We 
applied the multimodality of [[A3]](##Reference) to the Combination of Siamese network and NestedUNet structure 
presented in [[A2]](##Reference). Then, we used the channel-wise attention module used in the RCAN structure of 
[[A4]](##Reference) for each block.

## Project Structure
```
├──  data
│    └── dataset_training - train data folder
│       ├── A
│       ├── B
│       ├── ...
│       └── P
│    └── dataset_val - train data folder
│       ├── Q
│       ├── R
│       └── S 
│    └── dataset_test - test data for submission
│       ├── AA
│       ├── ...
│       └── EE
|
├──  data_load  
│    ├── Data_Loader.py
│    └── Data_Loader_Forward.py  
│
├──  model  
│    ├── mm_SNUNet_do.py  - modified SNUNet model
│    └── SNUNet.py  - original SNUNet for baseline
│
├──  result
│   ├── netname1    
│   └── netname2 ...
│
├── train     
│   ├── mm_snunet_train.ipynb   
│   └── train_mm_snunet.py
│
└── utils   - utility files  
    ├── f1_score.py
    ├── final_output.py
    ├── plot_result.py
    ├── save_reconstruction_forward.py       
    └── save_reconstruction_validation.py
```
## How to run
ArgParser will be updated soon!

### Train
modifty the `train_mm_snunet.py` file and run it
```
if __name__ == '__main__':
    NAME = 'TEST'
    train(NAME, gpu=False)
```
### Validation
modify the `save_reconstruction_validation.py` file and run it
```
if __name__ == '__main__':
    NET_NAME = 'TEST'
    MODEL_PATH = '../result/TEST.pt'
    recon(NET_NAME, MODEL_PATH)
```
### Test
modify the `save_reconstruction_forward.py` file and run it
```
if __name__ == '__main__':
    NET_NAME = 'TEST'
    MODEL_PATH = '../result/TEST.pt'
    recon(NET_NAME, MODEL_PATH)
```
## Model Structure
![model](../img/model.png)

## Details

### Data Processing
In the case of learning data, 16 out of 19 cities of different sizes were used for learning and 3 for validation. Since 
convolutional neural networks generally receive the same input size, the learning data were cut into patches of size 
96X96 and divided into 640 data in total. 

In the case of S2 (Sentinel-2 Satellite) data, there were a total of 13 channels. Among these channels, 
channels of B01, B07, B10, B11, and B05 were excluded and trained. These channels have low resolution, 
high similarity to RGB channels of B02, B03, and B04, and we concluded they were not very helpful in identifying 
meteorological phenomena such as clouds. Therefore, RGB channels of B02, B03, and B04 and Vegetation Red Edge channels 
of B06 and B8A, NIR channels of B08, Water Vapor channels of B09 and SWIR of B12 were used for learning. 
For the S1 channel, data file name was renamed for convenience. Both VV and VH channels of S1 (Sentinel-1)were used, 
and the S1 channel data names were all unified and changed to S1 (for convenience).

In addition, the two images before and after the change had many differences in brightness and luminance depending on 
the shooting time and season. In order to reduce the burden on the network, the histogram matching method (from sklearn)
was applied to the post-change image in consideration of the time/season histogram of the images used for learning. 
Through this, it was possible to reduce the difference in brightness and luminance of the two images.

Thereafter, in the case of the S2 image, the histogram distribution of pixels was checked and clipped with a value of 
[0, 2500], and in the case of the S1 image, the value was clipped with [-25, 0]. Since then, due to the characteristics 
of the data distribution, the skewness or kurtosis of the distribution excluding the RGB value was high, and all data 
were unified into MinMaxScale and distributed between 0 and 1. Due to the small number of data, Image Augmentation 
produced up to eight additional images per patch through random left and right inversion and 90 degrees, 180 degrees, 
and 270 degrees rotation. In the process of loading data, we tried to prevent overfitting according to order through 
the sequence shuffle.

Much of the preprocessing was similar to the baseline [[A2]](##Reference)

### Train Phase
For learning, we used the Pytorch framework. A batch size of 16 was used for learning, and ADAM was used as an 
optimizer. 1e-3 was used as the learning rate, and the Learning Rate Decay Scheduler, which decreases by 0.95 per epoch, 
was introduced. In order to prevent overfitting, Adam Optimizer was given a weight decay of 1e-3. 
We used KaiMing initialization method as a network weight initialization. Cross entropy loss was used as the loss 
of learning. The distribution of data with/without variation was not uniform and the number of pixels without variation
was much higher. Cross Entropy Loss, which can manually set weights according to target label's distribution, was 
a great help for learning, and the weights of Cross Entropy Loss were set by calculating the actual distribution. 
In the test process, 96*96 patches were collected to construct the original image, and F1 Score was 
obtained to confirm the test result. The Google Co-Lab environment was used for learning, and the number of parameters 
of the model was approximately 39277970. In addition, in order to derive the final result, the shape of the evaluation 
data was zero padded in a multiple of 96 and split into patches, and then input into the model to print out the final map.

### Result Example
![model]<img src=../img/output.png width="650" height="325" />

## TODO List
- Arg Parser be updated soon!

## Reference
```
[A1] Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2018, July. Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks. In IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2018 (pp. 2115-2118). IEEE.}
[A2] Fang, Sheng, et al. "SNUNet-CD: A densely connected Siamese network for change detection of VHR images." IEEE Geoscience and Remote Sensing Letters 19 (2021): 1-5.
[A3] Ebel, Patrick und Saha, Sudipan und Zhu, Xiao Xiang (2021) Fusing Multi-modal Data for Supervised Change Detection. ISPRS. XXIV ISPRS Congress 2021, 04 - 10 July 2021, Nice, France
[A4] Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." Proceedings of the European conference on computer vision (ECCV). 2018.
[A5] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
```
