# Planet A Challenge

### 🏆🏆🏆 **1st Place** 🏆🏆🏆

The final PyTorch code submssion for the **Planet A** Challenge.
**Additional README** available inside the Geo and Climate folder

## Field
- [Geological Disaster](##Geo/README.md)
- [Climate Science](##Climate/README.md)

## In a Nutshell
### Change Detection due to Geological Disaster
Geological disasters such as earthquakes, volcanoes, or landslides make buildings or roads in downtown areas lost, 
or buildings are newly built in the process of disaster recovery. 
The purpose of the ***Planet A: Geological Disaster Section*** is to develop a deep learning model that can detect 
changes in urban areas using satellite images.
Specifically, The geological disaster problem is to detect changes in five cities with unknown answers through  15 types
of channel data (various satellite bandwidth data) taken from 19 cities. [[A1]](##Reference) To solve the problem, We 
applied the multimodality of [[A3]](##Reference) to the Combination of Siamese network and NestedUNet structure 
presented in [[A2]](##Reference). Then, we used the channel-wise attention module used in the RCAN structure of 
[[A4]](##Reference) for each block.

### Climate Phenomena Segmentation
An atmospheric river (AR) is a narrow corridor or filament of concentrated moisture in the atmosphere
[[B1]](##Reference). ARs are a major cause of flooding and landslides in the western United States, so, 
it is important to detect and track ARs.
The purpose of the ***Planet A: Climate Science Section*** is to develop a deep learning model to detect and segment the
atmospheric river and typhoon through machine learning and deep learning.
Specifically, The climate science problem is to detect atmospheric river with 16 types of climate data.
Our team built a model similar to above one except for multi-modality.

## Requirements
- [PyTorch](https://pytorch.org/) (An open source deep learning platform)

## Reference 
```
# Geological Change Detection
[A1] Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2018, July. Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks. In IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2018 (pp. 2115-2118). IEEE.}
[A2] Fang, Sheng, et al. "SNUNet-CD: A densely connected Siamese network for change detection of VHR images." IEEE Geoscience and Remote Sensing Letters 19 (2021): 1-5.
[A3] Ebel, Patrick und Saha, Sudipan und Zhu, Xiao Xiang (2021) Fusing Multi-modal Data for Supervised Change Detection. ISPRS. XXIV ISPRS Congress 2021, 04 - 10 July 2021, Nice, France
[A4] Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." Proceedings of the European conference on computer vision (ECCV). 2018.
[A5] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

# Climate Phenomena Segmentation
[B1] Wikipedia contributors. (2022, September 19). Atmospheric river. In Wikipedia, The Free Encyclopedia. Retrieved 09:25, September 20, 2022, from https://en.wikipedia.org/w/index.php?title=Atmospheric_river&oldid=1111222086
[B2] Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." Proceedings of the European conference on computer vision (ECCV). 2018.
[B3] Fang, Sheng, et al. "SNUNet-CD: A densely connected Siamese network for change detection of VHR images." IEEE Geoscience and Remote Sensing Letters 19 (2021): 1-5.
```


