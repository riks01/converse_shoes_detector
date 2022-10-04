# Converse Logo Detection

<b><i>Note:</b></i> I have trained both yolov4 and yolov5 model on converse shoes images, for time being we will going to cover yolov4 model only.

## Predictions:

## YOLOv4

<img src='yolov4_pred.jpg'>

## YOLOv5

<img src='yolov5_pred.jpg'>
<b><i>Note:</i></b> Even though the prediction by YOLOv5 for the image looks weaker than YOLOv4, but if you compare the bouding boxes enclosing the logo by both the model. YOLOv5 works pretty well!

## Traning Results:

## YOLOv4

<img src='yolov4_train.png' height=650, width=750>

## YOLOv5

<b>mAP50 95%</b>
<img src='yolov5_train.png'>

> <b><i>Note:</i></b> YOLOv5 took just 200 epochs to get mAP50 95% on the other hand YOLOv4 took almost 3000 iterations to reach to mAP50 92%

## Folders and File:
1. yolov4: This folder contains all the required files along with images data and colab notebook with trained weights to build or test yolov4 model.
2. yolov5: This folder contains all the required files along with images data and colab notebook with trained weights to build or test yolov5 model.
3. static: This folder contains images of yolov4 architecture, results and predictions to display in readme.md file
4. bbox_tool.py: It is a desktop application (tkinter) which is used to find (x, y, w, h) co-ordinates of drawn bounding box over objects. 
5. process.py: This file is use to convert the co-ordinates of bouding box into yolo expected input format i.e (class, x, y, w, h).
6. convert.py: Storing images path for training and testing by creating train.txt and test.txt files.
7. rename.py: Contains a function to rename large number of files.
8. resize.py: Contains a function to resize the image to a pariticular dimension.


## Data:

The images of the converse shoes is downloaded from <a href="(https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged)">kaggle<a>, it contains three classes of colored image data namely Nike, Adidas and Converse. Since we are doing single object detection so I kept Converse data only and I have also downloaded some of the latest shoes images of converse from web to increase training samples and converted that to 240x240 dimension using resize.py. There are around 350 images samples present in the data folder.

## Traning Steps:

<b>1. Preparing data expected by YOLOv4: YOLOv4 expects data in the format of -> (class, x, y, w, h), to get this we will </b>
  <ol type='i'> 
  <li> draw bounding box over objects for each sample images using bbox_tool.py this will give (x, y, w, h) </li>
  <li> then use process.py file to insert classes that results in labels (class, x, y, w, h) </li>
  <li> we will create a folder e.g 'multiple_images' and will add all the images and labels in that folder (will be used during training)</li>
  </ol>

**2. Downloading Darknet YOLOv4 Object Detection Framework:**
  
Open a colab notebook inside a folder of google drive, and just clone the repository of darknet by running the following command 
    
  
    !git clone https://github.com/AlexeyAB/darknet'
  
  
**3. Making changes in the configuration file (yolov4_custom.cfg):**
  
Copy yolov4_custom.cfg folder and put it int a data folder and do the below mentioned changes
<ol type='i'>
   <li> Change batch=1 from batch=64 </li>
  <li> Change subdivision=1 from subdivision=16 </li>
  <li> Change max_batches = 500500 to max_batches = 4000 (max_batches=classes*2000; and atleast it has to be 4000) </li>
  <li> Change line classes=80 to 1 in each of 3 [yolo]-layers. classes=1 </li>
  <li> Change filters from filters=255 to filter = 18 (classes + 5)x3 in the 3 [convolutional] before each [yolo]-layer </li>
  <li> Change steps to 80% and 90% of max_batches. In our case we have 1 classes so max_bathces are 4000. steps=(4800,5400) </li>
  </ol>

**4. Upload multiple_images file,convert.py file along with .data and .names file (eg. piford.data and piford.names) to the data folder of darknet**

  .data file will store:
    
    classes= 1 
    train  = data/train.txt  
    valid  = data/test.txt  
    names = data/piford.names  
    backup = backup/

  .names file will store:

    converse 

**5. Rest all the steps are straightforward and is being done in the colab notebook which is in yolov4 folder, please refer that file.**


# About Model:

## You Only Look Once (YOLO):

The original YOLO (You Only Look Once) was written by Joseph Redmon in a custom framework called Darknet. Darknet is a very flexible research framework written in low level languages and has produced a series of the best realtime object detectors in computer vision: YOLO, YOLOv2, YOLOv3, YOLOv4.
  
The Original YOLO -  YOLO was the first object detection network to combine the problem of drawing bounding boxes and identifying class labels in one end-to-end differentiable network.

YOLOv2 -  YOLOv2 made a number of iterative improvements on top of YOLO including BatchNorm, higher resolution, and anchor boxes.

YOLOv3 - YOLOv3 built upon previous models by adding an objectness score to bounding box prediction, added connections to the backbone network layers, and made predictions at three separate levels of granularity to improve performance on smaller objects.

> <i><b>As of now there are total 7 versions of YOLO available, since our task of detecting a single object from an image is fairly simple and we managed to achieve mAP50 (popular evaluation metric for object detection, explained shortly) of 92 and 95 by using YOLOv4 and YOLOv5 models respectively.</b></i>

  
## YOLOv4:
  <img src='y4_architecture.png'>
  
### Building blocks of the YOLOv4:

  1) Backbone: Backbone is the deep learning architecture that basically acts as a feature extractor. All of the backbone models are basically classification models. 
  
  2) Neck: Neck is a subset of the Bag of Specials (BOS), it basically collects feature maps from different stages of the backbone. In simple terms, it’s a feature aggregator. 
  
  3) Head: Head is also known as the object detector, it basically finds the region where the object might be present but doesn't tell about which object is present in that region. We have two-stage detectors and one stage-detectors which are further subdivided into anchor-based and anchor-free detectors.
  
  ### Bag of freebies (BOF):
  
  <img src='y4_bof.png'>
  
  Bag of freebies are those methods that only change the training strategy or only increase the training cost (nothing to do with inference). You can see from the above diagram that there are an insane amount of things you can try, but we will discuss only the most important ones.
  
  ### Bag of specials (BOS):
  
  <img src='y4_bos.png'>
  
  Bag of specials are those Plugin modules and post-processing methods that only increase the inference cost by a small amount but can significantly improve the accuracy.
  
## Key changes in YOLOv4 considering previous versions:
  
1) Mosaic: Allows detection of objects outside their normal context
  
2) Self-Adversarial Training (SAT): We know that DL performs super bad with adversarial data and thus YOLOv4 uses SAT such that it introduces the X amount of perturbation to the training data till the predicted label remains the same as the original class. This helps the model to become more generalized.
  
3) Cross-Stage Partial Connection (CSP): The DenseNet has been edited to separate the feature map of the base layer by copying it and sending one copy through the dense block and sending another straight on to the next stage. 
  
4) Cross-Iteration mini Batch Normalizaion (CmBN): To improve the calculated statistics for BN it considers statistics of previous time stamps.
  
5) Spatial Pyramid Pooling (SPP): SPP use Conv layers to extract image's feature map, then use the max pool of window size_1 to generate a feature set, then repeat this n times and will have different feature maps in height and width dimension, thus it makes a pyramid. YOLOv4 takes this to the next step, instead of applying SPP it divides the feature along depth dimension, applies SPP on each part, and then combines it again to generate an output feature map.
  <img src="y4_spp.png">
  
6) Spatial Attention Module (SAM): Uses an attention mechanism along with both filters (depth) and spatial (width and height).
  
7) PAN Path - Aggregation Block: PANet is basically a Feature pyramid network that extracts important features from the backbone classifier. It uses SPP to make this FPN possible.
  
 ## Evaluation Metric (mAP50):
  
  <b>Let's first understand the term IoU (Intersection over Union).</b>
  
  > IoU measures the overlap between 2 boundaries. We use that to measure how much our predicted boundary overlaps with the ground truth (the real object boundary). In some datasets, we predefine an IoU threshold (say 0.5) in classifying whether the prediction is a true positive or a false positive.
  
  ### mAP50:
  
  > The mAP for object detection is the average of the AP (Average Precision) calculated for all the classes. mAP@0.5 means that it is the mAP calculated at IOU threshold 0.5. The general definition for the Average Precision(AP) is finding the area under the precision-recall curve.


