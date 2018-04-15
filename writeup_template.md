
---

# **Vehicle Detection Project**

### **Author:** Daniela Yassuda Yamashita
### **Data:** 14/04/2018


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_not_car.png
[image2]: ./images/HOG_features_Cars.png
[image3]: ./images/HOG_features_notCars.png
[image4]: ./images/Spatial_features_Cars.png
[image5]: ./images/Spatial_features_notCars.png
[image6]: ./images/Histogram_features_Cars.png
[image7]: ./images/Histogram_features_notCars.png
[image8]: ./images/window_detection.png
[image9]: ./images/window_64.png
[image10]: ./images/window_96.png
[image11]: ./images/window_120.png
[image12]: ./images/window_180.png
[image13]: ./images/window_232.png
[image14]: ./images/heat_map.png
[image15]: ./images/apply_threshold.png

[video1]: ./Project5_final_video.mp4.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the section _"Step 2: Functions definition"_ in the sub-section  _"1) Color and HOG function"_. Indeed it is the function `get_hog_features()` in the third code cell of the IPython notebook (_Project5_Vehicle_Detection_v2.ipynb_). This function take as input the image (_img_), the orientation (_orient_) 
, pixel per cell (_pix_per_cell_), pixel per block (_cell_per_block_) and two boolean variables.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSL` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

* Car Images

![alt text][image2]

* Not car Images

![alt text][image3]


#### 1.1 Explain how (and identify where in your code) you extracted other features from the training images.

Besides the HOG feature, I combined two other features from the images: histogram feature and spatial feature.

With spatial feature I can take into account the colors position. On the other hand, from the histogram feature, I can highlight the average color of each image. As a result, these two features combined with the HOG feature allow the SVM has a better performance.

- **Example [Histogram feature]:**

![alt text][image6]

![alt text][image7]

- **Example [Spatial feature]:**

![alt text][image4]

![alt text][image5]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I realised that the color space was the parameter that has more impact in the performance of the HOG. I observed that increasing the number of orientation, the result was a little bit better. Therefore, my final pararemeters were:

* color_space='HLS'
* spatial_size=(32,32)
* hist_bins=32
* orient=9
* pix_per_cell=8
* cell_per_block=2
* hog_channel='ALL'

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained my SVM in a huge amount of images, in order to have a good accuracy. My data is splited as follow:

 1) VEHICLE IMAGES:
    * GTI_Far :  163 elements
    * GTI_Left :  134 elements
    * GTI_MiddleClose :  71 elements
    * GTI_Right :  125 elements
    * KITTI_extracted :  5967 elements
    * TOTAL:  6460 elements

2) NON-VEHICLE IMAGES:
    * Extras :  5074 elements
    * GTI :  3901 elements
    * TOTAL:  8975 elements

We can verify that the amount of images is equilibrated, once the total of vehicles images is very close to the the total of images of non-vehicle images. Therefore, it is reasonable to use all these images for training and test the SVM.

I have trained my SVM in the section _"Step 3: Train the SVM"_. First of all, I appended all my images in two lists: `notcars` and `cars` list. As a result I had this following length of lists:

```
Number of vehicle images: 6459
Number of non-vehicle images: 8973
```

Afterwards, I extracted all the features of each image. Using the function `single_img_features()`. This function take into account three types of features:
* spatial feature
* histogram feature
* hog feature

Finally, I trained my SVM using SVC function (`from sklearn.svm import SVC`). The input parameters are as beneath:
```
C=1.0
gama = 0.7
svc = SVC(kernel = 'linear', gamma=gama, C=C)
```
I used the function `train_test_split()` to create my test and training set. I splited them into 20% for the test set and 80% for the training set.

In order to be faster in the SVM training, I used AWS instance to train it. Therefore, I saved the SVM parameters in a [pickle file](./svc_pickle2.p)

As a result, I obtained 98.59% percent of accuracy in the test set.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search in 5 different window positions with 5 different sizes, as described above. 

| Position y direction | Size |
| ------------- | ------------- |
| [350, 500]  | (64,64)  |
| [400, 500]  | (96,96)  |
| [450, 600]  | (120,120)  |
| [400, 700]  | (180,180)  |
| [400, 700]  | (232,232)  |

* 64 pixels:

![alt text][image9]

* 96 pixels:

![alt text][image10]
* 120 pixels:

![alt text][image11]

* 180 pixels:

![alt text][image13]


* 232 pixels:

![alt text][image12]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using HSL 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

* **1st Step:** 

Detect the windows using the trained SVM with 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.

![alt text][image8]

* **2nd Step:**

Calculate the heat map.

![alt text][image14]


* **3rd Step:**

Apply threshold.

![alt text][image15]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./Project5_final_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

In the begining, I trained my SVM with only 1000 images of each category, as a result, the car identification didn't work well and there was a lot of false positives.

To overcome this problem, I increased the number of sample images to around 7000 images. Therefore, the accuracy was slight better and the car detection worked better.

Another trouble that I faced was a lot of false positives in the tree's shadow. Since the shadow appers in the left size of the image, I cut it off in the SVM analyses.

However, I still had the problem of false positives. In order to improve it, I used the last five frames in the video and increased the threshold in the heat map. Finally, the result zwas pretty better.



