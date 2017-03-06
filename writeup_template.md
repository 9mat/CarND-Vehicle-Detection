##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_noncar.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/HOG_orient.png
[image4]: ./output_images/windows.png
[image5]: ./output_images/bboxes.png
[image6]: ./output_images/heatmap_6frames.png
[image7]: ./examples/labels_map.png
[image8]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the  HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`on three color spaces `LUV`, `BGR` and `YCrCb`

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

##### Color space
As shown in the example above, `BGR` does not seem to differentiate well between cars and non-cars. All three channels of `LUV` and `YCrCb` show clear distintion between cars and non-cars, but the information seems to be substitution to each other (specially the `L` channel looks very similar to the `Y` channel, and the `UV` channels look very similar to the `CrCb` channels). As a result, and also due to computing time constraint, I have chosen only one of them (the `LUV` color space) for the `HOG` features used in the eventual classifier.

##### Parameters
I tried different sets of parameters to choose one that can capture as much information from HOG as possible without incurring too much computational and memory cost in terms of number of extracted features. For example the following figure shows the HOG visualization whne increase the number of orientation bins from 8 to 12 and to 16. There is some information gain from increasing `orient` to 12, indicated by a clearer shaped gradient of `HOG` with `orient=12` compared to when `orient=8`. However, goiing from `orient=12` to `orient=16` does not make any noticable change, while resulting in an increase of 30% in the number of features extracted. Eventually, I have chosen `orient=12` in the eventual classifier.

I did similar exercise for the other parameters and chose the following parameter combination: `orient=12, pixels_per_cell=(8,8), cells_per_block=(1,1)`.

![alt text][image3]

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In addition to the HOG, I also use color histogram and color bins from `HLS` color space as features for the classifier. The number of bins for the color histogram is set at 32 and the size for the spatial bins is set at `(16,16)`.

To account for the effects of cropping (due to sliding windows, as discussed later) and changing lighting conditions, I agumented the data by randomly croping and change brightness of the existing images in the training set.

I trained a linear SVM using `sklearn.svm.LinearSVC` class. I have also tried the general `sklearn.svm.SVC` classifier with different kernels. The `RBF` kernel is unacceptably slow. The `poly` kernel with `deg=3`gives very good performance (in both accuracy and precision) after using principle componenent ananlysis to extract top 500 features (accuracy and precision both above 99%). However it is still very slow. Therefore, in the end, I chose the linear kernel (and used `LinearSVC`) to achieve better speed with minimal compromise in accurary and/or precision (accuracy 97.7%, precision 96.4%, recall 99.1%).

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used 4 different window sizes to capture cars at different positions. The 4 sizes are `192, 128, 96, 64`. In the figure below, the `192`-windows are drawn in red, `128`-windows in green, `96`-windows in blue and `64`-windows in purple.

The windows only cover the bottom third of the image. The sky and scenery in the top two-thirds, as will the very bottom 100 pixels will be ignored as they contain no relevant information to detect vehicles. Moreover, as cars will appear larger when they are near, and smaller when they are far apart, the bigger windows will only cover the very bottom of the chosen region, while the smaller windows will only cover the top portion of the chosen region.

I used an overlapping of `0.75` for the three bigger windows, and `0.5` overlapping for the `64`-windows. The reason for smaller overlapping is due to computing time contraints. `64`-windows are the most numeruous, due to their small size, thus a small increase in the overlapping rate will increase the number of windows significantly. Moreoever, they cover regions far away from the driver, and hence less important compared to the other set of windows.

I ended up having a total of 376 windows.

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using `LUV` 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to my video result

[![Video](http://img.youtube.com/vi/SDJunGr37JQ/0.jpg)](http://www.youtube.com/watch?v=SDJunGr37JQ "Video Title")


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap.

As feature extraction is a very computationally expensive task, and the time spent for feature extraction is proportional to the number of search windows, it is essential to optimize the number of search windows. With the assumption that the video is a continuous stream of images from actual road, we can safely take as given that vehicles will gradually appear and disappear from the video over time, the their positions will remain stable one frame to the next. Thus, I only used the full set of 376 windows every 5 framesF For the rest of the frames, I only used the subset of windows that within 192 pixel from the center of the bounding boxes detected in the last frame. This trick increase the speed of the pipeline by about 3 times.

To smooth out the noise due to false positives, I make use of 2 adjustments. First, I keep track of the last 8 heatmaps. Second, instead of doing the thresholding on the heatmaps, I do the thresholding on the integrated heatmaps over time. The integrated heatmaps is constructed as an exponential moving average with the update rate of 0.2 and the decay rate of 0.9, i.e  `integratedheatmap(t) = 0.2*heatmap(t-1) + 0.9*integratedheatmap(t-1)`

The integrated heatmap will then be thresholded with the threshold being 2.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Compared to project 2 and 3 where we can use neural network as feature extraction, this project requres us to be very explitcit about the features that we employ in the classifier. Thus, the most challenging aspect of this project to me is how to fine tune various parameters (color space, bins, window size, etc...) to obtain good features for the classifier.

Fine tuning parameters depends a lot on a good validation strategy, especially considering the time series nature of the data. In the above implementation, I have overlooked the validation step, and I belive this is the reasons why, despite very high accuracy and precision, the classifier still ends up with a lot of false positive in the actual video.

The most important improvement, I believe, is to have a more systematic validation step to fine tune parameters. 

Another improvement would be the handling of multiple vehicles, especially when they block each other. `scipy.ndimage.measurements.label()` does take into account the time-dimension and assign separate labels if in a certain prior frame the vehicles do not block each other. Speed and location tracking could improve the this task significantly.

Tracking vehicles going oposite direction could be important in certain situations (when the road is narrow and there is no barrier), and it requires a different strategy in terms of thresholding and time-integrating the heatmaps.
