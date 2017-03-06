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
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
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

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used 4 different window sizes to capture cars at different positions. The 4 sizes are `192, 128, 96, 64`. In the figure below, the `192`-windows are drawn in red, `128`-windows in green, `96`-windows in blue and `64`-windows in purple.

The windows only cover the bottom third of the image. The sky and scenery in the top two-thirds, as will the very bottom 100 pixels will be ignored as they contain no relevant information to detect vehicles. Moreover, as cars will appear larger when they are near, and smaller when they are far apart, the bigger windows will only cover the very bottom of the chosen region, while the smaller windows will only cover the top portion of the chosen region.
![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

