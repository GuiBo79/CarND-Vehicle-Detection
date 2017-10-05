## Writeup - Vehicle Detection
### This Writeup contains all the procedures and techniques used to perform vehicle detection in a given video.

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
[image1]: ./output_images/car_and_notcar.png
[image2]: ./output_images/HOG_car_image.png
[image3]: ./output_images/normalized_features.png
[image4]: ./output_images/find_cars.png
[image5]: ./output_images/heat_image.png
[image6]: ./output_images/pipeline_result_1.png
[image7]: ./output_images/pipeline_result_2.png
[video1]: ./vehicle_detection.mp4

## Project Files

1. writeup.md
2. find_cars.ipynb
3. lanesdetect.py
4. Lines.py
5. vehicle_detection.mp4

## Project Environment 

Intel® Core™ i5-5200U CPU @ 2.20GHz × 4 - 6Gb

GeForce 930M/PCIe/SSE2

Ubuntu 17.04 - 64 bits

Anaconda - Jupyter Notebook



### 1. Histogram of Oriented Gradients (HOG)

The HOG features are extracted by the function get_hog_features() in the first cell of the Python Notebook find_cars.ipynb

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

Finally , after some experiments, I decided to use the same parameters as described in the example above.


### 2. Training the classifier SVM and extract_features() Function

I trained a linear SVM using the GRidSearch to reach the best tunning. The code is located in the 4th cell of the jupyter notebook. 

    Using: 9 orientations 8 pixels per cell and 2 cells per block
    Feature vector length: 8460
    149.23 Seconds to train SVC...
    Test Accuracy of SVC =  0.9935

To extratct the features for the dataset was used the function extract_features() with the following parameters:

    color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [400, 720] # Min and max in y to search in slide_window()

    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    
        
![alt text][image3]



### 3.Sliding Window Search

The sliding window search is performed by the function find_cars() using HOG Sub-sampling. The function is located in the 6th cell of the jupyter notebook.
After experiments I decided to use the scales 1.0, 1.25, 1.50, 1.75 and 2.0 , this is necessary to detect the cars, but unfortunately decrease in a significant way the performance.

Every cicle the search steps 2 cells. (line 29)

Here is an example with scale = 1.0 . It's possible to see that the function detected some false positives.

![alt text][image4]

To prevent and decrease false positives was used the heat map. 

### 4. Multiple Detection (Add Heat & Threshold Function)

As mentioned , the heat function is used to filter false positives in the pipeline. The function add_heat() is located at the 7th cell of the jupyter notebook. After process the image is applied an threshold and result can be view above.


![alt text][image5]


### 5. Pipeline result in images 

Ultimately I searched on 5 scales using RGB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:



![alt text][image6]
![alt text][image7]


### 6. Video Implementation


Here's a [link to my video result](./project_video.mp4)


As done for images ,I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  To prevent blinking and over updating  , I created an global variable called prev_labels to record the previous label. I update the boxes every 8 cycles. 





### 7. Discussion

For me , this was the most challenger project and in my current professional situation was also the most valuable. Support Vector Machines fits perfectly for the project I am working on. One of the biggest challenges I had was to process the code in my laptop, in my company we are already investing in a workstation for me. I tried to use the Numba library to accelerate the process, but unfortunately did not work as I was expecting. 

I believe with a larger dataset and detecting in more scales the detection could be more precise, and I very conscious that have a lot to improve in this code. 

To this final project I also merge the lanes detection , but just a short video due my hardware limitations, anyway, I am very happy about everything I have learn in this course and I am very excited to start the next Term.






