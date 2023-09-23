# YOLO_Object_Detection
Dealing with sight loss is a major challenge due to limited resources, reduced accessibility, societal stigma, and increased risks for visually impaired individuals. To improve their lifestyle, thermal sensors and imaging are proving effective by detecting objects in various lighting conditions. We enhance object recognition for the visually impaired using image processing techniques like histogram equalization, grey level thresholding, deep learning, and convolutional neural networks. Our approach employs the YOLO V2 Network algorithm for training, incorporating high-resolution classifiers and anchor boxes to predict bounding boxes and recognize features, though ongoing development aims for even greater accuracy in object detection.

# Dataset:
Our Dataset contains over 14,000 thermal images, both synced annotated thermal imagery and non-annotated RGB imagery for reference. The camera centrelines approximately 2 inches apart and collimated to minimize 
parallax.
https://www.flir.com/oem/adas/adas-dataset-form/

# Project flow
![image](https://github.com/Srividya083/YOLO_Object_Detection/assets/145384296/077cc262-9eaf-42da-bd7e-98ff5c3ed8a3)

# Experimental Result Analysis
The performance of the proposed system is evaluated for various types of images in different environment from our dataset. Every image is processed using different image processing techniques like image enhancement using histogram equalization, restoration using IFFT, image segmentation using grey-level thresholding and morphological processing by dilation and erosion. After processing the dataset image, we imported pre-existing image sets for training. For training our model we use YOLOv2 which is a single-stage real-time object detection model with Darknet-19 as a backbone, batch normalizer, using anchor boxes to predict the bounding boxes. This transfer learning method allowed us to build models in accordance with similar and a larger dataset (vehicleTrainingData.mat). The vehicle training data is loaded into the workspace and the training samples are stores in a data directory. We then randomly shuffle data for training. We use preinitialized YOLO v2 object detection network (yolov2VehicleDetector.mat). The layer inspected has about 25x1 layer arrays containing layers for the input image, convolution, batch normalization, max pooling, ReLU, YOLO v2 transform layer, 4 anchor output.

![image](https://github.com/Srividya083/YOLO_Object_Detection/assets/145384296/5ab0fca8-a41f-43b7-9b3a-e46b39deb6b2)

The network training is configured using the training options. To train the YOLO v2 network, we use single CPU with different object classes using the command - [detector,info] = trainYOLOv2ObjectDetector(ds,lgraph,options);

![image](https://github.com/Srividya083/YOLO_Object_Detection/assets/145384296/922b8f58-646a-4ca5-9bd3-6d7f72e23439)

The RMSE is used to estimate the positional accuracy. We can verify the training accuracy of the detector by inspecting the training loss for each iteration and the graph is obtained.

![image](https://github.com/Srividya083/YOLO_Object_Detection/assets/145384296/a6a3c906-1772-43dc-90ce-61b1e1ea009c)

Images are classified using faster RCNN network to detect cars in an image and annotate with detection scores. The detector is uses modified version of MobileNet-v2 network architecture. We use functions from the deep learning toolbox for the process. 

To detect people in the image, we use the object peopleDetector that uses Histogram of Oriented Gradient feature and Support Vector Machine (SVM) classifier. 

We load the image and store the locations of the bounding boxes and their detection scores.

![image](https://github.com/Srividya083/YOLO_Object_Detection/assets/145384296/0adb9117-fb19-440c-95fb-c4d9c3424510)
