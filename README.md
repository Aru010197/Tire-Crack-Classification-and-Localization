# Tire-Crack Classification and Localization
The tire crack project aims to develop a machine learning model that can accurately classify tire images as either "cracked" or "not cracked and identify where the cracks are by inputting an image by using tensorflow.
Detection of surface cracks is an important task in monitoring the structural health of tires. The manual process of crack detection is painstakingly time-consuming and suffers from subjective judgments of inspectors. Manual inspection can also be difficult to perform in case of high-rise buildings and bridges. In this blog, we use deep learning to build a simple yet very accurate model for crack detection. Furthermore, we test the model on real world data and see that the model is accurate in detecting surface cracks in tire.

## Dataset

In our project, starting with only 20 images of cracked and healthy tires, used the Augmenter library to increase the dataset size. Out of the initial 20 images of tires, data augmentation techniques were applied to generate a larger dataset of 8000 images.

<img width="600" alt="image" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/461be698-fcfc-4c90-9995-cdb89460cb89">

### Normal Tire

<img width="200" alt="image" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/b5223ce8-32c9-456c-b106-7a1c305e5583"> <img width="150" alt="image" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/7db0e690-6979-4a68-b8d3-77eb1f512964"> <img width="200" alt="image" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/2a8a82f2-ba03-472b-882f-0ee961315047">

### Cracked Tire

<img width="200" alt="image" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/a4873006-187c-4b44-aa9e-c3f29028bf55"> <img width="170" alt="image" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/d283f87e-b525-4d75-b6f4-42bc5c085422"> <img width="200" alt="image" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/154044b7-599b-46d4-a130-8604569acf9e">


## Model Built

For this problem, We have build a Convolution Neural Network (CNN) in TensorFlow. Since we have a limited number of images, we will use a pretrained network as a starting point and use image augmentations to further improve accuracy. Image augmentations allow us to do transformations like — vertical and horizontal flip, rotation and brightness changes significantly increasing the sample and helping the model generalize.

The training dataset is divided into 80% for training and 20% for testing. The training set is used to train the machine learning model, allowing it to learn patterns and classify tire images accurately. The remaining 20% serves as an independent testing set to evaluate the model's performance on unseen data. This split ensures unbiased evaluation and helps validate the model's ability to generalize to new tire images.

We have used VGG16 & Inception Ensemble model and Inception & ResNet Ensemble model for Image classification and YOLOv8 for image segmentation.


### Evaluating Performance of Image Classification Models

<img width="805" alt="Screenshot 2023-11-07 at 3 23 45 AM" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/77816d0c-5426-4af6-ac60-329fbee853b0">

### Conclusion 

Achieving an accuracy of 83% on crack detection using the Inception and ResNet models is a significant accomplishment. This high accuracy indicates that the models are successfully distinguishing between cracked and non-cracked tire images with a high level of precision. With an accuracy of 83%, the model shows promising results and can potentially be utilized in various real-world scenarios, such as tire inspection systems, automotive safety applications, or quality control in tire manufacturing processes.

### Image Segmentation Using YOLOv8

Semantic segmentation is a deep learning algorithm that associates a label or category with every pixel in an image. It is used to recognize a collection of pixels that form distinct categories. YOLOv8, like other YOLO models, is an acronym for "You Only Look Once," referring to its ability to perform object detection tasks in a single forward pass of the neural network. However, YOLOv8 is much more than just a faster version of its predecessor. It is the newest state-of-the-art model in the YOLO family, featuring a new backbone network, a new loss function, a new anchor-free detection head, and other architectural improvements.

<img width="400" alt="image" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/1e2b7e67-a16b-4fe3-bc3b-952da00ba991">


## Steps for using cvat.ai for annotations:
    Access the Platform: Visit the cvat.ai website and log in to your account.

      1.	Create a Task: Start by creating a new annotation task. Provide relevant details such as the task name, project,and data source. 

      2.	Upload Data: Upload the images or videos that need annotation to the task.

      3.	Define Annotation Types: Specify the annotation types. For object detection for cracks, we used polyline shape and for image segmentation we used polygon shape.

      4.	Annotate Data: Open the task and begin annotating the data. Use the provided tools to draw polyline and polygons, as needed. 

      5.	Labeling Guidelines: Ensure consistency by providing labeling guidelines for annotators to follow. 

      6.	Quality Control: Review and verify annotations for accuracy and consistency. 

      7.	Save Annotations: Save the annotated data within the platform. 

      8.	Export Data: Export the annotations in the YOLO 1.1 format, for integration into the machine learning pipeline. 

      9.	Feedback and Iteration: If required, provide feedback to annotators for continuous improvement and iterate on the annotation process. 

      10.	Project Management: Utilize cvat.ai's project management features to organize and monitor multiple annotation tasks.



### Performance 

<img width="600" alt="image" src="https://github.com/Aru010197/Tire-Crack-Classification-and-Localization/assets/150113909/8fa67f0a-46c0-4630-82e3-fe230b491faf"> 

Considering the provided dataset, the YOLOv8 model demonstrated commendable performance by effectively localizing cracks within the images. However, there is a notable potential for further enhancement in its performance. This can be achieved by expanding the dataset size and incorporating a greater number of annotated images. The inclusion of more diverse instances and scenarios of cracks will enable the model to develop a broader understanding of crack patterns, leading to improved accuracy and robustness in crack localization. Therefore, by augmenting the dataset and increasing the volume of annotated images, the YOLOv8 model's performance can be elevated to even higher levels, delivering more precise and reliable results.


## Applications for the models created:

  •	Car Parts Inspection: The model enables automated inspection of cracks in other parts of car like-Engine Components, Bumper, dashboard, door panels, and center console.

  •	Quality Assurance: It assists in automating tire inspection processes, improving the accuracy and efficiency of quality control checks during tire manufacturing.

  •	Safety Enhancement: By identifying tire cracks accurately, the model contributes to enhanced safety standards and reduces the risk of tire failures.

  •	Predictive Maintenance: The model's crack detection capabilities enable timely identification of potential tire issues, leading to proactive maintenance and cost savings.








