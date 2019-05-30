# its_all_in_the_family

## Motivaiton
Identifying images through a picture is a very interesting topic. It becomes even more interesting by going one step further and differentiating between images. This project focuses on identifying unique features in images of faces and classfying given images as being blood related or not. This idea brings forth much real world implications and neural network theory advancements.

## Faces Dataset
The training and test images can be found through the "data/" path.

The images were obtained from a publicly open dataset called Faces in the Wild (http://vis-www.cs.umass.edu/lfw/). The dataset consists of 1000 different families with 100,000 individual images of faces that have already been resized and face centered with each other.

One caveat to this dataset is that every person did not have an equal amount of images. Some had 1 while others had 50 or more. Similarly, some families had more images than others e.g. family F0601 (The Royal Family) with 776 photos which is 6.27% of all the photos, comparitively to the average which was around 26 photos. This imbalance however is not large enough that it would impose a bias within the training set.

## Method
The method that is employed to classify the faces is the method of using Convolutional Neural Networks(CNN). More specifically, this project focuses on implementing a Siamese CNN. Siamese CNNs was chosen for its advantages over CNNs. As much of the classes, which were the families, was on average 26 photos, there wasn't much data to go for in terms of learning unique features. Siamese CNNs allowed for better classifications as its purpose was to focus its attention on two inputs. Regular CNNs are great at differentiating between photos and identifying what is e.g. hair, or even a mouth in photos. However, this problem looks for similarities between images and Siamese CNNs perform much better in that field.

The basic idea of Siamese CNNs is inputting two images on seperate CNNs which have the same architecture. These two sepearte CNNs output a feature vector corresponding to the input images. Some distance metric such as cosine distance, euclidean distance, or absolute distance is used between the two feature vectors which is then sent through a fully connected layer with a sigmoid activation function to output a score between 0 and 1. This represents the probably that the images are similar, with values closer to 1 labeled as more similar. 







