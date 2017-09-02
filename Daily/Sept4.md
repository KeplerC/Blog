




### OCR

Photo Character Recognition Pipeline
* Text detection
* character segmentation
* character classification 

#### Sliding window
taking a rectangular patch throughout the image
need to define step size and then take larger/smaller image patch 

For text detection, 
use a classifier to find where text has highest probability 
then use expansion method to determine the sliding window of the image

then we use a 1D sliding window to go through image patch to find gaps: character segmentation 

#### Application 
##### artificial data synthesis 
we produce a letter in different background and have different picture as synthetic data

##### Introducing distortions 
we can distort the training set to create more 
another application is speech recognition, also have a lot of noisy background 

however, having same value again and again will result a same theta

Discussion
* make sure to have a low bias classifier 
* "how much work would it be to get 10x as much data as we currently have" 


##### ceiling analysis 
the earlier the stage it is, the more influencial the stage is. 
i.e.
the ceiling the the next stage is determined by the previous stage

