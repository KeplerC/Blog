# Generating Sequences with Recurrent Neural Networks 
by Alex Graves 

### Interesting Explaination of Neural Network
RNNs are ‘fuzzy’ in the sense that they do not use exact templates from
the training data to make predictions, but rather—like other neural network use their internal representation to perform a high-dimensional interpolation between training examples.

### RNN
standard RNN is unable to store information about past inputs for very long, and it generates based on only a few previous inputs. 

_____________________
 o  o o o o o o o 
 |  | | | | | | |
 h \h \ h \ h \ h 
 |  | | | | | | | 
 i  i i i i i i i 
______________________

h_t^n = H()
H is hidden layer function

### LSTM
In RNN, the hidden layer function is softmax function, but in LSTM, the whole cell is what we are familar with, which consists of forgetting gate, input, output and etc. 

### Discrete
Text Prediction 
* word level
* character-level language modelling: predict one character at a time, sometimes invent novel words/strings and generate flexibility 

#### Penn Treebank 
network architecture: 1000 LSTM units to compare word and character level of LSTM predictors on Penn corpus. Because the input texts are targets, the network can adapt its weights as it is being evaluated(which is called dynamic evaluation).

Results: the word-level RNN performed better than the character-
level network, but the gap appeared to close when regularisation is used.LSTM is better at rapidly adapting to new data than ordinary RNNs. 

#### Wikipedia Experiment 
Hutter Prize: to compress the first 100 million bytes of the complete English Wikipedia data (as it was at a certain time on March 3rd 2006) to as small a file as possible.

### Real continuous 

#### Handwriting Prediction
Online handwriting data: writing is recorded as a sequence of pen-tip locations. Dataset is from IAM-OnDB and original data is just x-y coordinates and end-of-stroke markes. 

Mixture density output: the sequence of probability distributions for the predicted pen locations as the word ‘under’ is written. By adding them together, we form a heatmap. 

#### Handwriting Synthesis 
It is to generate handwriting by given text. The main problem is two sequences have different length. 

#### future

### Thoughts 
This paper is so fantastic that includes many interesting experiments. I cannot elaborate more on it because of the limitation of markdown(like including images and formula). Highly recommended!