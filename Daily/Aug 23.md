# Machine Learning in Spam Filtering 

I studied novelty detection for the last two months. Distinguishing  a novel pattern from many studied patterns can be a new step for me from one-class classification to general machine learning methods. I believe spam filter can be a good start. I will structure this blog by a review by Guzella and Caminhas[1].

I ran into this problem when I worked on ClassUCLA project: its latter iterations can send email to predefined accounts. Because of the privacy issues, I decide not to release those versions. At first I chose to use STMP servers of my Yahoo and a few other large companies' email account, but most of my emails were classified as spam email and failed to sent them. 

## The reason for data-driven spam filtering 
The old fashion way to do this is to filter certain keywords. For example, users can define keyword "stranger" such that all emails that have word stranger will be classified as spam email. However, spammers began to adopt content "obfuscation" to avoid these user defined filters, for example, using "sch001" instead of "school".  
The challenge faced by all spam filter is high cost of true-negative rate. 

## Basic Technique
For text-based spam filtering, it is more likely to do a text categorization or plagiarism detection. The following five steps are used to preprocess the information
1. tokenization: extract words
2. lemmatization: extract key words 
3. stop-word removal: remove common words like "to", "a", "for"
4. representation: convert the remaining information into machine learning algorithm input 
5. feature selection: selecting most representative features from previous steps

###Bag of Words(BoW) representation
A vector is usually used to represent a text. This technique is usually called bag-of-words. Given a set of _priori_ or predefined terms {t_1, t_2 ... t_N}, information can be represented by a N-dimensional feature vector. The feature vector can be formed token  by 
* a sliding window through a sequence of characters
* the number of occurrence of certain word, or n-gram model
and the feature can be represented by 
* binary representation: if certain word exist 
* frequency representation: the number of occurrences of that feature 
* term frequency-inverse document frequency: a log-based indication based on documents. 

Then we extract features by, for example, calculating information gain. We can calculate a frequency score to each term and select terms with highest core. Then we evaluate the difference of term-frequency variance for this document and that for all documents in the training set. By extracting features, we can focus more on meaningful and important features. 

The problem for this algorithm is similar to many probabilistic approaches: it can be represented with minimal space and ran with minimal time, but its cannot be trained online. As a result, the model cannot learn new spam messages by itself.


## Naïve Bayes 
Algorithms in this category are all based on Bayesian framework, which P(c|x)P(x) = P(x|c)P(c). P(x|c) is that when message belongs to c is known, the probability that x is in that message. P(x) is a _priori_ probability of a message by x. 

###Sahami et al. and Graham