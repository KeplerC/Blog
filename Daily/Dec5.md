# Skip-Gram Modelling 
The paper studied is A Closer Look at Skip-gram Modelling by Guthrie et al and word2vec Parameter Learning Explained by Xin Rong. 

##### First of all, beyond n-grams 

In addition to n-grams, it allows tokens to be skipped. 


### definition 
we define k-skip-n-gram for a sentence w_1 ... w_m to be the set 
{w_i_1, w_i_2, ... w_i_n | \sum_{j=1}^n i_j - i_{j-1} < k}

In English, it is that for a certain skil distance k allow a total k or less skips to construct the n gram. For example, 12345 can be constructed for 2-skip-bi-gram to be {12, 13, 23, 24, 34, ... }

//TODO: TMW