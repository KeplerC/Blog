# Neural Translation of Musical Style
This is a model that can learn to perform sheet music like human does, which can pass "musical Turing tests". The key for just playing a note and human-performed music is a global structure and dependency of music. 
This is why we have LSTM to solve it. RNN: of course it has vanishing gradient problem. 


### architecture: GenreNet
Sheet music -> birectional LSTM -> linear layer ->dyanmics 
* bidirectional LSTM: "can look ahead" 
* linear layer: activation function 

StyleNet: rendition model that can play variety of styles 
    Interpretation layer: converts musical input into own representation
    
# thoughts 
One of our experiment as previous encoder-decoder neural network is this "musical turing tests". We produce sounds and let them pass "sound turing test". 