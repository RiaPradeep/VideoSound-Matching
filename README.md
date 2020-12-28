# Audio Classification

The aim of the project is to see if the given audio and video sample belong to the same category. We've also presented three models for this problem, each of which first creates an encoding for the audio and video input independently, using various techniques, then maps them to the same space. Our results show that these models perform well with various loss functions. We also presented a novel loss function for this problem, Multi-Similarity, which builds on Cosine-BCE with increased components in the regularized term.

# Models
The models folders consists of two subfolders- AudioEncoders and VideoEncoders. 

## Audio Encoders
This subfolder consists of models for creating the audio encodings given the audio in STFT format. Three models are presents
* CNN: The STFT input is regarded as an image, and a 2D-convolutional network is used to construct the encoding
* LSTM; The STFT input is regarded as a temporal input, and passed through a bidirectional LSTM to construct the encoding
* Transformer: The STFT input is passed through a classical multi-headed transformer to construct the encoding

## Video Encoders
This subfolder consists of models for creating the video encodings given the video. Three models are presents- 
* CNN: This considers the temporal dimension of the input as the third dimension, and used 3D-convolutional layers to construct the encoding
* CNN_LSTM : The input is first reshaped to include Time in the batch dimension, and then passed through a 2D-convolutional layer. The time dimension of the output is then expanded and the result is passed through a LSTM layer to capture the temporal dependencies of the audio. 
* CNN_AVG: In this model, we pass each of the input frames through a convolutional layer and then take the average of the results to find the final encoding
* Transformer: Each frame is first passed through a pretrained ResNet and then through a classical multi-headed transformer to construct the encoding

## Classification models
These files are including in the models folders and combine different video and audio encoders to construct the final output. Fusional siamese models are used after the encodings are constructed. 

# Loss
In addition to different models, the project tests different losses for training including binary cross entropy, contrastive loss, cosine loss, and triplet loss(usual definitions). in addition we test a new loss-Multi loss specific to this problem that enforces the relations between the different modalities for training. 



