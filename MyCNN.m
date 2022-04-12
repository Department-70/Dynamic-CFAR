%%%This file is used to generate and train the pretrained CNN for the RL
%%%agent

clear

%Can initialize the RNG to allow for repeatability
% rng(2021) %Network training - run 2 (no MP) and 3 (with MP)
rng(5021) %Network training - run 4 (with MP)

%Load trainind data and labels
load('trainingdata.mat')

%Divide training data into training set (80%) and validation set (20%)
Xtrain = abs(traindata(1:4000,:,:,:));
Xtrain = permute(Xtrain,[2,3,4,1]);
Xval = abs(traindata(4001:5000,:,:,:));
Xval = permute(Xval,[2,3,4,1]);

%Divide training labels into training set (80%) and validation set (20%)
Ytrain = uint8(trainlabels(1:4000));
Ytrain = discretize(Ytrain,0:1:5, 'categorical');
Yval = uint8(trainlabels(4001:5000));
Yval = discretize(Yval,0:1:5, 'categorical');

%%%Define convolutional neural network:
%   1) Convolutional Layer
%   2) Batch Normalization Layer
%   3) ReLU Layer
%   4) Convolutional Layer
%   5) Batch Normalization Layer
%   6) ReLU Layer
%   7) Convolutional Layer
%   8) Batch Normalization Layer
%   9) ReLU Layer
%   10) Maxpooling Layer
%   11) Fully Connected Layer
%   12) Softmax Layer
%   13) Classification Layer
layers = [
    imageInputLayer([256 64,1])
    
    convolution2dLayer(3,32,'Stride',2)
    batchNormalizationLayer
    reluLayer   
    
    convolution2dLayer(3,64,'Stride',2)
    batchNormalizationLayer
    reluLayer 
    
    convolution2dLayer(3,128,'Stride',2)
    batchNormalizationLayer
    reluLayer 
    
    maxPooling2dLayer(2,'Stride',2)  
    
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];

%Set CNN training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',0.005, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

%Train the network, plotting training progress/accuracy
net = trainNetwork(Xtrain,Ytrain,layers,opts);
%Calculate validation accuracy of network
Ypred = classify(net,Xval);
Valaccuracy = sum(Ypred == Yval)/numel(Yval)