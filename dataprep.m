%%%This file is used to prepare the training data for pretraining the CNN

%%%Note: This pretrained CNN is used as part of the RL agent, but does not
%        train during the RL process.

%%%Note: CNN labels are one-hot encoded:
%           0 = Three targets detected
%           1 = Two targets detected - close
%           2 = Two targets detected - far
%           3 = One target detected - close
%           4 = One target detected - far
%%%

clear

%Initialize data matrix
data = zeros(5000,256,65);

%Load training data for case where all three targets are present
load('test1.mat')
labels = zeros(1000,256);
%Add training data/labels to data matrix
data(1:1000,:,1:64) = outmap;
data(1:1000,:,65) = labels;

%Load training data for case where two adjacent targets are present
load('test4.mat')
labels = ones(1000,256);
%Add training data/labels to data matrix
data(1001:2000,:,1:64) = outmap;
data(1001:2000,:,65) = labels;

%Load training data for case where two distant targets are present
load('test5.mat')
labels = 2*ones(1000,256);
%Add training data/labels to data matrix
data(2001:3000,:,1:64) = outmap;
data(2001:3000,:,65) = labels;

%Load training data for case where one nearby target is present
load('test6.mat')
labels = 3*ones(1000,256);
%Add training data/labels to data matrix
data(3001:4000,:,1:64) = outmap;
data(3001:4000,:,65) = labels;

%Load training data for case where one distant target is present
load('test7.mat')
labels = 4*ones(1000,256);
%Add training data/labels to data matrix
data(4001:5000,:,1:64) = outmap;
data(4001:5000,:,65) = labels;

%Shuffle training data/labels
shuffleddata = data(randperm(size(data,1)),:,:);

%Separate training data/labels into separte matrices
traindata = shuffleddata(:,:,1:64);
trainlabels = shuffleddata(:,1,65);

%Save training data and labels to file
filename = 'trainingdata.mat';
save(filename,'traindata','trainlabels')