%% TRAIN ------------------------------------ 
% ---------- Full train and test example: Tokyo
% Train: Tokyo Time Machine, Test: Tokyo 24/7
clc; clear;
% Set up the train/val datasets
dbTrain= dbNaverLabs_B1('train');
dbVal= dbNaverLabs_B1('val');
lr= 0.0001;

% --- Train the VGG-16 network + NetVLAD, tuning down to conv5_1
sessionID= trainWeakly(dbTrain, dbVal, ...
    'netID', 'vd16', 'layerName', 'conv5_3', 'backPropToLayer', 'conv5_1', ...
    'method', 'vlad_preL2_intra', ... 
    'learningRate', lr, ...
    'doDraw', true,'useGPU', true);

% Get the best network
% This can be done even if training is not finished, it will find the best network so far
[~, bestNet]= pickBestNet(sessionID);

% Either use the above network as the image representation extractor (do: finalNet= bestNet), or do whitening (recommended):
finalNet= addPCA(bestNet, dbTrain, 'doWhite', true, 'pcaDim', 4096);
% 
% ,'Resize',[692 681],'CropAnisotropy',[1 1],'CropSize',[1 1],'CropLocation','center'