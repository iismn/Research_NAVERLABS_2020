%% NAVERLABS Place Recognition Challenge 2020
%
%
% Lee Sang Min 
% Civil and Evironmental Engineering
% iismn@kaist.ac.kr
% KAIST W16 Geocentrifuge Research Center IRiS Lab.

%% A. ROS Init
% A-1. Clear Work Space
clc; clear; close all; rosshutdown

% % A-2. ROS initialize 
% IP_BackEnd = '192.168.1.14';
% rosinit(IP_BackEnd,11311);
% 
% %% B. ROS Communication : Publish Topic
% % B-1. Set Publisher / Image Transport
% [jointTauPub_1, jtMsg_1] = rospublisher('/test_features');
% [jointTauPub_1, jtMsg_1] = rospublisher('/test_name');
% [jointTauPub_1, jtMsg_1] = rospublisher('/train_features');
% [jointTauPub_1, jtMsg_1] = rospublisher('/train_name');

%% C. NetVLAD Network Load
% C-1. Load NAVERLABS_NetVLAD Network
load('/home/iris_dl/IRiS_WS/Sang Min Lee/NAVERLABS_PlaceRecognition/Place_Recognition_NetVLAD/NetVLAD_Main/datasets/NAVERLABS_Indoor_B1_test.mat')       % Structured Data
load('/home/iris_dl/IRiS_WS/Sang Min Lee/NAVERLABS_PlaceRecognition/Place_Recognition_NetVLAD/NetVLAD_OutPut/B1/NetTotal/Test_PredefinedData.mat')       % Structured Data

% load('/home/iismn/IISMN_CODE/NAVER_LABS/IISMN/Place_Recognition_NetVLAD/NetVLAD_OutPut/1F/NetTotal/SavedSession.mat')                        % 1F
% load('/home/iismn/IISMN_CODE/NAVER_LABS/IISMN/Place_Recognition_NetVLAD/NetVLAD_OutPut/1F/NetTotal/SavedSession.mat')                   % 2F

% C-2. Insert Network
net= relja_simplenn_tidy(finalNet);
netID = 'NAVERLABS_Indoor_B1';

% C-3. Setup Data Base (Test Set)
paths= localPaths();
dbTest= dbNaverLabs_B1_Test();
dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);

% C-4. NetVLAD Bag of Words Creation
serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 10);
serialAllFeats_test(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); 

%% D. Place Recognition Main
% D-1. Layer Parameter Set
opts= struct(...
    'nTestRankSample', 0, ...
    'nTestSample', inf, ...
    'recallNs', [1:5, 10:5:100], ...
    'margin', 0.1, ...
    'nNegChoice', 1000, ...
    'cropToDim', 0, ...
    'printN', 10 ...    
    );

opts= vl_argparse(opts, {});
relja_display('testFromFn:\n%s\n%s', dbFeatFn, qFeatFn);

% D-2. Query Feature / DB Feature Load
qFeat= fread( fopen(qFeatFn, 'rb'), inf, 'float32=>single');
qFeat= reshape(qFeat, [], dbTest.numQueries);
nDims= size(qFeat, 1);
dbFeat= fread( fopen(dbFeatFn, 'rb'), [nDims, dbTest.numImages], 'float32=>single');
assert(size(dbFeat,2)==dbTest.numImages);

% D-3. Loss Function Express
searcherRAW= @(iQuery, nTop) rawNnSearch(qFeat(:,iQuery), dbFeat, nTop);
toTest= 1:dbTest.numQueries;
nTop= max(opts.recallNs);

%% E. Main Loop
for iTestSample= 1:length(toTest)

    wait =1;
    iTest= toTest(iTestSample);
    ids= searcherRAW(iTest, nTop);
   
%     figure(1)
%     refresh
%     subplot(2,5,1:5)
%     imshow(dbStruct.qImageFns{iTestSample})
%     title('Query Image')
%     subplot(2,5,6)
%     imshow(dbStruct.dbImageFns{ids(1)})
%     title('Retrived Image')
%     subplot(2,5,7)
%     imshow(dbStruct.dbImageFns{ids(2)})
%     title('Retrived Image')
%     subplot(2,5,8)
%     imshow(dbStruct.dbImageFns{ids(3)})
%     title('Retrived Image')    
%     subplot(2,5,9)
%     imshow(dbStruct.dbImageFns{ids(4)})
%     title('Retrived Image')   
%     subplot(2,5,10)
%     imshow(dbStruct.dbImageFns{ids(5)})
%     title('Retrived Image')

    PrvMatchedPoint = 0;
    for i = 1:5
        
        Origin_Query_Img = imread(dbStruct.qImageFns{iTestSample});
        Origin_DB_Img = imread(dbStruct.dbImageFns{ids(i)});

        Query_Img = rgb2gray(imread(dbStruct.qImageFns{iTestSample}));
        DB_Img = rgb2gray(imread(dbStruct.dbImageFns{ids(i)}));

        % Detect dense feature points (Dense Feature)
        imagePoints_Q = detectBRISKFeatures(Query_Img);
        imagePoints_DB = detectBRISKFeatures(DB_Img);
        [features_Q,valid_points_Q] = extractFeatures(Query_Img,imagePoints_Q);
        [features_DB,valid_points_DB] = extractFeatures(DB_Img,imagePoints_DB);
        indexPairs_Prv= matchFeatures(features_Q,features_DB);

        matchedPoints_Q = valid_points_Q(indexPairs_Prv(:,1),:);
        matchedPoints_DB = valid_points_DB(indexPairs_Prv(:,2),:);

        if size(matchedPoints_Q,1) > PrvMatchedPoint
            matched_Image_Q = Origin_Query_Img;
            matched_image_DB = Origin_DB_Img;
            matched_Point_Q_FNL = matchedPoints_Q;
            matched_Point_DB_FNL = matchedPoints_DB;
            matched_Index_Q_FNL = dbStruct.qImageFns{iTestSample};
            matched_Index_DB_FNL = dbStruct.dbImageFns{ids(i)};
            PrvMatchedPoint = size(matchedPoints_Q,1);
            

        end
    end
    
    iTestSample

%     [fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
%       matched_Point_Q_FNL, matched_Point_DB_FNL, 'Method', 'RANSAC', ...
%       'NumTrials', 10000, 'DistanceThreshold', 0.1, 'Confidence', 99.99);
% 
%     matched_Point_Q_FNL = matched_Point_Q_FNL(epipolarInliers, :);
%     matched_Point_DB_FNL = matched_Point_DB_FNL(epipolarInliers, :);
    
    
%     figure(2)
%     refresh
%     ax = axes;
%     showMatchedFeatures(matched_Image_Q,matched_image_DB,matched_Point_Q_FNL,matched_Point_DB_FNL,'montage','Parent',ax);    
    
    Matched_Result_DB{iTestSample} = matched_Index_DB_FNL;
    Matched_Result_Q{iTestSample} = matched_Index_Q_FNL;
    
%     jtMsg_1.Data = tau(1);
%     send(jointTauPub_1,jtMsg_1);
%      jtMsg_2.Data = tau(2);
%     send(jointTauPub_2,jtMsg_2);
%      jtMsg_3.Data = tau(3);
%     send(jointTauPub_3,jtMsg_3);
%      jtMsg_4.Data = tau(4);
%     send(jointTauPub_4,jtMsg_4); 
    
%     while wait == 1
%         msg2 = receive(sub,10);
%         wait = msg2.answer;
%     end
    
end

































