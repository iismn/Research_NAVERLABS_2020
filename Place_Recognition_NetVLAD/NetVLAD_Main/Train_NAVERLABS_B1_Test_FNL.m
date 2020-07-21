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


%% C. NetVLAD Network Load
% C-1. Load NAVERLABS_NetVLAD Network
load('/home/iris_dl/IRiS_WS/SangMinLee/NAVERLABS_PlaceRecognition/Place_Recognition_NetVLAD/NetVLAD_Main/datasets/NAVERLABS_Indoor_B1_test.mat')       % Structured Data
load('/home/iris_dl/IRiS_WS/SangMinLee/NAVERLABS_PlaceRecognition/Place_Recognition_NetVLAD/NetVLAD_OutPut/B1/NetTotal/Test_PredefinedData.mat')       % Structured Data
load('/home/iris_dl/IRiS_WS/SangMinLee/NAVERLABS_PlaceRecognition/Place_Recognition_NetVLAD/NetVLAD_OutPut/B1/NetTotal/SavedSession.mat')                        % 1F

% C-2. Insert Network
net= relja_simplenn_tidy(finalNet);
netID = 'NAVERLABS_Indoor_B1';

% C-3. Setup Data Base (Test Set)
paths= localPaths();
dbTest= dbNaverLabs_B1_Test();
dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);

% C-4. NetVLAD Bag of Words Creation
% serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 10);
% serialAllFeats_test(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); 

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
Matched_Result_DB = [];
Matched_Result_Q = [];
tc = zeros(1969,5);

fileDB = fopen('Result_B1_DB_0720.txt','w');
fileQ = fopen('Result_B1_Q_0720.txt','w');

for iTestSample= 1:length(toTest)

    wait =1;
    iTest= toTest(iTestSample);
    ids= searcherRAW(iTest, nTop);

    PrvMatchedPoint = 0;
    matched_Index_Q_FNL = [];
    matched_Index_DB_FNL = [];
    for i = 1:5

        matched_Index_Q= dbStruct.qImageFns{iTestSample};
        matched_Index_DB = dbStruct.dbImageFns{ids(i)};
        
        fprintf(fileQ,'%s\n',matched_Index_Q);
        fprintf(fileDB,'%s\n',matched_Index_DB);
    end
    
    iTestSample
    
    Matched_Result_DB = [Matched_Result_DB; matched_Index_DB_FNL];
    Matched_Result_Q = [Matched_Result_Q; matched_Index_Q_FNL];
    
    
end

fclose(fileDB);
fclose(fileQ);