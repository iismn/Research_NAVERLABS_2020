%% Test ------------------------------------ 
%
% ---------- Use/test our networks
clc; clear;
% Load our network
load('/home/iismn/IISMN_CODE/NAVER_LABS/IISMN/Place_Recognition_NetVLAD/NetVLAD_OutPut/1F/NetTotal/SavedSession.mat')
net= relja_simplenn_tidy(finalNet);
netID = 'NAVERLABS_Indoor_1F';
%  To test the network on a place recognition dataset, set up the test dataset
dbTest= dbNaverLabs_1F_Test();

% Set the output filenames for the database/query image representations
paths= localPaths();
dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);

% Compute db/query image representations
serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 10); % adjust batchSize depending on your GPU / network size
serialAllFeats_test(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1

% Measure recall@N
[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
plot(opts.recallNs, recall, 'ro-'); grid on; xlabel('N'); ylabel('Recall@N');