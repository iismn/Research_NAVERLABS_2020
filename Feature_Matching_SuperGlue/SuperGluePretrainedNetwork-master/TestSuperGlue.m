clc; clear; close all;

LeftImg = imread('22970289_1555396661779869.jpg');
RightImg = imread('40027089_1566000000000067.jpg');

% SuperFeatureLeftMatched = readNPY('SuperFeatureLeftMatched.npy');
% SuperFeatureRightMatched = readNPY('SuperFeatureRightMatched.npy');
% SuperGlueMatch = readNPY('matches.npy');
% SuperGlueConfidence = readNPY('match_confidence.npy');

% [SuperGlueConfidenceIndex,~] = find(SuperGlueConfidence(:,1)>0.45);
% [SuperGlueIndexLeft,~] = find(SuperGlueMatch(:,1)>0);
% SuperGlueIndexRight = SuperGlueMatch(SuperGlueConfidenceIndex);

% SuperFeatureLeftMatched = SuperFeatureLeft(SuperGlueConfidenceIndex,:);
% SuperFeatureRightMatched = SuperFeatureRight(SuperGlueIndexRight+1,:);


% figure(1)
% subplot(1,2,1)
% imshow(LeftImg); hold on;
% plot(SuperFeatureLeft(:,1), SuperFeatureLeft(:,2),'yx')
% 
% subplot(1,2,2)
% imshow(RightImg); hold on;
% plot(SuperFeatureRight(:,1), SuperFeatureRight(:,2),'yx')

figure(2)
ax = axes;
showMatchedFeatures(LeftImg,RightImg,SuperFeatureLeftMatched,SuperFeatureRightMatched,'montage','Parent',ax);
pause


