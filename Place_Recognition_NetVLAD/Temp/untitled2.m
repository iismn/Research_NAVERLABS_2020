srcFiles = dir('NetVLAD_TestDataset/Test/B1/images/*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
filename = strcat('NetVLAD_TestDataset/Test/B1/images/',srcFiles(i).name);
im = imread(filename);
k=imresize(im,0.45);
newfilename=strcat('NetVLAD_TestDataset/Test/B1/images/',srcFiles(i).name);
imwrite(k,newfilename,'jpg');
i
end