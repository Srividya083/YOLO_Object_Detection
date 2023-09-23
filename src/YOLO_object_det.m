clc;
clear all;
close all;
orgimg = imread("FLIR_06563.jpeg");
%Histogram equalization
f = histeq(orgimg);
montage({orgimg, f },"Size",[1 2]);
title("Original Image and Enhanced Images using histeq ");
%DFT Transform
f=imresize(f,[256 256])
figure,(imshow(f))
[M,N]=size(f);
h=fspecial('gaussian',260,2);
g=(imfilter(f,h,'circular'));
figure,imshow(g,[]);
G = fftshift(fft2(g));
figure,imshow(log(abs(G)),[]);
H = fftshift(fft2(h));
figure,imshow(log(abs(H)),[]);
F = zeros(size(f));
R=70;
for u=1:size(f,2)
 for v=1:size(f,1)
 du = u - size(f,2)/2;
 dv = v - size(f,1)/2;
 if du^2 + dv^2 <= R^2;
 F(v,u) = G(v,u)./H(v,u);
 end
 end
end
figure,imshow(log(abs(F)),[]);
fRestored = abs(ifftshift(ifft2(F)));
figure,imshow(fRestored, []), title('restored image');
%Gray level Thresolding
level=graythresh(f);
c= im2bw(f,level);
subplot(1,2,1), imshow(f),title('original image');
subplot(1,2,2), imshow(c),title('threshold image');
%Morphological processing
subplot(2,2,1), imshow(c),title('original image');
s=strel('line',3,3);
dilated=imdilate(c,s);
subplot(222), imshow(dilated),title('dilated image');
eroded=imerode(c,s);
subplot(223), imshow(eroded),title('eroded image'); 
figure,imshow(fRestored, []), title('restored image');
%Object Detection
data = load('vehicleTrainingData.mat');
trainingData = data.vehicleTrainingData;
dataDir = fullfile(toolboxdir('vision'),'visiondata');
trainingData.imageFilename = fullfile(dataDir,trainingData.imageFilename);
rng(0);
shuffledIdx = randperm(height(trainingData));
trainingData = trainingData(shuffledIdx,:);
imds = imageDatastore(trainingData.imageFilename);
blds = boxLabelDatastore(trainingData(:,2:end));
ds = combine(imds, blds);
net = load('yolov2VehicleDetector.mat');
detector = net.detector;
results = detect(detector, imds);
[ap, recall, precision] = evaluateDetectionPrecision(results, blds);
figure;
plot(recall, precision);
grid on
title(sprintf('Average precision = %.1f', ap))
lgraph = net.lgraph
lgraph.Layers
options = trainingOptions('sgdm',...
 'InitialLearnRate',0.001,...
 'Verbose',true,...
 'MiniBatchSize',16,...
 'MaxEpochs',30,...
 'Shuffle','never',...
 'VerboseFrequency',30,...
 'CheckpointPath',tempdir);
[detector,info] = trainYOLOv2ObjectDetector(ds,lgraph,options);
detector
figure
plot(info.TrainingLoss)
grid on
xlabel('Number of Iterations')
ylabel('Training Loss for Each Iteration')
fasterRCNN = vehicleDetectorFasterRCNN;
I = imread('FLIR_03447.jpeg');
[bboxes,scores] = detect(fasterRCNN,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
title('Detected Vehicles and Detection Scores')
peopleDetector=vision.PeopleDetector;
[bboxes,score]=peopleDetector(I);
if(sum(sum(bboxes))~=0)
 x=insertObjectAnnotation(I,'rectangle',bboxes,score);
 imshow(x);
 title('Detected People and detection scores');
else
 imshow(I);
 title('No People Detected');
end