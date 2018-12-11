
reqToolboxes = {'Computer Vision System Toolbox', 'Image Processing Toolbox'};
if( ~checkToolboxes(reqToolboxes) )
 error('detectFaceParts requires: Computer Vision System Toolbox and Image Processing Toolbox. Please install these toolboxes.');
end

videoFileReader = vision.VideoFileReader('data/adrian/adrian.mp4');
img             = step(videoFileReader);

detector = buildDetector();
[bbox bbimg faces bbfaces] = detectFaceParts(detector,img,2);

figure;imshow(bbimg);
for i=1:size(bbfaces,1)
 figure;imshow(bbfaces{i});
end

% Please uncoment to run demonstration of detectRotFaceParts

 img = imrotate(img,180);
 detector = buildDetector(2,2);
 [fp bbimg faces bbfaces] = detectRotFaceParts(detector,img,2,15);

 figure;imshow(bbimg);
 for i=1:size(bbfaces,1)
  figure;imshow(bbfaces{i});
 end
