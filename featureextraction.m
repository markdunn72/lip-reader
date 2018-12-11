function featureextraction(filename)
% Config
fps         = 59.94;
minQuality  = 0.005;
cornerSize  = 0.12;
dataPoints  = 100;
saturation  = 1.8;
refreshRate = 80; 

% Create a detector object
detector = buildDetector(2,2);

% Read a video frame and run the face detector.
videoFileReader = vision.VideoFileReader(filename);
videoFrame      = step(videoFileReader);
[bbox, ~, ~, ~] = detectFaceParts(detector,videoFrame,2);
bboxMouth       = bbox(:,13:16);

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bboxMouth);

% Edge detection
% [E,thresh] = edge(rgb2gray(videoFrame/255),'canny');
% imshow(E)

% hsv colorspace
videoFrameF = rgb2hsv(videoFrame);
% increase saturation
videoFrameF(:, :, 2) = videoFrameF(:, :, 2) * saturation;
points = detectHarrisFeatures(rgb2gray(videoFrameF), ...
    'ROI', bboxMouth, 'MinQuality', minQuality);
points = points.selectStrongest(dataPoints);

pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
points = removeCornerPoints(points, cornerSize);

initialize(pointTracker, points, videoFrame);


        

% x = double(points(:,1));
% y = double(points(:,2));
% hold on
% plot(x,y,'b*');
% k = boundary(x,y);
% plot(x(k), y(k));

%videoPlayer  = vision.VideoPlayer('Position',...
%    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]]);

oldPoints = points;
i = 1;

% Get dimensions of points
[x,y] = getPointDims(points);

% Write to feature vector
featureVector(1,i) = x;
featureVector(2,i) = y;

while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);
    
    if(mod(i,refreshRate) == 0)
        if(mod(i,refreshRate*2) == 0)
            % redetect mouth region
            [bbox, bbimg, faces, bbfaces] = detectFaceParts(detector,videoFrame,2);
            bboxMouth       = bbox(:,13:16);
        else
            bboxMouth = [bboxPoints(1,1), bboxPoints(2,2), ...
            bboxPoints(2,1)-bboxPoints(1,1), ...
            bboxPoints(3,2)-bboxPoints(2,2)];
        end
        % hsv colorspace
        videoFrameF = rgb2hsv(videoFrame);
        % increase saturation 20% 
        videoFrameF(:, :, 2) = videoFrameF(:, :, 2) * saturation;
        % reestimate points
        points = detectHarrisFeatures(rgb2gray(videoFrameF), ...
            'ROI', bboxMouth, 'MinQuality', minQuality);
        points = points.selectStrongest(dataPoints);
        points = points.Location;
        points = removeCornerPoints(points, cornerSize);
        
        % Get dimensions of points
        [x,y] = getPointDims(points);
        
        % Write to feature vector
        featureVector(1,i+1) = x;
        featureVector(2,i+1) = y;
        
        oldPoints = points;
        setPoints(pointTracker, oldPoints);
        bboxPoints = bbox2points(bboxMouth);
        
        % Display tracked points
%         videoFrame = insertMarker(videoFrame, oldPoints, '+', ...
%             'Color', 'green');

    else
        % Track the points. Note that some points may be lost.
        [points, isFound] = step(pointTracker, videoFrame);
        visiblePoints = points(isFound, :);
        oldInliers = oldPoints(isFound, :);
        if size(visiblePoints, 1) >= 2 % need at least 2 points

            % Estimate the geometric transformation between the old points
            % and the new points and eliminate outliers
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 5);

            % Apply the transformation to the bounding box points
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Insert a bounding box around the object being tracked
%             bboxPolygon = reshape(bboxPoints', 1, []);
%             videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
%                 'LineWidth', 2);

            % Display tracked points
%             videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
%                 'Color', 'white');
            
            % Get dimensions of points
            [x,y] = getPointDims(visiblePoints);
            
            % Write to feature vector
            featureVector(1,i+1) = x;
            featureVector(2,i+1) = y;
            
            % Reset the points
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end
    end        
    
    % Display the annotated video frame using the video player object
%     step(videoPlayer, videoFrame);
    i = i+1;
end

disp('Writing file...')
writehtk_lite(strcat(filename(1:end-4),'.mfcc'), ...
    featureVector, 1/fps, 9);

% Clean up
release(videoFileReader);
release(videoPlayer);
release(pointTracker);

end

% while ~isDone(videoFileReader)
%     videoFrame = videoFileReader(); 
%     % Track the points. Note that some points may be lost.
%     [points, isFound] = step(pointTracker, videoFrame);
%     visiblePoints = points(isFound, :);
%     oldInliers = oldPoints(isFound, :);
%     
%     if size(visiblePoints, 1) >= 2 % need at least 2 points
%         if(mod(i,refreshRate) == 0)
%             % Reestimate points
%             [bbox, ~, ~, ~] = detectFaceParts(detector,videoFrame,2);
%             bboxMouth       = bbox(:,13:16);
%             release(pointTracker);
%             % hsv colorspace
%             videoFrameF = rgb2hsv(videoFrame);
%             % increase saturation 20% 
%             videoFrameF(:, :, 2) = videoFrameF(:, :, 2) * saturation;
%             points = detectHarrisFeatures(rgb2gray(videoFrameF), ...
%                 'ROI', bboxMouth, 'MinQuality', minQuality);
%             points = points.selectStrongest(dataPoints);
%             points = points.Location;
%             initialize(pointTracker, points, videoFrame);   
%         end 
%         
%         % Estimate the geometric transformation between the old points
%         % and the new points and eliminate outliers
%         [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
%             oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
% 
%         % Apply the transformation to the bounding box points
%         bboxPoints = transformPointsForward(xform, bboxPoints);
% 
%         % Insert a bounding box around the object being tracked
%         bboxPolygon = reshape(bboxPoints', 1, []);
%         videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
%             'LineWidth', 2);
% 
%         % Display tracked points
%         videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
%             'Color', 'white');
% 
%         % Reset the points
%         oldPoints = visiblePoints;
%         setPoints(pointTracker, oldPoints);
%         [points,validity] = pointTracker(videoFrame);
%         out = insertMarker(videoFrame,points(validity, :),'+');
%         videoPlayer(out);
%         i = i+1;
%     end
% end



function cleanPoints = removeCornerPoints(points, amount)
    % size = 0.0 - 1.0 size of exclusion corners
    % find min and max
    min_x = min(points(:,1));
    max_x = max(points(:,1));
    min_y = min(points(:,2));
    max_y = max(points(:,2));
    x = max_x - min_x;
    y = max_y - min_y;
    xThresh = (x * amount);
    yThresh = (y * amount);
    [rows, ~] = size(points);
    for i = 1:rows
        inPoints = points(i,:);
        point_x = inPoints(1); 
        point_y = inPoints(2);
        if((point_x > (min_x + xThresh)) && point_x < (max_x - xThresh) ...
                && point_y > (min_y + yThresh)  && point_y < (max_y - yThresh))
            cleanPoints(i,:) = inPoints;
        end
    end
    % remove empty points
    cleanPoints = cleanPoints(any(cleanPoints,2),:);
end

function [x, y] = getPointDims(points)
    min_x = min(points(:,1));
    max_x = max(points(:,1));
    min_y = min(points(:,2));
    max_y = max(points(:,2));
    x = max_x - min_x;
    y = max_y - min_y;
end

function writehtk_lite( filename, features, sampPeriod, parmKind )
% WRITEHTK_LITE Simple routine for writing HTK feature files.
%
%   WRITEHTK_LITE( FILENAME, FEATURES, SAMPPERIOD, PARMKIND )
%   writes FEATURES to HTK [1] feature file specified by FILENAME,
%   with sample period (s) defined in SAMPPERIOD and parameter kind
%   in PARAMKIND. Note that this function provides a trivial 
%   implementation with limited functionality. For fully featured 
%   support of HTK I/O refer for example to the VOICEBOX toolbox [2].
%   
%   Inputs
%           FILENAME is a filename as string for a HTK feature file
%
%           FEATURES is a feature matrix with feature vectors 
%           as rows and feature dimensions as columns
%
%           SAMPPERIOD is a sample period (s)
%
%           PARMKIND is a code indicating a sample kind
%           (see Sec. 5.10.1 of [1], pp. 80-81)
%
%   Example
%           % write features to sp10_htk.mfc file with sample period 
%           % set to 10 ms and feature type specified as MFCC_0
%           readhtk_lite( 'sp10_htk.mfc', features, 10E-3, 6+8192 );
%
%   References
%
%           [1] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., 
%               Liu, X., Moore, G., Odell, J., Ollason, D., Povey, D., 
%               Valtchev, V., Woodland, P., 2006. The HTK Book (for HTK 
%               Version 3.4.1). Engineering Department, Cambridge University.
%               (see also: http://htk.eng.cam.ac.uk)
%
%           [2] VOICEBOX: MATLAB toolbox for speech processing by Mike Brookes
%               url: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%   Author: Kamil Wojcicki, September 2011
    mfcfile = fopen( filename, 'w', 'b' );
    [ nSamples, sampSize ] = size( features );
    
    fwrite( mfcfile, nSamples, 'int32' );
    fwrite( mfcfile, sampPeriod*1E7, 'int32' );
    fwrite( mfcfile, 4*sampSize, 'int16' );
    fwrite( mfcfile, parmKind, 'int16' );
    
    count = fwrite( mfcfile, features.', 'float' );
    fclose( mfcfile );
    if count~=nSamples*sampSize
        error( sprintf('write_HTK_file: count!=nSamples*sampSize (%i!=%i), filename: %s', count, nSamples*sampSize, filename)); 
    end
end
% EOF
