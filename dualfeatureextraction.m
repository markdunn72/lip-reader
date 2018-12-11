function dualfeatureextraction(vFilename, aFilename)
% Config
minQuality  = 0.005;
cornerSize  = 0.12;
dataPoints  = 100;
saturation  = 1.8;
refreshRate = 80; 

% Create a detector object
detector = buildDetector(2,2);

% Read a video frame and run the face detector.
vidObj          = VideoReader(vFilename);
fps             = vidObj.FrameRate;
numFrames       = ceil(vidObj.FrameRate*vidObj.Duration);
videoFileReader = vision.VideoFileReader(vFilename);
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

% Init feature vector
featureVector = zeros(2, numFrames);

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
writehtk_lite(strcat(vFilename(1:end-4),'.mfcc'), ...
    featureVector, 1/fps, 9);

% Clean up
release(videoFileReader);
% release(videoPlayer);
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

function sigout = preemphasis(y)
% This function applies pre emphasis to
% a time-domain signal, returns signal
b = [1 -0.95];
sigout = filter(b,1,y);       % high-pass filter
end

function ps = spectral(frame,S,K)
% This function returns a 257 point 
% 1 sided power spectral density
% estimate of a column vector
dft = fft(frame,S); % S point DFT of frames
p = abs(dft).^2;    % periodogram of DFT
ps = p(1:K);        % 1 sided power spectrum
end

function ev = filterbank(ps,H,M,K)
% H = filterbank matrix
% M = number of filters
ev = zeros(M,1);         % energy vector ev
for filter = 1:M
    fi = H(filter,:).';  % get filter vector
    fiev = zeros(K,1);   % stores energy coefficients of ps for each filter
    for x = 1:K
        fiev(x) = (ps(x)*fi(x)); % calculate filter energy
    end
    fie = sum(fiev);     % take sum of coefficients
    ev(filter) = fie;    % store energy of each filter
end
end

function fv = cepstral(ev,MT)
% This function takes the discrete cosine transform
% of an energy vector and removes the pitch-related
% quefrencies with truncation
ev = dct(ev);  % take DCT of ev to get cepstral domain
fv = ev(1:MT); % truncate
end

function fve = loge(fv)
% This function returns the feature vector with
% log energy component
[N,~] = size(fv);
ec = (sum(fv))^2; % calculate energy component
lec = log(ec);    % take log of ec
fve = zeros(N+1,1);
for i = 1:N
    fve(i) = fv(i);
end
fve(N+1) = lec;   % append energy component
end

    
function [ H, f, c ] = trifbank( M, K, R, fs, h2w, w2h )
% TRIFBANK Triangular filterbank.
%
%   [H,F,C]=TRIFBANK(M,K,R,FS,H2W,W2H) returns matrix of M triangular filters 
%   (one per row), each K coefficients long along with a K coefficient long 
%   frequency vector F and M+2 coefficient long cutoff frequency vector C. 
%   The triangular filters are between limits given in R (Hz) and are 
%   uniformly spaced on a warped scale defined by forward (H2W) and backward 
%   (W2H) warping functions.
%
%   Inputs
%           M is the number of filters, i.e., number of rows of H
%
%           K is the length of frequency response of each filter 
%             i.e., number of columns of H
%
%           R is a two element vector that specifies frequency limits (Hz), 
%             i.e., R = [ low_frequency high_frequency ];
%
%           FS is the sampling frequency (Hz)
%
%           H2W is a Hertz scale to warped scale function handle
%
%           W2H is a wared scale to Hertz scale function handle
%
%   Outputs
%           H is a M by K triangular filterbank matrix (one filter per row)
%
%           F is a frequency vector (Hz) of 1xK dimension
%
%           C is a vector of filter cutoff frequencies (Hz), 
%             note that C(2:end) also represents filter center frequencies,
%             and the dimension of C is 1x(M+2)
%
%   Example
%           fs = 16000;               % sampling frequency (Hz)
%           nfft = 2^12;              % fft size (number of frequency bins)
%           K = nfft/2+1;             % length of each filter
%           M = 23;                   % number of filters
%
%           hz2mel = @(hz)(1127*log(1+hz/700)); % Hertz to mel warping function
%           mel2hz = @(mel)(700*exp(mel/1127)-700); % mel to Hertz warping function
%
%           % Design mel filterbank of M filters each K coefficients long,
%           % filters are uniformly spaced on the mel scale between 0 and Fs/2 Hz
%           [ H1, freq ] = trifbank( M, K, [0 fs/2], fs, hz2mel, mel2hz );
%
%           % Design mel filterbank of M filters each K coefficients long,
%           % filters are uniformly spaced on the mel scale between 300 and 3750 Hz
%           [ H2, freq ] = trifbank( M, K, [300 3750], fs, hz2mel, mel2hz );
%
%           % Design mel filterbank of 18 filters each K coefficients long, 
%           % filters are uniformly spaced on the Hertz scale between 4 and 6 kHz
%           [ H3, freq ] = trifbank( 18, K, [4 6]*1E3, fs, @(h)(h), @(h)(h) );
%
%            hfig = figure('Position', [25 100 800 600], 'PaperPositionMode', ...
%                              'auto', 'Visible', 'on', 'color', 'w'); hold on; 
%           subplot( 3,1,1 ); 
%           plot( freq, H1 );
%           xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 
%       
%           subplot( 3,1,2 );
%           plot( freq, H2 );
%           xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 
%       
%           subplot( 3,1,3 ); 
%           plot( freq, H3 );
%           xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 
%
%   Reference
%           [1] Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing: 
%               A guide to theory, algorithm, and system development. 
%               Prentice Hall, Upper Saddle River, NJ, USA (pp. 314-315).
%
%   Author
%           Kamil Wojcicki, June 2011

    if( nargin~= 6 ), help trifbank; return; end; % very lite input validation

    f_min = 0;          % filter coefficients start at this frequency (Hz)
    f_low = R(1);       % lower cutoff frequency (Hz) for the filterbank 
    f_high = R(2);      % upper cutoff frequency (Hz) for the filterbank 
    f_max = 0.5*fs;     % filter coefficients end at this frequency (Hz)
    f = linspace( f_min, f_max, K ); % frequency range (Hz), size 1xK

    % filter cutoff frequencies (Hz) for all filters, size 1x(M+2)
    c = w2h( h2w(f_low)+[0:M+1]*((h2w(f_high)-h2w(f_low))/(M+1)) );

    % implements Eq. (6.140) given in [1] (for a given set of warping functions)
    H = zeros( M, K );                  % zero otherwise
    for m = 1:M 
        k = f>=c(m)&f<=c(m+1);          % up-slope
        H(m,k) = 2*(f(k)-c(m)) / ((c(m+2)-c(m))*(c(m+1)-c(m)));
        k = f>=c(m+1)&f<=c(m+2);        % down-slope
        H(m,k) = 2*(c(m+2)-f(k)) / ((c(m+2)-c(m))*(c(m+2)-c(m+1)));
    end

    % H = H./repmat(max(H,[],2),1,K);  % normalize to unit height
    % H = H./repmat(trapz(f,H,2),1,K); % normalize to unit area (inherently done)
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