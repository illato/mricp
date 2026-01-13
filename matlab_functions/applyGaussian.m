
function applyGaussian(inputFile, outputFile, sigma)
    % applyGaussian applies a Gaussian blur to an image.
    %
    %   inputFile  : full path to the input image (assumed grayscale)
    %   outputFile : full path where the output image should be saved
    %   sigma      : standard deviation of the Gaussian kernel
    
    % Read the image.
    img = imread(inputFile);

    % In case the image is RGB, convert it to grayscale.
    if size(img,3) > 1
        img = rgb2gray(img);
    end
        
    % Apply the Gaussian filter.
    blurred = imgaussfilt(img, sigma);
    
    % Write the output image.
    imwrite(blurred, outputFile);
end
