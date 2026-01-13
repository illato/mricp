
function applyImadjust(inputFile, outputFile)
    % applyImadjust adjusts the intensity values of an image using imadjust.
    %
    %   inputFile  : full path to the input image (assumed grayscale)
    %   outputFile : full path where the output image should be saved

    % Read the image.
    img = imread(inputFile);

    % If the image is RGB, convert it to grayscale.
    if size(img,3) > 1
        img = rgb2gray(img);
    end

    % Adjust the image intensity values.
    adjusted = imadjust(img);

    % Write the output image.
    imwrite(adjusted, outputFile);
end
