
function applyHisteq(inputFile, outputFile)
    % applyHisteq applies histogram equalization to an image.
    %
    %   inputFile  : full path to the input image (assumed grayscale)
    %   outputFile : full path where the output image should be saved

    % Read the image.
    img = imread(inputFile);

    % If the image is RGB, convert it to grayscale.
    if size(img,3) > 1
        img = rgb2gray(img);
    end

    % Apply histogram equalization.
    equalized = histeq(img);

    % Write the output image.
    imwrite(equalized, outputFile);
end
