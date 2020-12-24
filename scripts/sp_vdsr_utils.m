% This script constains helpful code to generate and reconstruct data to be
% used with SP-VDSR. 
% To get an interpolated image using the Lanczos filter, you can use
% Imagemagick with the following command:
% magick convert .\input.png -filter lanczos -resize 200% ./interpolated.png

% Loading the input pair
input_rgb = imread('input.png');
interpolated_rgb = imread('interpolated.png');

% Converting from RGB to YCbCr
input_ycbcr = rgb2ycbcr(input_rgb);
interpolated_ycbcr = rgb2ycbcr(interpolated_rgb);

% Saving the Y channel
imwrite(input_ycbcr(:,:,1), 'input_y.png');
imwrite(interpolated_ycbcr(:,:,1), 'interpolated_y.png');

% Use this input pair to feed the network.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loading the data generated from the neural network
load('predictions.mat');

% Turning the data into a single image in the case there's only one in the
% .mat file
pred_image = squeeze(pred_data);

% Reconstructing a new image
output_ycbcr(:,:,1) = pred_image;
output_ycbcr(:,:,2) = interpolated_ycbcr(:,:,2);
output_ycbcr(:,:,3) = interpolated_ycbcr(:,:,3);
output_rgb = ycbcr2rgb(output_ycbcr);
imwrite(output_rgb, 'output.png');