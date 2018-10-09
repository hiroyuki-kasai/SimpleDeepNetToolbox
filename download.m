% download.m 

close all;
clear;
clc;


%% dataset
fprintf('###### Dataset downlowd ######\n\n');
% local folder
local_folder_name = 'datasets_new';


if ~exist(local_folder_name, 'dir')   
    mkdir(local_folder_name);
end

% remote url, path, and filelist
site_url = 'http://www.kasailab.com/';
site_path = 'public/github/SimpleDeepNetToolbox/datasets/';
filename_array = {'USPS.mat', 'MNIST.mat', 'COIL20.mat', 'COIL100.mat', ...
    'CIFAR-100_img_32x32.mat', 'ORL_Face_img.mat'};
dataset_num = length(filename_array);

% download files
for i = 1 : dataset_num
    file_name = filename_array{i};
    filename_full = sprintf('%s%s%s', site_url, site_path, file_name);
    filename_full_local = sprintf('%s/%s', local_folder_name, file_name);
    
    fprintf('# Dataset downlaoding [%d/%d]\n', i, dataset_num);
    
    if ~exist(filename_full_local, 'file')  
        fprintf('  * file:\t "%s"\n', file_name);
        fprintf('  * from:\t "%s%s"\n', site_url, site_path); 
        fprintf('  * into:\t "%s"\n', filename_full_local); 
        fprintf('  * ........ ');
        websave(filename_full_local, filename_full);
        fprintf('done\n\n');
    else   
        fprintf('  * ........ %s already exists. Skip downloading.\n\n', file_name);
    end
end

fprintf('# %d dataset files have been successfully downloaded.\n\n\n', dataset_num);


