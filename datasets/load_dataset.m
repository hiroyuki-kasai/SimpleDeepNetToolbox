function [x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, height, width] = load_dataset(dataset_name, dataset_dir, max_train_num, max_test_num, mat_format_flag)
% This file defines dataset loading function.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 06, 2018
% Modified by H.Kasai on Oct. 07, 2018


    switch lower(dataset_name)
        case {'mnist', 'usps', 'coil20', 'coil100', 'cifar-100', 'orl_face'}
            
            if strcmpi(dataset_name, 'mnist')
                file_name = sprintf('%s/MNIST.mat', dataset_dir);
            elseif strcmpi(dataset_name, 'usps')
                file_name = sprintf('%s/USPS.mat', dataset_dir); 
            elseif strcmpi(dataset_name, 'coil20')
                file_name = sprintf('%s/COIL20.mat', dataset_dir);                
            elseif strcmpi(dataset_name, 'coil100')
                file_name = sprintf('%s/COIL100.mat', dataset_dir); 
            elseif strcmpi(dataset_name, 'cifar-100')
                file_name = sprintf('%s/CIFAR-100_img_32x32.mat', dataset_dir);                  
            elseif strcmpi(dataset_name, 'orl_face')
                file_name = sprintf('%s/ORL_Face_img.mat', dataset_dir);
            end

            input_data = load(file_name);
            x_train = input_data.TrainSet.X';
            t_train_vec = input_data.TrainSet.y;
            x_test = input_data.TestSet.X';
            t_test_vec = input_data.TestSet.y;
            class_num = input_data.class_num;
            dimension = size(x_train, 2);
            height = input_data.height;
            width = input_data.width;
            
            
            % decide train_num and test_num
            if input_data.train_num < max_train_num
                train_num = input_data.train_num;
            else
                train_num = max_train_num;
            end
            
            if input_data.test_num < max_test_num
                test_num = input_data.test_num;
            else
                test_num = max_test_num;                
            end 
            
            % extract
            x_train   = x_train(1:train_num,:);
            t_train   = t_train_vec(:,1:train_num);
            x_test    = x_test(1:test_num,:);
            t_test    = t_test_vec(:,1:test_num);

            % convert vectorized label to matrix form
            t_train = convert_labelvec_to_mat(t_train, length(t_train), class_num);
            t_train = t_train';
            t_test = convert_labelvec_to_mat(t_test, length(t_test), class_num);
            t_test = t_test';  
            
            % convert vectorized pixels to original image maxrix form
            if mat_format_flag
                
                img_dim = [input_data.height, input_data.width];
                
                train_img = zeros(train_num, 1, img_dim(1), img_dim(2));
                for i = 1 : train_num
                    img_col = x_train(i,:);
                    train_img(i, 1, :, : ) = reshape(img_col, img_dim);
                end

                x_train = train_img;

                test_img = zeros(test_num, 1, img_dim(1), img_dim(2));
                for i = 1 : test_num
                    img_col = x_test(i,:);
                    test_img(i, 1, :, : ) = reshape(img_col, img_dim);
                end
                
                x_test = test_img;
            end
            

            clear train_img;
            clear test_img;

        otherwise
            warning('Unexpected dataset name. Not loaded.')
    end
    
    clear input_data;    

end

