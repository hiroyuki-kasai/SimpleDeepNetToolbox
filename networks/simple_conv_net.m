classdef simple_conv_net < handle
% This file defines simple convolutional class.
%
%
%   Network structure: 
%       Conv - relu - pool - affine - relu - affine - softmax.
%
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 04, 2018
%
% Change log: 
%
%   Nov. 07, 2018 (H.Kasai)
%       Moved optimizer to trainer class.
%       Added get_params method.
%       Added calculate_grads method.
%
%   Nov. 09, 2018 (H.Kasai)
%       Moved data properties from nn_trainer class to this class.
%       Moved params and grads from nn_trainer class to this class.
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.
% Major modifications have been made for MATLAB implementation and  
% its efficient implementation.


    properties
        
        name; 
        
        % layers
        layer_manager; 
        
        % size        
        hidden_size_list;
        output_size;             
        
        % parameters (W, b)
        params;
        
        % grads
        grads;        
        
        % data
        x_train;
        y_train;
        x_test;
        y_test; 
        train_size;        
        dataset_dim;             
        
        % else
        use_num_grad;
        
    end
    
    methods
        function obj = simple_conv_net(x_train, y_train, x_test, y_test, input_dim, conv_param, hidden_size, output_size, weight_init_std, use_num_grad)       
            

            obj.name = 'simple_conv_net';
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;  
            obj.train_size = size(x_train, 1);
            obj.dataset_dim = ndims(x_train);              
            
            obj.use_num_grad = use_num_grad;
            
            filter_num = conv_param.filter_num;
            filter_size = conv_param.filter_size;
            filter_pad = conv_param.pad;
            filter_stride = conv_param.stride;
            input_size = input_dim(2);
            conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1;
            pool_output_size = fix(filter_num * (conv_output_size/2) * (conv_output_size/2));           
            

            
            %% initialize weights
            obj.params = containers.Map('KeyType', 'char','ValueType', 'any');
            obj.params('W1') = weight_init_std * randn(filter_num, input_dim(1), filter_size, filter_size);
            obj.params('b1') = zeros(1, filter_num);
            obj.params('W2') = weight_init_std * randn(pool_output_size, hidden_size);
            obj.params('b2') = zeros(1, hidden_size);
            obj.params('W3') = weight_init_std * randn(hidden_size, output_size);
            obj.params('b3') = zeros(1, output_size);
            
            obj.grads = containers.Map('KeyType', 'char','ValueType', 'any');
            obj.grads('W1') = [];
            obj.grads('b1') = [];
            obj.grads('W2') = [];
            obj.grads('b2') = [];
            obj.grads('W3') = [];
            obj.grads('b3') = [];


            
            %% generate layers
            obj.layer_manager = layer_manager();
            % generate affine layers
            obj.layer_manager = obj.layer_manager.add_layer('convolution', obj.params('W1'), obj.params('b1'), conv_param.stride, conv_param.pad);
            % generate activation layers
            obj.layer_manager = obj.layer_manager.add_layer('relu');
            % generate pooling layers
            obj.layer_manager = obj.layer_manager.add_layer('pooling', 2, 2, 2);        
            % generate affine layers
            obj.layer_manager = obj.layer_manager.add_layer('affine', obj.params('W2'), obj.params('b2'));
            % generate activation layers
            obj.layer_manager = obj.layer_manager.add_layer('relu'); 
            % generate affine layers
            obj.layer_manager = obj.layer_manager.add_layer('affine', obj.params('W3'), obj.params('b3'));  
            % generate softmax_with_loss layer
            obj.layer_manager = obj.layer_manager.add_last_layer('softmax');  

        end
        
        
        % get params        
        function params = get_params(obj)
            
            params = obj.params;
            
        end      
        
        
        % set params
        function obj = set_params(obj, params)
            
            obj.params = params;
            
            % deliver internal params to convolutional layer
            obj.layer_manager.conv_layers{1}.update_params(params('W1'), params('b1')); 
            % deliver internal params to all affine layers
            obj.layer_manager.aff_layers{1}.update_params(params('W2'), params('b2')); 
            obj.layer_manager.aff_layers{2}.update_params(params('W3'), params('b3')); 
            
        end        
        
      
        function f = loss(obj, varargin)
            
            if nargin < 2
                train_flag = false;
            else
                train_flag = varargin{1};
            end               
            
            f = loss_partial(obj, 1:obj.train_size, train_flag);
            
        end        
        
        
        function f = loss_partial(obj, indice, varargin)            
            
            if nargin < 3
                train_flag = false;
            else
                train_flag = varargin{1};
            end      
            
            if obj.dataset_dim == 2
                x_curr_batch = obj.x_train(indice,:);
                y_curr_batch = obj.y_train(indice,:);
            elseif obj.dataset_dim == 4
                x_curr_batch = obj.x_train(indice,:,:,:);
                y_curr_batch = obj.y_train(indice,:,:,:);   
            else
            end              
            
            y = obj.predict(x_curr_batch, train_flag);
            f = obj.layer_manager.last_layer.forward(y, y_curr_batch);
            
        end
                
        
        
        
        function y = predict(obj, x, varargin)
            
            if nargin < 3
                train_flag = false;
            else
                train_flag = varargin{1};
            end              
            
            for idx = 1 : obj.layer_manager.total_num
                if strcmpi(obj.layer_manager.layers{idx}.name, 'dropout') || strcmpi(obj.layer_manager.layers{idx}.name, 'batch_normalization')
                    x = obj.layer_manager.layers{idx}.forward(x, train_flag);
                else
                    x = obj.layer_manager.layers{idx}.forward(x);
                end
            end

            y = x;

        end
        
        
        
        function accuracy = accuracy(obj, varargin)
            
            if nargin < 2
                train_flag = false;
            else
                train_flag = varargin{1};
            end 
            
            if strcmp(train_flag, 'train')
                x = obj.x_train;
                t = obj.y_train;
            else
                x = obj.x_test;
                t = obj.y_test;                
            end               
            
            batch_size = size(x, 1);

            y = obj.predict(x, false);
            [~, y] = max(y, [], 2);
            [~, t] = max(t, [], 2);
            
            accuracy = sum(y == t)/batch_size;
            
        end
        
        
        
        

        %% calculate gradient
        function grads = calculate_grads(obj, indice)
            
            if obj.use_num_grad
                obj.grads = obj.numerical_gradient(indice);
            else
                obj.grads = obj.gradient(indice);
            end  
            
            grads = obj.grads;
        end     
        
        
        
        
        % 1. numerical gradient
        function grads = numerical_gradient(obj, indice)
            
            grads = [];
            for idx = 1 : 3
                grads.W{idx} = obj.calc_numerical_gradient(obj.params.W{idx}, 'W', idx, indice);
                grads.b{idx} = obj.calc_numerical_gradient(obj.params.b{idx}, 'b', idx, indice); 
            end    
            
            %fprintf('W:%.16e, b:%.16e\n', norm(grads.W{1}), norm(grads.b{1}));            

        end
        
        
        function grad = calc_numerical_gradient(obj, x, id, idx, indice)

            h = 1e-4;
            
            if obj.dataset_dim == 2
                x_curr_batch = obj.x_train(indice,:);
                y_curr_batch = obj.y_train(indice,:);
            elseif obj.dataset_dim == 4
                x_curr_batch = obj.x_train(indice,:,:,:);
                y_curr_batch = obj.y_train(indice,:,:,:);   
            else
            end              

            row = size(x, 1);
            col = size(x, 2);
            grad = zeros(row, col);

            for row_idx = 1:row
                for col_idx = 1:col
                    % store original value
                    tmp_val = x(row_idx, col_idx);
                    
                    % replace idx-th element with "vec_x(idx) + h" 
                    x(row_idx, col_idx) = tmp_val + h;
                    f_plus_h = obj.loss_for_numerical_grad_calc(x, id, idx, x_curr_batch, y_curr_batch); % f(x+h)

                    % replace idx-th element with "vec_x(idx) - h"
                    x(row_idx, col_idx) = tmp_val - h;
                    f_minus_h = obj.loss_for_numerical_grad_calc(x, id, idx, x_curr_batch, y_curr_batch);% f(x-h)

                    % calculate gradient
                    grad(row_idx, col_idx) = (f_plus_h - f_minus_h) / (2*h);
                    %fprintf('(%d,%d)=%.16e\n', row_idx, col_idx, grad(row_idx, col_idx));

                    % recover the original value
                    x(row_idx, col_idx) = tmp_val;
                end
            end            
        end        
        


        function f = loss_for_numerical_grad_calc(obj, w, id, idx, x_curr_batch, t_curr_batch)
            
            y = obj.predict_for_numerical_grad(w, id, idx, x_curr_batch);
            f = obj.last_layer.forward(y, t_curr_batch);
            
        end        
        
 
        
        function y = predict_for_numerical_grad(obj, w, id, idx, x_curr_batch)
            
            ori_params = obj.params;
            if strcmp(id, 'W')
                obj.params.W{idx} = w;
            elseif strcmp(id, 'b')
                obj.params.b{idx} = w;
            end
            
            % change
            % update internal params in each convolutional layer
            obj.layer_manager.conv_layers{1}.update_params(obj.params.W{1}, obj.params.b{1}); 
            % update internal params in each affine layer
            obj.layer_manager.aff_layers{1}.update_params(obj.params.W{2}, obj.params.b{2}); 
            obj.layer_manager.aff_layers{2}.update_params(obj.params.W{3}, obj.params.b{3});             

            % predict
            y = obj.predict(x_curr_batch); 

            % restore
            obj.params = ori_params;
            % update internal params in each convolutional layer
            obj.layer_manager.conv_layers{1}.update_params(obj.params.W{1}, obj.params.b{1}); 
            % update internal params in each affine layer
            obj.layer_manager.aff_layers{1}.update_params(obj.params.W{2}, obj.params.b{2}); 
            obj.layer_manager.aff_layers{2}.update_params(obj.params.W{3}, obj.params.b{3});                

        end         
        
        
        
        
        
        % 2. backprop gradient
        function grads = gradient(obj, indice)
            
            % calculate gradients
            
            % forward
            loss_partial(obj, indice, true);                 
            
            dout = 1;
            dout = obj.layer_manager.last_layer.backward(dout);
            
            for idx = obj.layer_manager.total_num : -1 : 1
                dout = obj.layer_manager.layers{idx}.backward(dout);
            end
            
            % calculate gradients
            obj.grads('W1') = obj.layer_manager.conv_layers{1}.dW;
            obj.grads('b1') = obj.layer_manager.conv_layers{1}.db;
            obj.grads('W2') = obj.layer_manager.aff_layers{1}.dW;
            obj.grads('b2') = obj.layer_manager.aff_layers{1}.db;
            obj.grads('W3') = obj.layer_manager.aff_layers{2}.dW;
            obj.grads('b3') = obj.layer_manager.aff_layers{2}.db; 
            
            grads = obj.grads;

        end
        
    end

end

