classdef two_layer_net < handle
% This file defines multi-layer neural network class.
%
%
%   Network structure: 
%       affine - activation - softmax.
%
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 02, 2018
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
        input_size;  
        hidden_size;
        output_size;             
        
        % parameters (W, b)
        params;
        
        % data
        x_train;
        y_train;
        x_test;
        y_test; 
        samples;        
        dataset_dim;
        
        % else
        weight_init_std;
        use_num_grad;
        
    end
    
    methods
        function obj = two_layer_net(x_train, y_train, x_test, y_test, input_size, hidden_size, output_size, varargin) 
            
            if nargin < 8
                % no network options
                obj.weight_init_std = 0.01;
                obj.use_num_grad = 0;
            else
                options = varargin{1};
                if ~isfield(options, 'weight_init_std')
                    obj.weight_init_std = 0.01;
                else
                    obj.weight_init_std = options.weight_init_std;
                end
                
                if ~isfield(options, 'use_num_grad')
                    obj.use_num_grad = 0;
                else
                    obj.use_num_grad = options.use_num_grad;
                end                

            end

            obj.name = 'two_layer_net';
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;            
            obj.hidden_size = hidden_size;
            obj.output_size = output_size;
            obj.samples = size(x_train, 1);
            obj.dataset_dim = ndims(x_train);            

            

            
            %% initialize weights
            obj.params = containers.Map('KeyType','char','ValueType','any');
            obj.params('W1') = obj.weight_init_std * randn(input_size, hidden_size);
            obj.params('b1') = zeros(1, hidden_size);
            obj.params('W2') = obj.weight_init_std * randn(hidden_size, output_size);
            obj.params('b2') = zeros(1, output_size);            
            
            
            %% generate layers
            obj.layer_manager = layer_manager();            
            
            % generate affine layers
            obj.layer_manager = obj.layer_manager.add_layer('affine', obj.params('W1'), obj.params('b1'));
            % generate activation layers
            obj.layer_manager = obj.layer_manager.add_layer('relu');            
            % generate affine layers
            obj.layer_manager = obj.layer_manager.add_layer('affine', obj.params('W2'), obj.params('b2'));
            % generate softmax_with_loss layer
            obj.layer_manager = obj.layer_manager.add_last_layer('softmax');
            
        end
        
        
        % get paramters
        function params = get_params(obj)
            
            params = obj.params;
            
        end
        
        
        % set paramters
        function obj = set_params(obj, params)
            
            obj.params = params;
           
            % update internal params in each affine layer
            obj.layer_manager.aff_layers{1}.update_params(params('W1'), params('b1'));
            obj.layer_manager.aff_layers{2}.update_params(params('W2'), params('b2'));
            
        end        
        
        
        
        function f = loss(obj)
            
            f = loss_partial(obj, 1:obj.samples);
            
        end
        
        
        function f = loss_partial(obj, indice)
            
            if obj.dataset_dim == 2
                x_curr_batch = obj.x_train(indice,:);
                y_curr_batch = obj.y_train(indice,:);
            elseif obj.dataset_dim == 4
                x_curr_batch = obj.x_train(indice,:,:,:);
                y_curr_batch = obj.y_train(indice,:,:,:);   
            else
            end              
            
            y = obj.predict(x_curr_batch);

            f = obj.layer_manager.last_layer.forward(y, y_curr_batch);
            
        end        
        
        
        
        function y = predict(obj, x)
            
            for idx = 1 : obj.layer_manager.total_num
                x = obj.layer_manager.layers{idx}.forward(x);
            end

            y = x;

        end 
        

        
        function accuracy = accuracy(obj, train_flag)
            
            if strcmp(train_flag, 'train')
                x = obj.x_train;
                t = obj.y_train;
            else
                x = obj.x_test;
                t = obj.y_test;                
            end
            
            batch_size = size(x, 1);

            y = obj.predict(x);
            [~, y] = max(y, [], 2);
            [~, t] = max(t, [], 2);
            
            accuracy = sum(y == t)/batch_size;
            
        end
        
        
        
        
        %% calculate gradient
        function [grads, calc_cnt] = calculate_grads(obj, ignore_me, indice)
            
            
            if obj.use_num_grad
                grads = obj.numerical_gradient(indice);
            else
                grads = obj.gradient(indice);
            end  
            
            calc_cnt = length(indice); 
        end
        
        
        
        % 1. numerical gradient
        function grads = numerical_gradient(obj, indice)

            grads = [];

            grads('W1') = obj.calc_numerical_gradient(obj.params('W1'), 'W1', indice);
            grads('b1') = obj.calc_numerical_gradient(obj.params('b1'), 'b1', indice);
            grads('W2') = obj.calc_numerical_gradient(obj.params('W2'), 'W2', indice);
            grads('b2') = obj.calc_numerical_gradient(obj.params('b2'), 'b2', indice);                

        end
        
        
        function grad = calc_numerical_gradient(obj, x, id, indice)

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
                    %f_plus_h = f(x, id, x_curr_batch, t_curr_batch); % f(x+h)
                    f_plus_h = obj.loss_for_numerical_grad_calc(x, id, x_curr_batch, y_curr_batch);

                    % replace idx-th element with "vec_x(idx) - h"
                    x(row_idx, col_idx) = tmp_val - h;
                    %f_minus_h = f(x, id, x_curr_batch, t_curr_batch); % f(x+h)
                    f_minus_h = obj.loss_for_numerical_grad_calc(x, id, x_curr_batch, y_curr_batch);

                    % calculate gradient
                    grad(row_idx, col_idx) = (f_plus_h - f_minus_h) / (2*h);
                    %fprintf('(%d,%d)=%.16e\n', row_idx, col_idx, grad(row_idx, col_idx));

                    % recover the original value
                    x(row_idx, col_idx) = tmp_val;
                end
            end            
        end
        

        function f = loss_for_numerical_grad_calc(obj, w, id, indice)
            
            if obj.dataset_dim == 2
                x_curr_batch = obj.x_train(indice,:);
                y_curr_batch = obj.y_train(indice,:);
            elseif obj.dataset_dim == 4
                x_curr_batch = obj.x_train(indice,:,:,:);
                y_curr_batch = obj.y_train(indice,:,:,:);   
            else
            end            
            
            y = obj.predict_for_numerical_grad(w, id, x_curr_batch);
            f = cross_entropy_error(y, y_curr_batch);
            
        end
        
        
        function y = predict_for_numerical_grad(obj, w, id, indice)
            
            if obj.dataset_dim == 2
                x_curr_batch = obj.x_train(indice,:);
            elseif obj.dataset_dim == 4
                x_curr_batch = obj.x_train(indice,:,:,:);
            else
            end                
            
            batch_size = size(x_curr_batch, 1);
            
            tmp_params = obj.params;
            if strcmp(id, 'W1')
                tmp_params.W{1} = w;
            elseif strcmp(id, 'b1')
                tmp_params.b{1} = w;
            elseif strcmp(id, 'W2')
                tmp_params.W{2} = w;
            elseif strcmp(id, 'b2')
                tmp_params.b{2} = w;           
            else
                
            end
            
            W1 = tmp_params.W{1};
            W2 = tmp_params.W{2};
            b1 = tmp_params.b{1};
            b2 = tmp_params.b{2};
            
            a1 = x_curr_batch * W1 + repmat(b1, batch_size, 1);
            z1 = calc_sigmoid(a1);
            
            a2 = z1 * W2 + b2;
            y = softmax(a2);

        end 
        
        
        
        % 2. backprop gradient
        function grads = gradient(obj, indice)
            
            % forward
            loss_partial(obj, indice);
            
            dout = 1;
            dout = obj.layer_manager.last_layer.backward(dout);
            
            for idx = obj.layer_manager.total_num : -1 : 1
                dout = obj.layer_manager.layers{idx}.backward(dout);
            end
            
            %
            grads = containers.Map('KeyType','char','ValueType','any');
            grads('W1') = obj.layer_manager.aff_layers{1}.dW;
            grads('b1') = obj.layer_manager.aff_layers{1}.db;
            grads('W2') = obj.layer_manager.aff_layers{2}.dW;
            grads('b2') = obj.layer_manager.aff_layers{2}.db; 
            
        end
        
    end

end

