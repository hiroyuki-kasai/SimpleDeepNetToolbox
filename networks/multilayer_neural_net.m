classdef multilayer_neural_net < nn_layer_basis
% This file defines multi-layer neural network class.
%
%
%   Network structure: 
%       affine - (batch norm -) activation - (dropout -) 
%           ..... 
%       affine - (batch norm -) activation - (dropout -) softmax.
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
%       Moved params from nn_trainer class to this class.
%
%   Nov. 13, 2018 (H.Kasai)
%       Inherit from nn_layer_basis
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.
% Major modifications have been made for MATLAB implementation and  
% its efficient implementation.


    properties
        
    end
    
    methods
        function obj = multilayer_neural_net(x_train, y_train, x_test, y_test, input_size, hidden_size_list, output_size, ...
                act_type, w_init_std, w_decay_lambda, use_do, do_ratio, use_bn, use_num_grad)
            
            

            obj.name = 'multilayer_neural_net'; 
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;               
            obj.input_size = input_size;
            obj.hidden_size_list = hidden_size_list;
            obj.output_size = output_size;
            obj.hidden_layer_num = length(hidden_size_list);
            obj.samples = size(x_train, 1);
            obj.dataset_dim = ndims(x_train);               
            
            obj.activation_type = act_type;
            obj.weight_init_std_type = w_init_std;
            obj.weight_decay_lambda = w_decay_lambda;
        
            obj.use_dropout = use_do;
            obj.dropout_ratio = do_ratio;
            obj.use_batchnorm = use_bn;
            obj.use_num_grad = use_num_grad;
            

            
            %% initialize weights
            obj.params = containers.Map('KeyType','char','ValueType','any');
            all_size_list = [input_size hidden_size_list output_size];
            all_size_list_num = length(all_size_list);
            
            param_num = 0;
            for idx = 1 : all_size_list_num-1
                
                % set initial values of paramters
                if strcmpi(obj.weight_init_std_type, 'relu')
                    scale = sqrt(2.0 / all_size_list(idx));
                elseif strcmpi(obj.weight_init_std_type, 'sigmoid') || strcmpi(obj.weight_init_std_type, 'xavier')
                    scale = sqrt(1.0 / all_size_list(idx));
                else
                    scale = sqrt(2.0 / all_size_list(idx)); % user ReLU setting
                end
                obj.params(['W', num2str(idx)]) = scale * randn(all_size_list(idx), all_size_list(idx+1));
                param_num = param_num + 1;
                obj.param_keys{param_num} = ['W', num2str(idx)];
                
                obj.params(['b', num2str(idx)]) = zeros(1, all_size_list(idx+1));
                param_num = param_num + 1;
                obj.param_keys{param_num} = ['b', num2str(idx)];                

            end
            
            
            %% generate layers
            obj.layer_manager = layer_manager();
            for idx = 1 : obj.hidden_layer_num

                % generate affine layers
                obj.layer_manager = obj.layer_manager.add_layer('affine', obj.params(['W', num2str(idx)]), obj.params(['b', num2str(idx)]));

                % generate batch normalization layers
                if obj.use_batchnorm
                    obj.params(['gamma', num2str(idx)]) = ones(1, obj.hidden_size_list(idx));
                    param_num = param_num + 1;
                    obj.param_keys{param_num} = ['gamma', num2str(idx)]; 
                
                    obj.params(['beta', num2str(idx)]) = zeros(1, obj.hidden_size_list(idx));
                    param_num = param_num + 1;
                    obj.param_keys{param_num} = ['beta', num2str(idx)]; 
                    
                    obj.layer_manager = obj.layer_manager.add_layer('batchnorm', obj.params(['gamma', num2str(idx)]), obj.params(['beta', num2str(idx)]), []);
                end

                % generate activation layers
                obj.layer_manager = obj.layer_manager.add_layer(obj.activation_type);

                % generate droput layers
                if obj.use_dropout
                    obj.layer_manager = obj.layer_manager.add_layer('dropout', obj.dropout_ratio);
                end

            end
            
            obj.param_num = param_num;

            % generate final affine layer
            obj.layer_manager = obj.layer_manager.add_layer('affine', obj.params(['W', num2str(idx+1)]), obj.params(['b', num2str(idx+1)]));

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
            
            for idx = 1 : obj.hidden_layer_num + 1

                % update internal params in each affine layer
                obj.layer_manager.aff_layers{idx}.update_params(params(['W', num2str(idx)]), params(['b', num2str(idx)]));
                
                if obj.use_batchnorm && idx ~= obj.hidden_layer_num + 1
                    
                    % update internal params in each batchnorm layer
                    obj.layer_manager.batchnorm_layers{idx}.update_params(params(['gamma', num2str(idx)]), params(['beta', num2str(idx)]));
                    
                end                
            end
            
        end        
        
        

        function f = loss(obj, varargin)
            
            if nargin < 2
                train_flag = false;
            else
                train_flag = varargin{1};
            end               
            
            f = loss_partial(obj, 1:obj.samples, train_flag);
            
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
            
            weight_decay = 0;
            for idx = 1 : obj.hidden_layer_num + 1
                W = obj.params(['W', num2str(idx)]);
                weight_decay = weight_decay + 0.5 * obj.weight_decay_lambda * sum(sum(W.^2));
            end

            f = obj.layer_manager.last_layer.forward(y, y_curr_batch) + weight_decay;
            
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
            
            for idx = 1 : obj.hidden_layer_num + 1
                grads.W{idx} = obj.calc_numerical_gradient(obj.params.W{idx}, 'W', idx, indice);
                grads.b{idx} = obj.calc_numerical_gradient(obj.params.b{idx}, 'b', idx, indice); 

                if obj.use_batchnorm && idx ~= obj.hidden_layer_num + 1
                    grads.gamma{idx} = obj.calc_numerical_gradient(obj.batchnorm_layers{idx}.gamma, 'gamma', idx, indice);
                    grads.beta{idx} = obj.calc_numerical_gradient(obj.batchnorm_layers{idx}.beta, 'beta', idx, indice);                    
                end

            end    
            
            fprintf('W:%.16e, b:%.16e\n', norm(grads.W{1}), norm(grads.b{1}));            

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
            elseif strcmp(id, 'gamma')
                ori_gamma = obj.batchnorm_layers{idx}.gamma;
                obj.batchnorm_layers{idx}.gamma = w;                
            elseif strcmp(id, 'beta')
                ori_beta = obj.batchnorm_layers{idx}.beta;
                obj.batchnorm_layers{idx}.beta = w;
            end
            
            % change
            for idx = 1 : obj.hidden_layer_num + 1 % Should inform affine layers of this changes 
                obj.affine_layers{idx}.update_params(obj.params.W{idx}, obj.params.b{idx});
                
                if obj.use_batchnorm && idx ~= obj.hidden_layer_num + 1
                    if strcmp(id, 'gamma')
                        obj.batchnorm_layers{idx}.update_params(obj.batchnorm_layers{idx}.gamma, obj.batchnorm_layers{idx}.beta);                
                    elseif strcmp(id, 'beta')
                        obj.batchnorm_layers{idx}.update_params(obj.batchnorm_layers{idx}.gamma, obj.batchnorm_layers{idx}.beta);
                    end
                end
            end

            % predict
            y = obj.predict(x_curr_batch); 

            % restore
            obj.params = ori_params;
            for idx = 1 : obj.hidden_layer_num + 1 % Should inform affine layers of this changes
                obj.affine_layers{idx}.update_params(obj.params.W{idx}, obj.params.b{idx});
                
                if obj.use_batchnorm && idx ~= obj.hidden_layer_num + 1
                    if strcmp(id, 'gamma')
                        obj.batchnorm_layers{idx}.update_params(ori_gamma, obj.batchnorm_layers{idx}.beta);                
                    elseif strcmp(id, 'beta')
                        obj.batchnorm_layers{idx}.update_params(obj.batchnorm_layers{idx}.gamma, ori_beta); 
                    end
                end                
            end                

        end         
        
        
        
        
        
        % 2. backprop gradient
        function grads = gradient(obj, indice)
            
            % forward
            loss_partial(obj, indice, true);            
            
            dout = 1;
            dout = obj.layer_manager.last_layer.backward(dout);
            
            for idx = obj.layer_manager.total_num : -1 : 1
                dout = obj.layer_manager.layers{idx}.backward(dout);
            end
            
            % calculate gradients
            grads = containers.Map('KeyType','char','ValueType','any');
            for idx = 1 : obj.hidden_layer_num + 1
                grads(['W', num2str(idx)]) = obj.layer_manager.aff_layers{idx}.dW + obj.weight_decay_lambda * obj.layer_manager.aff_layers{idx}.W;
                grads(['b', num2str(idx)]) = obj.layer_manager.aff_layers{idx}.db;
                
                if obj.use_batchnorm && idx ~= obj.hidden_layer_num + 1
                    grads(['gamma', num2str(idx)]) = obj.layer_manager.batchnorm_layers{idx}.dgamma;
                    grads(['beta', num2str(idx)]) = obj.layer_manager.batchnorm_layers{idx}.dbeta;                    
                end
            end
            
        end
        
        
    end

end

