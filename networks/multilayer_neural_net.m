classdef multilayer_neural_net < handle
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
% Modified by H.Kasai on Oct. 05, 2018
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.
% Major modification have been made for MATLAB implementation and  
% its efficient implementation.


    properties
        name; 
        
        % size
        input_size;
        hidden_size_list;
        output_size;        
        
        % parameters (W, b)
        params;
        param_keys;        
        
        % layers
        layer_manager;
        hidden_layer_num;
        
        % grads
        grads;        
        
        % optimizer
        optimizer;
        opt_algorithm;
        learning_rate;

        % else
        activation_type;
        weight_init_std_type;
        weight_decay_lambda;
        use_dropout;
        dropout_ratio;
        use_batchnorm;
    end
    
    methods
        function obj = multilayer_neural_net(input_size, hidden_size_list, output_size, ...
                act_type, w_init_std, w_decay_lambda, use_do, do_ratio, use_bn, opt_alg, lrate)       
            

            obj.name = 'multilayer_neural_net';  
            obj.input_size = input_size;
            obj.hidden_size_list = hidden_size_list;
            obj.output_size = output_size;
            obj.hidden_layer_num = length(hidden_size_list);
            
            obj.activation_type = act_type;
            obj.weight_init_std_type = w_init_std;
            obj.weight_decay_lambda = w_decay_lambda;
        
            obj.use_dropout = use_do;
            obj.dropout_ratio = do_ratio;
            obj.use_batchnorm = use_bn;
            
            obj.opt_algorithm = opt_alg;
            obj.learning_rate = lrate;            
            

            
            %% initialize weights
            obj.params = containers.Map('KeyType','char','ValueType','any');
            obj.grads = containers.Map('KeyType','char','ValueType','any');
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
                obj.grads(['W', num2str(idx)]) = [];
                
                obj.params(['b', num2str(idx)]) = zeros(1, all_size_list(idx+1));
                param_num = param_num + 1;
                obj.param_keys{param_num} = ['b', num2str(idx)];                
                obj.grads(['b', num2str(idx)]) = [];

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

            % generate final affine layer
            obj.layer_manager = obj.layer_manager.add_layer('affine', obj.params(['W', num2str(idx+1)]), obj.params(['b', num2str(idx+1)]));

            % generate softmax_with_loss layer
            obj.layer_manager = obj.layer_manager.add_last_layer('softmax');
            
            
            
            %% generate optimizer
            obj.optimizer = stochastic_optimizer(obj.params, obj.opt_algorithm, obj.learning_rate, []);                 
            
        end
        
        

        function f = loss(obj, x, t, varargin)
            
            if nargin < 4
                train_flg = false;
            else
                train_flg = varargin{1};
            end             
            
            y = obj.predict(x, train_flg);
            
            weight_decay = 0;
            for idx = 1 : obj.hidden_layer_num + 1
                W = obj.params(['W', num2str(idx)]);
                weight_decay = weight_decay + 0.5 * obj.weight_decay_lambda * sum(sum(W.^2));
            end

            f = obj.layer_manager.last_layer.forward(y, t) + weight_decay;
            
        end
        
        
        
        function y = predict(obj, x, varargin)
            
            if nargin < 3
                train_flg = false;
            else
                train_flg = varargin{1};
            end              
            
            for idx = 1 : obj.layer_manager.total_num
                if strcmpi(obj.layer_manager.layers{idx}.name, 'dropout') || strcmpi(obj.layer_manager.layers{idx}.name, 'batch_normalization')
                    x = obj.layer_manager.layers{idx}.forward(x, train_flg);
                else
                    x = obj.layer_manager.layers{idx}.forward(x);
                end
            end

            y = x;

        end
        
        
        
        function accuracy = accuracy(obj, x, t, varargin)
            
            if nargin < 4
                train_flg = false;
            else
                train_flg = varargin{1};
            end              
            
            batch_size = size(x, 1);

            y = obj.predict(x, train_flg);
            [~, y] = max(y, [], 2);
            [~, t] = max(t, [], 2);
            
            accuracy = sum(y == t)/batch_size;
            
        end
        
        
        
        
        
        %% numerical gradient
        function grads = numerical_gradient(obj, x_curr_batch, t_curr_batch)
            
            grads = [];
            for idx = 1 : obj.hidden_layer_num + 1
                grads.W{idx} = obj.calc_numerical_gradient(obj.params.W{idx}, 'W', idx, x_curr_batch, t_curr_batch);
                grads.b{idx} = obj.calc_numerical_gradient(obj.params.b{idx}, 'b', idx, x_curr_batch, t_curr_batch); 

                if obj.use_batchnorm && idx ~= obj.hidden_layer_num + 1
                    grads.gamma{idx} = obj.calc_numerical_gradient(obj.batchnorm_layers{idx}.gamma, 'gamma', idx, x_curr_batch, t_curr_batch);
                    grads.beta{idx} = obj.calc_numerical_gradient(obj.batchnorm_layers{idx}.beta, 'beta', idx, x_curr_batch, t_curr_batch);                    
                end

            end    
            
            fprintf('W:%.16e, b:%.16e\n', norm(grads.W{1}), norm(grads.b{1}));            

        end
        
        
        
        function grad = calc_numerical_gradient(obj, x, id, idx, x_curr_batch, t_curr_batch)

            h = 1e-4;

            row = size(x, 1);
            col = size(x, 2);
            grad = zeros(row, col);

            for row_idx = 1:row
                for col_idx = 1:col
                    % store original value
                    tmp_val = x(row_idx, col_idx);
                    
                    % replace idx-th element with "vec_x(idx) + h" 
                    x(row_idx, col_idx) = tmp_val + h;
                    f_plus_h = obj.loss_for_numerical_grad_calc(x, id, idx, x_curr_batch, t_curr_batch); % f(x+h)

                    % replace idx-th element with "vec_x(idx) - h"
                    x(row_idx, col_idx) = tmp_val - h;
                    f_minus_h = obj.loss_for_numerical_grad_calc(x, id, idx, x_curr_batch, t_curr_batch);% f(x-h)

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
        
        
        
        
        
        %% backprop gradient
        function grads = gradient(obj, x, t)
            
            % forward
            loss(obj, x, t, true);
            
            dout = 1;
            dout = obj.layer_manager.last_layer.backward(dout);
            
            for idx = obj.layer_manager.total_num : -1 : 1
                dout = obj.layer_manager.layers{idx}.backward(dout);
            end
            
            % calculate gradients
            for idx = 1 : obj.hidden_layer_num + 1
                obj.grads(['W', num2str(idx)]) = obj.layer_manager.aff_layers{idx}.dW + obj.weight_decay_lambda * obj.layer_manager.aff_layers{idx}.W;
                obj.grads(['b', num2str(idx)]) = obj.layer_manager.aff_layers{idx}.db;
                
                if obj.use_batchnorm && idx ~= obj.hidden_layer_num + 1
                    obj.grads(['gamma', num2str(idx)]) = obj.layer_manager.batchnorm_layers{idx}.dgamma;
                    obj.grads(['beta', num2str(idx)]) = obj.layer_manager.batchnorm_layers{idx}.dbeta;                    
                end
            end
            
            %fprintf('W:%.16e, b:%.16e\n', norm(grads.W{1}), norm(grads.b{1}));
            
            grads = obj.grads;
            
        end
        
        

        
        %% update
        function obj = update(obj, grads)
            
            % update
            obj.params = obj.optimizer.update(obj.params, grads);            
            
            for idx = 1 : obj.hidden_layer_num + 1

                % update internal params in each affine layer
                obj.layer_manager.aff_layers{idx}.update_params(obj.params(['W', num2str(idx)]), obj.params(['b', num2str(idx)]));
                
                if obj.use_batchnorm && idx ~= obj.hidden_layer_num + 1
                    
                    % update internal params in each batchnorm layer
                    obj.layer_manager.batchnorm_layers{idx}.update_params(obj.params(['gamma', num2str(idx)]), obj.params(['beta', num2str(idx)]));
                    
                end                
            end
            
        end
        
    end

end

