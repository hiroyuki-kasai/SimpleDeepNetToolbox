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
% Modified by H.Kasai on Oct. 08, 2018
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.
% Major modification have been made for MATLAB implementation and  
% its efficient implementation.


    properties
        name;  
        
        % size
        input_size;
        hidden_size;
        output_size;        
        
        % parameters (W, b)
        params;
        param_keys;        
        
        % layers
        layer_manager;
        
        % grads
        grads;        
        
        % optimizer
        optimizer;
        opt_algorithm;
        learning_rate;
        
        % else
        weight_init_std;
    end
    
    methods
        function obj = two_layer_net(input_size, hidden_size, output_size, opt_alg, lrate, varargin) 
            
            if nargin < 6
                obj.weight_init_std = 0.01;
            else
                obj.weight_init_std = varargin{1};
            end

            obj.name = 'two_layer_net';  
            obj.hidden_size = hidden_size;
            obj.output_size = output_size;
            obj.opt_algorithm = opt_alg;
            obj.learning_rate = lrate;
            

            
            %% initialize weights
            obj.params = containers.Map('KeyType','char','ValueType','any');
            obj.grads = containers.Map('KeyType','char','ValueType','any');
            
            obj.params('W1') = obj.weight_init_std * randn(input_size, hidden_size);
            obj.param_keys{1} = 'W1';
            obj.grads('W1') = [];

            obj.params('b1') = zeros(1, hidden_size);
            obj.param_keys{2} = 'b1';                
            obj.grads('b1') = [];     
            
            obj.params('W2') = obj.weight_init_std * randn(hidden_size, output_size);
            obj.param_keys{3} = 'W2';
            obj.grads('W2') = [];

            obj.params('b2') = zeros(1, output_size);
            obj.param_keys{4} = 'b2';                
            obj.grads('b2') = []; 
            
            
            
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
            

            
            %% generate optimizer
            obj.optimizer = stochastic_optimizer(obj.params, obj.opt_algorithm, obj.learning_rate, []);              

        end
        
        
        function f = loss(obj, x, t)
            
            y = obj.predict(x);

            f = obj.layer_manager.last_layer.forward(y, t);
            
        end
        
        
        
        function y = predict(obj, x)
            
            for idx = 1 : obj.layer_manager.total_num
                x = obj.layer_manager.layers{idx}.forward(x);
            end

            y = x;

        end 
        

        
        function accuracy = accuracy(obj, x, t)
            
            batch_size = size(x, 1);

            y = obj.predict(x);
            [~, y] = max(y, [], 2);
            [~, t] = max(t, [], 2);
            
            accuracy = sum(y == t)/batch_size;
            
        end
        
        
        
        
        
        %% numerical gradient
        function grads = numerical_gradient(obj, x_curr_batch, t_curr_batch)
            
            grads = [];

                grads('W1') = obj.calc_numerical_gradient(obj.params('W1'), 'W1', x_curr_batch, t_curr_batch);
                grads('b1') = obj.calc_numerical_gradient(obj.params('b1'), 'b1', x_curr_batch, t_curr_batch);
                grads('W2') = obj.calc_numerical_gradient(obj.params('W2'), 'W2', x_curr_batch, t_curr_batch);
                grads('b2') = obj.calc_numerical_gradient(obj.params('b2'), 'b2', x_curr_batch, t_curr_batch);                

        end
        
        function grad = calc_numerical_gradient(obj,  x, id, x_curr_batch, t_curr_batch)

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
                    %f_plus_h = f(x, id, x_curr_batch, t_curr_batch); % f(x+h)
                    f_plus_h = obj.loss_for_numerical_grad_calc(x, id, x_curr_batch, t_curr_batch);

                    % replace idx-th element with "vec_x(idx) - h"
                    x(row_idx, col_idx) = tmp_val - h;
                    %f_minus_h = f(x, id, x_curr_batch, t_curr_batch); % f(x+h)
                    f_minus_h = obj.loss_for_numerical_grad_calc(x, id, x_curr_batch, t_curr_batch);

                    % calculate gradient
                    grad(row_idx, col_idx) = (f_plus_h - f_minus_h) / (2*h);
                    %fprintf('(%d,%d)=%.16e\n', row_idx, col_idx, grad(row_idx, col_idx));

                    % recover the original value
                    x(row_idx, col_idx) = tmp_val;
                end
            end            
        end
        

        function f = loss_for_numerical_grad_calc(obj, w, id, x_curr_batch, t_curr_batch)
            
            y = obj.predict_for_numerical_grad(w, id, x_curr_batch);
            f = cross_entropy_error(y, t_curr_batch);
            
        end
        
        
        function y = predict_for_numerical_grad(obj, w, id, x_curr_batch)
            
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
        
        
        
        
        
        %% backprop gradient
        function grads = gradient(obj, x, t)
            
            % forward
            loss(obj, x, t);
            
            dout = 1;
            dout = obj.layer_manager.last_layer.backward(dout);
            
            for idx = obj.layer_manager.total_num : -1 : 1
                dout = obj.layer_manager.layers{idx}.backward(dout);
            end
            
            %

            obj.grads('W1') = obj.layer_manager.aff_layers{1}.dW;
            obj.grads('b1') = obj.layer_manager.aff_layers{1}.db;
            obj.grads('W2') = obj.layer_manager.aff_layers{2}.dW;
            obj.grads('b2') = obj.layer_manager.aff_layers{2}.db;           

            grads = obj.grads;
        end
        
        

        
        %% update
        function obj = update(obj, grads)
           
            % update
            obj.params = obj.optimizer.update(obj.params, grads);
            
            % update internal params in each affine layer
            obj.layer_manager.aff_layers{1}.update_params(obj.params('W1'), obj.params('b1'));
            obj.layer_manager.aff_layers{2}.update_params(obj.params('W2'), obj.params('b2'));
            
        end
        
    end

end

