classdef stochastic_optimizer < handle
% This file defines stochastic optimization class.
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
        param_keys;
        param_num;
        algorithm;
        learning_rate;
        update;
        
        % Momentum
        momentum;
        v;
        
        % AdaGrad
        h;
    end
    
    methods
        
        function obj = stochastic_optimizer(params, alg, lrate, options) 
            
            obj.name = 'stochastic_optimizer';
            obj.param_keys = keys(params);
            obj.param_num = numel(obj.param_keys);
            obj.algorithm = alg;
            obj.learning_rate = lrate;
            
          
            if strcmp(alg, 'SGD')
                
                obj.update = @obj.sgd_update;
                
            elseif strcmp(alg, 'Momuentum')
                
                obj.update = @obj.momentum_update;
                
                if ~isfield(options, 'momentum')
                    obj.momentum = 0.9;
                else
                    obj.momentum = options.momentum;
                end
                
                obj.v = containers.Map('KeyType','char','ValueType','any');
                for i = 1 : obj.param_num
                    key = obj.param_keys{i};
                    obj.v(key) = zeros(size(params(key)));
                end                
                
            elseif strcmp(alg, 'AdaGrad')
                
                obj.update = @obj.adagrad_update;
                
                obj.h = containers.Map('KeyType','char','ValueType','any');
                for i = 1 : obj.param_num
                    key = obj.param_keys{i};
                    obj.h(key) = zeros(size(params(key)));
                end                  
            end
        end
        
        % SGD
        function params_out = sgd_update(obj, params_in, grads)
            
            params_out = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                param = params_in(key);
                grad = grads(key);
                params_out(key) = param - obj.learning_rate * grad;
            end
            
        end        
        
        
        % Momentum
        function params_out = momentum_update(obj, params_in, grads)

            params_out = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                param = params_in(key);
                grad = grads(key);
                obj.v(key) = obj.momentum * obj.v(key) - obj.learning_rate * grad;
                params_out(key) = param + obj.v(key);
            end
            
        end
        
        
        % Adagrad
        function params_out = adagrad_update(obj, params_in, grads)
            
          
            params_out = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                param = params_in(key);
                grad = grads(key);
                obj.h(key) = obj.h(key) + grad .* grad;
                params_out(key) = param - obj.learning_rate * grad ./ (sqrt(obj.h(key)) + 1e-7);
            end
            
        end          
          
    end

end

