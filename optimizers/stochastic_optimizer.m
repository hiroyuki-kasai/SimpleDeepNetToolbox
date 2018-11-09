classdef stochastic_optimizer < handle
% This file defines stochastic optimization class.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 02, 2018
% Change log: 
%
%   Nov. 07, 2018 (H.Kasai)
%       Added SVRG.
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.
% Major modification have been made for MATLAB implementation and  
% its efficient implementation.


    properties
        name;
        problem;
        param_keys;
        param_num;
        algorithm;
        learning_rate;
        update;
        preprocess;
        disp_name;
        
        % Momentum
        momentum;
        v;
        
        % AdaGrad
        h;
        
        % SVRG
        full_grads;
        params0;

    end
    
    methods
        
        function obj = stochastic_optimizer(problem, params, alg, lrate, options) 
            
            obj.name = 'stochastic_optimizer';
            obj.problem = problem;
            obj.param_keys = keys(params);
            obj.param_num = numel(obj.param_keys);
            obj.algorithm = alg;
            obj.learning_rate = lrate;
            obj.preprocess = [];
          
            if strcmp(alg, 'SGD')
                
                obj.update = @obj.sgd_update;
                obj.disp_name = 'SGD';
                
            elseif strcmp(alg, 'Momuentum')
                
                obj.update = @obj.momentum_update;
                obj.disp_name = 'Momuentum';
                
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
                obj.disp_name = 'AdaGrad';
                
                obj.h = containers.Map('KeyType','char','ValueType','any');
                for i = 1 : obj.param_num
                    key = obj.param_keys{i};
                    obj.h(key) = zeros(size(params(key)));
                end 
                
            elseif strcmp(alg, 'SVRG')
                
                obj.update = @obj.svrg_update;
                obj.preprocess = @obj.svrg_preprocess;
                obj.disp_name = 'SVRG'; 
                
            end
        end
        
        
        
        
        
        % SGD
        function [params_out, grads, calc_cnt]  = sgd_update(obj, params_in, step, indice)
            
            calc_cnt = 0;      
            [grads, cnt] = obj.problem.calculate_grads(params_in, indice);
            calc_cnt = calc_cnt + cnt;

            
            params_out = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                param = params_in(key);
                grad = grads(key);
                params_out(key) = param - step * grad;
            end
            
        end        
        
        
        % Momentum
        function [params_out, grads, calc_cnt]  = momentum_update(obj, params_in, step, indice)

            calc_cnt = 0;
            [grads, cnt] = obj.problem.calculate_grads(params_in, indice);
            calc_cnt = calc_cnt + cnt;
            
            params_out = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                param = params_in(key);
                grad = grads(key);
                obj.v(key) = obj.momentum * obj.v(key) - step * grad;
                params_out(key) = param + obj.v(key);
            end
            
        end
        
        
        % Adagrad
        function [params_out, grads, calc_cnt] = adagrad_update(obj, params_in, step, indice)
            
            calc_cnt = 0;
            [grads, cnt] = obj.problem.calculate_grads(params_in, indice);
            calc_cnt = calc_cnt + cnt;

            
            params_out = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                param = params_in(key);
                grad = grads(key);
                obj.h(key) = obj.h(key) + grad .* grad;
                params_out(key) = param - step * grad ./ (sqrt(obj.h(key)) + 1e-7);
            end
            
        end 
        
        
        % SVRG
        function [params_out, grads_out, calc_cnt] = svrg_update(obj, params_in, step, indice)
            
            calc_cnt = 0;

            % calculate gradient at current params
            [grads, cnt] = obj.problem.calculate_grads(params_in, indice);
            calc_cnt = calc_cnt + cnt;

            % calculate gradient at params0 (outer loop edge point)
            obj.problem.set_params(obj.params0);    % tell params0 to network 
            [grads0, cnt] = obj.problem.calculate_grads(obj.params0, indice);
            obj.problem.set_params(params_in);      % restore current params (params_in) to network 
            calc_cnt = calc_cnt + cnt;            
            
            % calculate modified stochastic gradient
            params_out = containers.Map('KeyType','char','ValueType','any');
            grads_out = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                grad = grads(key);
                grad0 = grads0(key);
                full_grad = obj.full_grads(key);
                grads_out(key) = full_grad + grad - grad0;
                params_out(key) = params_in(key) - step * grads_out(key);
            end
            
        end  
        
        function [full_grads, calc_cnt] = svrg_preprocess(obj, params_in)
            
            calc_cnt = 0;
            
            % store the loop edge point
            obj.params0 = params_in;
            
            % calculate full gradient at params_in (outer loop edge point)
        	[obj.full_grads, cnt] = obj.problem.calculate_grads(params_in, 1:obj.problem.samples);
            calc_cnt = calc_cnt + cnt;
            
            full_grads = obj.full_grads;
            
        end         
          
    end

end

