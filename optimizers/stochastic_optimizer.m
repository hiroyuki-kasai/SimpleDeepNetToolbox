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
%   Nov. 09, 2018 (H.Kasai)
%       Added SARAH and SAG/SAGA.
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
        update_routine;
        preprocess;
        disp_name;
        options;
        sub_mode;
        
        % Momentum
        momentum;
        v;
        
        % AdaGrad
        h;
        
        % SVRG
        full_grads;
        params0;
        
        % SARAH
        prev_params;
        recur_grads;
        sarah_plus;
        sarah_gamma;
        norm_v0;
        
        % SAG/SAGA
        grads_array;
        grads_ave;

    end
    
    methods
        
        function obj = stochastic_optimizer(problem, params, alg, in_options) 
            
            obj.name = 'stochastic_optimizer';
            obj.problem = problem;
            obj.param_keys = keys(params);
            obj.param_num = numel(obj.param_keys);
            obj.algorithm = alg;
            
            
            % set local options 
            local_options = [];

            % merge options
            options = mergeOptions(get_default_sto_opt_options(), local_options);   
            options = mergeOptions(options, in_options);              
            

            if strcmp(alg, 'SGD')
                
                obj.update_routine = @obj.sgd_update;
                obj.disp_name = 'SGD';
                
            elseif strcmp(alg, 'Momentum')
                
                obj.update_routine = @obj.momentum_update;
                obj.disp_name = 'Momentum';
                
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
                
                obj.update_routine = @obj.adagrad_update;
                obj.disp_name = 'AdaGrad';
                
                obj.h = containers.Map('KeyType','char','ValueType','any');
                for i = 1 : obj.param_num
                    key = obj.param_keys{i};
                    obj.h(key) = zeros(size(params(key)));
                end 
                
            elseif strcmp(alg, 'SVRG')
                
                obj.update_routine = @obj.svrg_update;
                obj.disp_name = 'SVRG'; 
                
            
            elseif strcmp(alg, 'SARAH')
                
                obj.update_routine = @obj.sarah_update;
                
                if ~isfield(options, 'sarah_gamma')
                    
                    obj.disp_name = 'SARAH';
                    
                else
                    obj.sarah_gamma = options.sarah_gamma;
                    
                    if obj.sarah_gamma < 1
                        obj.sarah_plus = 1;
                        obj.disp_name = 'SARAH+'; 
                    else
                        obj.sarah_plus = 0;
                        obj.disp_name = 'SARAH';                         
                    end
                end
                
            elseif strcmp(alg, 'SAG') || strcmp(alg, 'SAGA')
                
                obj.update_routine = @obj.sag_update;
                
                if strcmp(alg, 'SAG') 
                    obj.disp_name = 'SAG'; 
                    obj.sub_mode = 'sag';
                else
                    obj.disp_name = 'SAGA'; 
                    obj.sub_mode = 'saga';
                end
                                
                % prepare an array of gradients, and a valiable of average gradient
                obj.grads_array = containers.Map('KeyType','char','ValueType','any');
                obj.grads_ave = containers.Map('KeyType','char','ValueType','any');
                for i = 1 : obj.param_num
                    key = obj.param_keys{i};
                    obj.grads_array(key) = zeros(size(params(key),1), options.num_of_bachces);
                    obj.grads_ave(key) = mean(obj.grads_array(key), 2);
                end   

                
            end 
            
            obj.options = options;
        end
        
        
        
        % main update
        function [params_out, grads, step, calc_cnt, stop_flag] = update(obj, params_in, varargin)
            
            calc_cnt = 0;   
            
            % calculate gradient
            [grads, cnt] = obj.problem.calculate_grads(params_in, varargin{1});
            calc_cnt = calc_cnt + cnt;          
            
            % calculate stepsize
            step = obj.options.stepsizefun(varargin{3}, obj.options);            
            
            % update iterate
            [params_out, grads, cnt, stop_flag] = obj.update_routine(params_in, grads, step, varargin{1}, varargin{2}, varargin{3});
            calc_cnt = calc_cnt + cnt; 
            
        end
        
        
        
        
        
        
        
        
        
        
        % SGD
        function [params_out, grads, calc_cnt, stop_flag]  = sgd_update(obj, params_in, grads, step, varargin)
            
            stop_flag = 0;              
            calc_cnt = 0;      

            % calculate w = w - step * grad
            params_out = obj.problem.lincomb_vecvec(1, params_in, -1*step, grads);
            
        end        
        
        
        % Momentum
        function [params_out, grads, calc_cnt, stop_flag]  = momentum_update(obj, params_in, grads, step, varargin)

            stop_flag = 0;              
            calc_cnt = 0;
            
            % calculate v = v - step * grads;
            obj.v = obj.problem.lincomb_vecvec(obj.momentum, obj.v, -1*step, grads);
            % calculate w = w + obj.v
            params_out = obj.problem.lincomb_vecvec(1, params_in, 1, obj.v);            
            
        end
        
        
        % Adagrad
        function [params_out, grads, calc_cnt, stop_flag] = adagrad_update(obj, params_in, grads, step, varargin)
            
            stop_flag = 0;              
            calc_cnt = 0;
            
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
        function [params_out, modified_grad, calc_cnt, stop_flag] = svrg_update(obj, params_in, grads, step, varargin)
            
            stop_flag = 0;               
            calc_cnt = 0;
            
            if varargin{2} == 1
                
                % store the loop edge point
                obj.params0 = params_in;

                % calculate full gradient at params_in (outer loop edge point)
                [obj.full_grads, cnt] = obj.problem.calculate_grads(params_in, 1:obj.problem.samples);
                calc_cnt = calc_cnt + cnt;                  
                
            end            

            % calculate gradient at params0 (outer loop edge point)
            obj.problem.set_params(obj.params0);    % tell params0 to network 
            [grads0, cnt] = obj.problem.calculate_grads(obj.params0, varargin{1});
            obj.problem.set_params(params_in);      % restore current params (params_in) to network 
            calc_cnt = calc_cnt + cnt;            
            

            % calculate diff_grad = grad - grad0
            diff_grad = obj.problem.lincomb_vecvec(1, grads, -1, grads0);
            modified_grad = obj.problem.lincomb_vecvec(1, obj.full_grads, 1, diff_grad);
            % calculate w = w - step * modified_grad
            params_out = obj.problem.lincomb_vecvec(1, params_in, -1*step, modified_grad);              
            
        end  
        
        
        % SARAH
        function [params_out, recur_grads_out, calc_cnt, stop_flag] = sarah_update(obj, params_in, grads, step, varargin)
            
            stop_flag = 0;
            calc_cnt = 0;

            if varargin{2} == 1
                % store this point
                obj.prev_params = params_in;
                
                % calculate full gradient at params_in (outer loop edge point)
                [obj.recur_grads, cnt] = obj.problem.calculate_grads(params_in, 1:obj.problem.samples);
                calc_cnt = calc_cnt + cnt;

 
                if obj.sarah_plus
                    obj.norm_v0 = obj.problem.calculate_norm(obj.recur_grads);
                end                   
                
            end

            % calculate gradient at prev_params
            obj.problem.set_params(obj.prev_params);    % tell prev_params to network 
            [grads0, cnt] = obj.problem.calculate_grads(obj.prev_params, varargin{1});
            obj.problem.set_params(params_in);          % restore current params (params_in) to network 
            calc_cnt = calc_cnt + cnt;            
            
            % calculate diff_grad = grad - grad0
            diff_grad = obj.problem.lincomb_vecvec(1, grads, -1, grads0);
            recur_grads_out = obj.problem.lincomb_vecvec(1, obj.recur_grads, 1, diff_grad);
            % calculate w = w - step * recur_grads_out
            params_out = obj.problem.lincomb_vecvec(1, params_in, -1*step, recur_grads_out);            
            
            obj.recur_grads = recur_grads_out;
            
            % store this point
            obj.prev_params = params_in;
            
            if obj.sarah_plus
                curr_norm = obj.problem.calculate_norm(obj.recur_grads);
                
                if curr_norm <= obj.sarah_gamma * obj.norm_v0
                    stop_flag = 1;
                end
            end            
            
        end  
        
       
        
        % SAG/SAGA
        function [params_out, grads, calc_cnt, stop_flag] = sag_update(obj, params_in, grads, step, varargin)
            
            stop_flag = 0;
            calc_cnt = 0;

            params_out = containers.Map('KeyType','char','ValueType','any');
            grads_ave_new = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                grad = grads(key);
                grad_array = obj.grads_array(key);
                grad_ave = obj.grads_ave(key);

                if strcmp(obj.sub_mode, 'sag') 
                    grads_ave_new(key) = grad_ave + (grad - grad_array(:, varargin{2})) / obj.options.num_of_bachces;
                else
                    grads_ave_new(key) = grad_ave + (grad - grad_array(:, varargin{2}));
                end

                % replace with new grad
                grad_array(:, varargin{2}) = grad;
                obj.grads_array(key) = grad_array;
                params_out(key) = params_in(key) - step * grads_ave_new(key);
            end
            
            obj.grads_ave = grads_ave_new;
            
        end         
          
    end

end

