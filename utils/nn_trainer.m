classdef nn_trainer < handle
% This file defines neural-network trainer class
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 02, 2018
%
% Change log: 
%
%   Nov. 07, 2018 (H.Kasai)
%       Moved optimizer from network class to this class.
%
%   Nov. 09, 2018 (H.Kasai)
%       Moved data properties to network class.
%       Moved params and grads to network class.
%
%
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.
% However, major modifications have been made for MATLAB implementation and  
% its efficient implementation.    

    
    properties

        name;
        
        network;
        optimizer; 
        options;
        
        train_size;
        iter_num_per_epoch;
        iter_per_epoch;
        current_epoch;
        max_iter;
        info;
        rand_index;
        
    end
    
    methods
        function obj = nn_trainer(network, varargin)  
            
            if nargin < 2
                in_options = [];
            else
                in_options = varargin{1};
            end              
             
            obj.name = 'nn_trainer';               
            obj.network = network;
            obj.train_size = obj.network.train_size;

            
            % set local options 
            local_options = [];

            % merge options
            obj.options = mergeOptions(get_default_trainer_options(), local_options);   
            obj.options = mergeOptions(obj.options, in_options);  
            
            
            obj.iter_num_per_epoch = max(fix(obj.train_size / obj.options.batch_size), 1);
            obj.max_iter = obj.options.max_epoch * obj.iter_num_per_epoch;
            
            
            % generate optimizer
            params = obj.network.get_params();
            obj.optimizer = stochastic_optimizer([], params, obj.options.opt_alg, obj.options.step_init, []);              


        end
        
        
        function [] = train_step(obj, iter)
            
            % process every epoch
            if mod(iter, obj.iter_num_per_epoch) == 1
                if obj.options.permute_on
                    obj.rand_index = randperm(obj.train_size);
                else
                    obj.rand_index = 1:obj.train_size;
                end 

                obj.iter_per_epoch = 1;
                obj.current_epoch = obj.current_epoch + 1;
            else
                obj.iter_per_epoch = obj.iter_per_epoch + 1;
            end

            
            % determine sampe indice
            start_mask = (obj.iter_per_epoch-1)*obj.options.batch_size + 1;
            end_mask = obj.iter_per_epoch * obj.options.batch_size;
            indice = obj.rand_index(start_mask:end_mask);    

            % calculate gradient
            grads = obj.network.calculate_grads(indice);
            
            % get/update/set params from network
            params = obj.network.get_params();            
            params = obj.optimizer.update(params, grads, obj.options.step_init, []);
            obj.network.set_params(params);
            
            if mod(iter, obj.iter_num_per_epoch) == 0
                
                % store infos
                obj.info.epoch = [obj.info.epoch obj.current_epoch];

                % calculate loss
                loss = obj.network.loss();
                obj.info.cost = [obj.info.cost loss];

                % calcualte accuracy
                train_acc = obj.network.accuracy('train');
                test_acc = obj.network.accuracy('test');
                obj.info.train_acc = [obj.info.train_acc train_acc];
                obj.info.test_acc = [obj.info.test_acc test_acc];
                
                % display
                if obj.options.verbose > 0
                    if obj.options.verbose > 1
                        fprintf('# Epoch: %03d (iter:%05d): cost = %.10e, ', obj.current_epoch, iter, loss); 
                        fprintf('accuracy (train, test) = (%5.4f, %5.4f)\n', train_acc, test_acc);
                    else
                        fprintf('.');
                    end
                end
            end  

        end
        
        
        function info = train(obj)
            
            % initialize
            obj.current_epoch = 0;
            obj.info = [];
            obj.rand_index = [];               
            
            % store first infos
            obj.info.epoch = 0;

            % calculate loss
            loss = obj.network.loss();
            obj.info.cost = loss;

            % calcualte accuracy
            train_acc = obj.network.accuracy('train');
            test_acc = obj.network.accuracy('test');
            obj.info.train_acc = train_acc;
            obj.info.test_acc = test_acc; 
            
            
            
            % display
            if obj.options.verbose > 0 
                fprintf('### start %s (opt:%s, max_epoch:%d)\n', obj.network.name, obj.options.opt_alg, obj.options.max_epoch);
                if obj.options.verbose > 1
                    fprintf('# Epoch: 000 (iter:00000): cost = %.10e, ', loss);
                    fprintf('accuracy (train, test) = (%5.4f, %5.4f)\n', train_acc, test_acc);            
                end
            end


            % main loop
            for iter = 1 : obj.max_iter
                obj.train_step(iter);
            end
            
            
            % display
            if obj.options.verbose > 0 
                fprintf('\n### end\n');
            end            
            
            info = obj.info;
        end
        

    end
end
