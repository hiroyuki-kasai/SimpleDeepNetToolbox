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
%
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.
% However, major modifications have been made for MATLAB implementation and  
% its efficient implementation.    

    
    properties
        name;
        
        network;
        
        verbose;
        
        %
        x_train;
        t_train;
        x_test;
        t_test;
        dataset_dim;
        
        %
        train_size;
        max_epochs;
        batch_size;
        iter_num_per_epoch;
        iter_per_epoch;
        current_epoch;
        max_iter;
        
        %
        info;
        rand_index;
        
        %
        optimizer;
        opt_algorithm;
        learning_rate;
        
    end
    
    methods
        function obj = nn_trainer(network, x_train, t_train, x_test, t_test, opt_alg, lrate, ...
                 max_epochs, mini_batch_size, verbose)
             
             
            obj.name = 'nn_trainer';               
            obj.network = network;
            obj.verbose = verbose;
            obj.x_train = x_train;
            obj.t_train = t_train;
            obj.x_test = x_test;
            obj.t_test = t_test;
            obj.opt_algorithm = opt_alg;
            obj.learning_rate = lrate;            
            obj.max_epochs = max_epochs;
            obj.batch_size = mini_batch_size;

            obj.train_size = size(x_train, 1);
            obj.iter_num_per_epoch = max(fix(obj.train_size / mini_batch_size), 1);
            obj.max_iter = max_epochs * obj.iter_num_per_epoch;
            obj.current_epoch = 0;

            obj.info = [];
            obj.rand_index = [];
            
            obj.dataset_dim = ndims(x_train);
            
            % get parameters
            params = obj.network.get_params();            
            
            
            %% generate optimizer
            obj.optimizer = stochastic_optimizer([], params, obj.opt_algorithm, obj.learning_rate, []);              


        end
        
        
        function [] = train_step(obj, iter)
            
            if mod(iter, obj.iter_num_per_epoch) == 1
                if 1
                    obj.rand_index = randperm(obj.train_size);
                else
                    obj.rand_index = 1:obj.train_size;
                end 

                obj.iter_per_epoch = 1;
                obj.current_epoch = obj.current_epoch + 1;
            else
                obj.iter_per_epoch = obj.iter_per_epoch + 1;
            end

            start_mask = (obj.iter_per_epoch-1)*obj.batch_size + 1;
            end_mask = obj.iter_per_epoch * obj.batch_size;
            indice = obj.rand_index(start_mask:end_mask);    

            if obj.dataset_dim == 2
                x_curr_batch = obj.x_train(indice,:);
                t_curr_batch = obj.t_train(indice,:);
            elseif obj.dataset_dim == 4
                x_curr_batch = obj.x_train(indice,:,:,:);
                t_curr_batch = obj.t_train(indice,:,:,:);   
            else
            end
            
            
            % calculate gradient
            grads = obj.network.calculate_grads(x_curr_batch, t_curr_batch);
            
            % get params from network
            params = obj.network.get_params();            

            % update params
            params = obj.optimizer.update(params, grads, obj.learning_rate, []);
            
            % set params into network
            obj.network.set_params(params);
            
            if obj.verbose > 0
                if mod(iter, obj.iter_num_per_epoch) == 0
                    obj.info.epoch = [obj.info.epoch obj.current_epoch];

                    % calculate loss
                    loss = obj.network.loss(obj.x_train, obj.t_train);
                    fprintf('# Epoch: %03d (iter:%05d): cost = %.10e, ', obj.current_epoch, iter, loss);
                    obj.info.cost = [obj.info.cost loss];

                    % calcualte accuracy
                    train_acc = obj.network.accuracy(obj.x_train, obj.t_train);
                    test_acc = obj.network.accuracy(obj.x_test, obj.t_test);
                    fprintf('accuracy (train, test) = (%5.4f, %5.4f)\n', train_acc, test_acc);
                    obj.info.train_acc = [obj.info.train_acc train_acc];
                    obj.info.test_acc = [obj.info.test_acc test_acc];
                end  
            end

        end
        
        
        function info = train(obj)
            
            if obj.verbose > 0
                obj.info.epoch = 0;
                
                % calculate loss
                loss = obj.network.loss(obj.x_train, obj.t_train);
                fprintf('# Epoch: 000 (iter:00000): cost = %.10e, ', loss);
                obj.info.cost = loss;

                % calcualte accuracy
                train_acc = obj.network.accuracy(obj.x_train, obj.t_train);
                test_acc = obj.network.accuracy(obj.x_test, obj.t_test);
                fprintf('accuracy (train, test) = (%5.4f, %5.4f)\n', train_acc, test_acc);
                obj.info.train_acc = train_acc;
                obj.info.test_acc = test_acc;                
            end
            
            for iter = 1 : obj.max_iter
                obj.train_step(iter);
            end
            
            info = obj.info;
        end
        

    end
end
