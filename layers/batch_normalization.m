classdef batch_normalization < handle
% This file defines batch normalization layer class.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 02, 2018
% Modified by H.Kasai on Oct. 05, 2018
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.

    
    properties
        name;
        
        batch_size;
        
        gamma;
        beta;
        dgamma;
        dbeta; 
        
        momentum;
        input_shape;
        
        % 
        running_mean;
        running_var;
        
        %
        xc;
        xn;
        std;
        
    end
    
    methods
        
        function obj = batch_normalization(gamma, beta, options) 
            
            obj.name = 'batch_normalization';
            obj.input_shape = [];
            obj.gamma = gamma;
            obj.beta = beta;
            
            if isfield(options, 'momentum')
                obj.momentum = options.momentum;
            else
                obj.momentum = 0.9;
            end
            
            if isfield(options, 'running_mean')
                obj.running_mean = options.running_mean;
            else
                obj.running_mean = [];
            end
            
            if isfield(options, 'running_var')
                obj.running_var = options.running_var;
            else
                obj.running_var = [];
            end            
            
            obj.batch_size = [];
            obj.xc = [];
            obj.std = [];
            obj.dgamma = [];
            obj.dbeta = [];             

        end 
        
        function obj = update_params(obj, gamma_in, beta_in) 
            
            obj.gamma = gamma_in;
            obj.beta = beta_in;
        
        end             
        
        function out = forward(obj, x, varargin) 
            
            if 0
                out = x;
            else
                if nargin < 2
                    train_flag = true;
                else
                    train_flag = varargin{1};
                end  

                obj.input_shape = size(x);
                out = obj.forward_internal(x, train_flag);
                out = reshape(out, obj.input_shape);
            end
        end     
        
        function out = forward_internal(obj, x, train_flag)
            
            if isempty(obj.running_mean)
                [n, d] = size(x);
                obj.running_mean = zeros(1, d);
                obj.running_var = zeros(1, d);
            end
            
            if train_flag == 1
                mu = mean(x);
                xc = x - mu;
                var = mean(xc.^2);
                std = sqrt(var + 10e-7);
                xn = xc ./ std;
                
                obj.batch_size = size(x, 1);
                obj.xc = xc;
                obj.xn = xn;
                obj.std = std;
                obj.running_mean = obj.momentum * obj.running_mean + (1 - obj.momentum) * mu;
                obj.running_var = obj.momentum * obj.running_var + (1 - obj.momentum) * var;
            else
                xc = x - obj.running_mean;
                xn = xc / (sqrt(obj.running_var + 10e-7));
            end
            
            out = obj.gamma .* xn + obj.beta;
        end
        
        function dx = backward(obj, dout)
            
            if 0
                obj.dgamma = zeros(1,size(dout,2));
                obj.dbeta = zeros(1,size(dout,2));            
                dx = dout;
            else
                dx = obj.backward_internal(dout);
                dx = reshape(dx, obj.input_shape);
            end
       
        end
        
        
        function dx = backward_internal(obj, dout)
            
            dbeta = sum(dout);
            dgamma = sum(obj.xn .* dout);
            dxn = obj.gamma .* dout;
            dxc = dxn ./ obj.std;
            dstd = - sum((dxn .* obj.xc) ./ (obj.std .* obj.std));
            dvar = 0.5 * dstd ./ obj.std;
            dxc = dxc + (2.0 / obj.batch_size) * obj.xc .* dvar;
            dmu = sum(dxc);
            dx = dxc - dmu ./ obj.batch_size;
            
            obj.dgamma = dgamma;
            obj.dbeta = dbeta;
            
        end
        
    end
    
end

