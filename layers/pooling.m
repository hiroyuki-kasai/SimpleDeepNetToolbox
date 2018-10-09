classdef pooling < handle
% This file defines pooling layer class.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 04, 2018
% Modified by H.Kasai on Oct. 05, 2018
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.


    properties
        name;
            
        pool_h;
        pool_w;
        stride;
        pad;

        x;
        arg_max;
    end
    
    methods
        
        function obj = pooling(pool_h, pool_w, varargin) 
            
            if nargin < 4
                pad = 0;
            else
                pad = varargin{2};
            end             
            
            if nargin < 3
                stride = 1;
            else
                stride = varargin{1};
            end             
            
            obj.name = 'pooling';
            
            obj.pool_h = pool_h;
            obj.pool_w = pool_w;
            obj.stride = stride;
            obj.pad = pad;

            obj.x = [];
            obj.arg_max = [];
        
        end  
        
       
 
        
        function out = forward(obj, x)
 
            [N, C, H, W] = size(x);
            out_h = fix(1 + (H - obj.pool_h) / obj.stride);
            out_w = fix(1 + (W - obj.pool_w) / obj.stride);

            col = im2col(x, obj.pool_h, obj.pool_w, obj.stride, obj.pad);
            col = reshape(col, [], obj.pool_h * obj.pool_w);

            [out, arg_max] = max(col, [], 2);
            out = reshape(out, [N, out_h, out_w,C]);
            out = permute(out, [1, 4, 2, 3]);
            
            obj.x = x;
            obj.arg_max = arg_max;
            
        end  
        
        function dx = backward(obj, dout)
            
            dout = permute(dout, [1, 3, 4, 2]);
        
            pool_size = obj.pool_h * obj.pool_w;
            
            dmax = zeros(numel(dout), pool_size);
            
            %dmax[np.arange(self.arg_max.size), obj.arg_max.flatten()] = dout.flatten() % To Do
            dout_vec = dout(:);
            for i = 1 : numel(dout)
                dmax(i, obj.arg_max(i)) = dout_vec(i);
            end
            
            % dmax_new = zeros(numel(dout), pool_size); % does not work
            % dmax_new(1:numel(dout), obj.arg_max) = dout_vec; % does not work
            
            dmax = reshape(dmax, [size(dout) pool_size]);
            
            dmax_size_1 = size(dmax, 1);
            dmax_size_2 = size(dmax, 2);
            dmax_size_3 = size(dmax, 3);
        
            dcol = reshape(dmax, dmax_size_1 * dmax_size_2 * dmax_size_3, []);
            dx = col2im(dcol, size(obj.x), obj.pool_h, obj.pool_w, obj.stride, obj.pad);

        end          
        
    end
    
end