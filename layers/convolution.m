classdef convolution < handle
% This file defines convolution layer class.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 03, 2018
% Modified by H.Kasai on Oct. 05, 2018 
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.
       
    
    properties
        name;
        W;
        b;
        stride;
        pad;
        x;
        col;
        col_W;
        dW;
        db;
    end
    
    methods
        
        function obj = convolution(W, b, varargin) 
            
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
            
            obj.name = 'convolution';
            
            obj.W = W;
            obj.b = b;
            obj.stride = stride;
            obj.pad = pad;

            % inntermediate data for backward
            obj.x = [];   
            obj.col = [];
            obj.col_W = [];

            % gradients of parameters
            obj.dW = [];
            obj.db = [];
        
        end  
        
        function obj = update_params(obj, W_in, b_in) 
            
            obj.W = W_in;
            obj.b = b_in;
        
        end          
 
        
        function out = forward(obj, x)
 
            [FN, C, FH, FW] = size(obj.W);
            [N, C, H, W] = size(x);
            out_h = 1 + fix((H + 2*obj.pad - FH) / obj.stride);
            out_w = 1 + fix((W + 2*obj.pad - FW) / obj.stride);

            col = im2col(x, FH, FW, obj.stride, obj.pad);
            col_W = reshape(obj.W, FN, []);
            col_W = col_W';

            out = col * col_W + obj.b;
            
            %out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2);
            col_W_col_size = size(col_W, 2);
            out = reshape(out,[N, out_h, out_w, col_W_col_size]);
            out = permute(out, [1, 4, 2, 3]);
            

            obj.x = x;
            obj.col = col;
            obj.col_W = col_W;
        end  
        
        function dx = backward(obj, dout)
            
            [FN, C, FH, FW] = size(obj.W);
            dout = permute(dout, [1, 3, 4, 2]);
            dout = reshape(dout, [], FN);
            
            obj.db = sum(dout);
            obj.dW = (obj.col)' * dout;
            obj.dW = permute(obj.dW, [2, 1]);
            obj.dW = reshape(obj.dW, [FN, C, FH, FW]);

            dcol = dout * (obj.col_W)';
            dx = col2im(dcol, size(obj.x), FH, FW, obj.stride, obj.pad);

        end          
        
    end
    
end