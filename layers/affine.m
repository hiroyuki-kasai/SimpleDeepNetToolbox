classdef affine < handle
% This file defines Affine layer class.
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
        W;
        b;
        x;
        original_x_shape;
        dW;
        db;
    end
    
    methods
        
        function obj = affine(W_in, b_in) 
            
            obj.name = 'affine';
            
            obj.W = W_in;
            obj.b = b_in;

            obj.x = [];
            obj.original_x_shape = [];
            obj.dW = [];
            obj.db = [];
        
        end  
        
        function obj = update_params(obj, W_in, b_in) 
            
            obj.W = W_in;
            obj.b = b_in;
        
        end          
 
        
        function out = forward(obj, x)
 
            obj.original_x_shape = size(x);
            x = reshape(x, obj.original_x_shape(1), []); % for tensor
            obj.x = x;
            
            out = obj.x * obj.W + obj.b;
        end  
        
        function dx = backward(obj, dout)
            
            WT = (obj.W)';
            dx = dout * WT;
            obj.dW = (obj.x)' * dout;
            obj.db = sum(dout, 1);

            dx = reshape(dx, obj.original_x_shape);

        end          
        
    end
    
end