classdef dropout < handle
% This file defines dropout class.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 02, 2018
% Modified by H.Kasai on Oct. 04, 2018
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.
 
    
    properties
        name;
        dropout_ratio;
        train_flag;
        mask;
    end
    
    methods
        
        function obj = dropout(varargin) 
            
            obj.name = 'dropout';
            
            if nargin < 1
                obj.dropout_ratio = 0.5;
            else
                obj.dropout_ratio = varargin{1};
            end            
       
        end   
        
        function masked_x = forward(obj, x, varargin) 
            
            if nargin < 3
                obj.train_flag = true;
            else
                obj.train_flag = varargin{1};
            end            
            
            if obj.train_flag
                obj.mask = rand(size(x)) > obj.dropout_ratio;
                masked_x = x .* obj.mask;
            else
                masked_x = x .* (1.0 - obj.dropout_ratio);
            end
       

        end     
        
        function dout_new = backward(obj, dout)
            
            dout_new = dout .* obj.mask;
       
        end          
        
    end
    
end

