classdef sigmoid < handle
% This file defines sigmoid layer class.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 01, 2018
% Modified by H.Kasai on Oct. 02, 2018    
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.


    properties
        name;
        out;
    end
    
    methods
        
        function obj = sigmoid() 
            
            obj.name = 'sigmoid';
            obj.out = [];
        
        end  
        

        
        function out = forward(obj, x) 
            
            out = calc_sigmoid(x);
            obj.out = out;
        
        end          
        
        
        function dx = backward(obj, dout) 
            
            dx = dout .* (1.0 - obj.out) .* obj.out;
        
        end 

    end
    
end

