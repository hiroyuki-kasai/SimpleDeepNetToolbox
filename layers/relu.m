classdef relu < handle
% This file defines Relu class.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 01, 2018
% Modified by H.Kasai on Oct. 03, 2018
%
% This class was originally ported from the python library below.
% https://github.com/oreilly-japan/deep-learning-from-scratch.

    
    properties
        name;
        mask;
    end
    
    methods
        
        function obj = relu() 
            
            obj.name = 'relu';
            obj.mask = [];
        
        end  
        

        
        function out = forward(obj, x) 
            
            obj.mask = (x <= 0);
            out = x;
            out(obj.mask) = 0;
        
        end          
        
        
        function dx = backward(obj, dout) 
            
            dout(obj.mask) = 0;
            dx = dout;
        
        end 

    end
    
end

