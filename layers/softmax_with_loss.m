classdef softmax_with_loss < handle
% This file defines softmax_with_loss layer class.
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
        loss;
        y;
        t;
    end
    
    methods
        
        function obj = softmax_with_loss() 
            
            obj.name = 'softmax_with_loss';
            obj.loss = [];
            obj.y = [];
            obj.t = [];
        
        end   
        
        function loss = forward(obj, x, t) 
            
            obj.y = softmax(x);
            obj.t = t;
            obj.loss = cross_entropy_error(obj.y, obj.t);

            loss = obj.loss;
        
        end    
        
        function dx = backward(obj, dout) 
            
            batch_size = size(obj.t, 1);
            
            if size(obj.t) == size(obj.y) % When y is one-hot-vector;
                dx = (obj.y - obj.t) / batch_size;
            else
                dx = obj.y;
                %dx[np.arange(batch_size), self.t] -= 1 % to do
                dx = dx / batch_size;
            end
        
        end          
    end
    
end