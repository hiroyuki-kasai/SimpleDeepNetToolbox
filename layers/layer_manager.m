classdef layer_manager  < handle
% This file defines layer manager class.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 05, 2018
% Modified by H.Kasai on Oct. 05, 2018


    properties
        name; 
        
        aff_layers;
        aff_layer_num;
        act_layers;
        act_layer_num;
        conv_layers;
        conv_layer_num;
        pool_layers;
        pool_layer_num; 
        dropout_layers;
        dropout_layer_num;  
        batchnorm_layers;
        batchnorm_layer_num;         
        
        last_layer;

        layers;
        total_num;      % = aff_layer_num + act_layer_num + conv_layer_num;          

    end
    
    methods
        function obj = layer_manager() 
            
            obj.name = 'layer_manager';  
            
            obj.aff_layers = [];
            obj.aff_layer_num = 0;
            obj.act_layers = [];
            obj.act_layer_num = 0;
            obj.conv_layers = [];
            obj.conv_layer_num = 0;
            obj.pool_layers = [];
            obj.pool_layer_num = 0; 
            obj.dropout_layers = [];
            obj.dropout_layer_num = 0;  
            obj.batchnorm_layers = [];
            obj.batchnorm_layer_num = 0;              
            
            obj.layers = [];
            obj.total_num = 0;            

        end
        
        
        function obj = add_layer(obj, type, varargin)
            
            obj.total_num = obj.total_num + 1;

            if strcmp(type, 'affine')
                obj.aff_layer_num = obj.aff_layer_num + 1;
                obj.aff_layers{obj.aff_layer_num} = affine(varargin{1}, varargin{2});
                obj.layers{obj.total_num} = obj.aff_layers{obj.aff_layer_num};
            
            elseif strcmp(type, 'relu') || strcmp(type, 'sigmoid')
                obj.act_layer_num = obj.act_layer_num + 1;
                if strcmp(type, 'relu')
                    obj.act_layers{obj.act_layer_num} = relu();
                else
                    obj.act_layers{obj.act_layer_num} = sigmoid();
                end
                obj.layers{obj.total_num} = obj.act_layers{obj.act_layer_num};
            
            elseif strcmp(type, 'convolution')
                obj.conv_layer_num = obj.conv_layer_num + 1;
                obj.conv_layers{obj.conv_layer_num} = convolution(varargin{1}, varargin{2}, varargin{3}, varargin{4});  
                obj.layers{obj.total_num} = obj.conv_layers{obj.conv_layer_num};
            
            elseif strcmp(type, 'pooling')
                obj.pool_layer_num = obj.pool_layer_num + 1;
                obj.pool_layers{obj.pool_layer_num} = pooling(varargin{1}, varargin{2}, varargin{3});  
                obj.layers{obj.total_num} = obj.pool_layers{obj.pool_layer_num};
                
            elseif strcmp(type, 'dropout')
                obj.dropout_layer_num = obj.dropout_layer_num + 1;
                obj.dropout_layers{obj.dropout_layer_num} = dropout(varargin{1});  
                obj.layers{obj.total_num} = obj.dropout_layers{obj.dropout_layer_num};   
                
            elseif strcmp(type, 'batchnorm')
                obj.batchnorm_layer_num = obj.batchnorm_layer_num + 1;
                obj.batchnorm_layers{obj.batchnorm_layer_num} = batch_normalization(varargin{1}, varargin{2}, varargin{3});  
                obj.layers{obj.total_num} = obj.batchnorm_layers{obj.batchnorm_layer_num};                  
            end            

        end  
        
        
        function obj = add_last_layer(obj, type)
            
            if strcmp(type, 'softmax')
                obj.last_layer = softmax_with_loss();
            end            

        end         
        

      end

end

