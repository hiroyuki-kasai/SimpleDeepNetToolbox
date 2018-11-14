classdef nn_layer_basis < handle
% This file defines basic class for neural network class.
%
% This file is part of OptSuite.
%
% Created by H.Kasai on Nov. 13, 2018


    properties
        
        name;  
        
        % layers
        layer_manager; 
        
        % size  
        input_size; 
        output_size;
        hidden_size_list;
        hidden_size;
        hidden_layer_num;
        
        % parameters (W, b)
        params;
        param_num;
        param_keys; 
        
        % data
        x_train;
        y_train;
        x_test;
        y_test; 
        samples;        
        dataset_dim;
        
        % else
        weight_init_std;
        use_num_grad;
        activation_type;
        weight_init_std_type;
        weight_decay_lambda;
        use_dropout;
        dropout_ratio;
        use_batchnorm;
     
    end
    
    methods
        
        function obj = nn_layer_basis(x_train, y_train, x_test, y_test, input_size, hidden_size, output_size, varargin)  

        end
        
        function params_init = set_initial_params(obj, w_init)        
            
            params_init = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                if ~isempty(w_init)
                    params_init(key) = w_init(key);
                else
                    params_init(key) = randn(obj.param_sizes{i});
                end
            end 
        end
        
        
%         function [] = set_params(obj, w)
%             
%             % do nothing
%             
%         end


        function ip = calculate_inner_product(obj, d1, d2)
            
            ip = 0;
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                ip = ip + d1(key)' * d2(key);
            end
            
        end
        
        
        function norm_val = calculate_norm(obj, d1)
            
            norm_val = 0;
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                norm_val = norm_val + norm(d1(key));
            end    
            
        end
        
        
        function eye_mat = generate_identity_mat(obj, params_in)
            
            % c1, c2: scholar value
            
            eye_mat = containers.Map('KeyType','char','ValueType','any');
            
            for i = 1 : obj.param_num
                key = obj.param_keys{i};
                param = params_in(key);
                param_dim = size(param, 1);
                eye_mat(key) = eye(param_dim);
            end
            
        end         
        
        
        function result = lincomb_vecvec(obj, c1, v1, c2, v2)
            
            % calculate result = c1 * v1 + c2 + v2.
            % c1, c2: scalar
            % v1, v2: vector            
            
            result = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                if (c1 ~= 0) && (c2 ~= 0)
                    result(key) = c1 * v1(key) + c2 * v2(key);
                elseif (c1 ~= 0) && (c2 == 0)
                    result(key) = c1 * v1(key);
                else
                    warning('Error');
                end
            end 
            
        end
        
        function result = lincomb_vecmatvec(obj, c1, v1, c2, M2, v2)
            
            % calculate result = c1 * M1 * v1 + c2 + M2 * v2.
            % c1, c2: scalar
            % M2: matrix
            % v1, v2: vector    
            
            result = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                result(key) = c1 * v1(key) + c2 * M2(key) * v2(key);

            end 
            
        end
        
        
        function result = lincomb_vecmatvec2(obj, c1, v1, c2, M2, v2)
            
            % calculate result = c1 * M1 * v1 + c2 + M2 * v2.
            % c1: vector, but each component behaves as a coefficient scalar
            % c2: scalar
            % M2: matrix
            % v1, v2: vector    
            
            result = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                result(key) = c1(key) * v1(key) + c2 * M2(key) * v2(key);

            end 
            
        end        
        
        
        function result = lincomb_matvecmatvec(obj, c1, M1, v1, c2, M2, v2)
            
            % calculate result = c1 * M1 * v1 + c2 + M2 * v2.
            % c1, c2: scalar
            % M1, M2: matrix
            % v1, v2: vector    
            
            result = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                if (c1 ~= 0) && (c2 ~= 0)
                    result(key) = c1 * M1(key) * v1(key) + c2 * M2(key) * v2(key);
                elseif (c1 ~= 0) && (c2 == 0)
                    result(key) = c1 * M1(key) * v1(key);
                else
                    warning('Error');
                end
            end 
            
        end  
        
                
        function result = mul_mat(obj, c1, M1)
            
            % calculate result = c1 * M1.
            % c1: scalar
            % M1: matrix
            
            result = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                result(key) = c1 .* M1(key);
            end
            
        end
        
        
        function result = mul_invmatvec(obj, c1, M1, v1)
            
            % calculate result = c1 * inv(M1) * v1.
            % c1: scalar
            % M1: matrix
            % v1: vector 
            
            result = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                result(key) = c1 * M1(key) \ v1(key);
            end
            
        end
        
        
        function result = transvecmatvec(obj, c1, v1, M1, v2)
            
            % calculate result = c1 * v1' * M1 * v2.
            % c1: scalar
            % M1: matrix
            % v1, v2: vector 
            
            result = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                result(key) = c1 * v1(key)' * M1(key) * v2(key);
            end
            
        end        
        
        
        function find_nan_flag = find_nan_mat(obj, M1)  
            
            find_nan_flag = 0;
            
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                M_element = M1(key);
                if isnan(M_element(:))
                    find_nan_flag = 1;
                    return;
                end
            end
            
        end
        
        
        function M1t = transpose_operator(obj, M1)  
            
            M1t = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                M1t(key) = M1(key)';
            end
            
        end        
        
        
        
        function M1diag = diag_operator(obj, M1)  
            
            M1diag = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                M1diag(key) = diag(1./diag(M1(key)));
                
            end
            
        end 
        
        
        function nonnegativeM1 = nonnegative_mat_operator(obj, M1)  
            
            nonnegativeM1 = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                nonnegativeM1(key) = max(0,M1(key));
                
            end
            
        end  
        
        function nonnegativev1 = nonnegative_vec_operator(obj, v1)  
            
            nonnegativev1 = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                nonnegativev1(key) = max(0,v1(key));
                
            end
            
        end          
        
        
        function result = div_scalars(obj, c1, c2)
            
            % calculate result = c1/c2.
            % c1, c2: scalar
            
            result = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                result(key) = c1(key) / c2(key);
            end
            
        end        
        
        
        
        %%
        function [L, error] = chol_dec(obj, M1)  
            
            error = 0;
            
            L = containers.Map('KeyType','char','ValueType','any');
            for i = 1 : obj.param_num
                key = obj.param_keys{i};

                M_element = M1(key);
                if isnan(M_element(:))
                    error = 1;
                    return;
                end
                
                [L(key), p] = chol(M_element, 'lower');                
                if p ~= 0
                    error = 1;
                    return;
                end
            end
            
        end
        
     
        %%
        function obj = problem_single_param(x_train, y_train, x_test, y_test, varargin)  
            obj.param_num = 1;
            obj.param_keys{1} = 'w'; 
        end        
        

        function f = calculate_cost(obj, params)
            
            f = cost(obj, params('w'));
            
        end
        
        function r = calculate_reg(obj, params)
            
            r = reg(obj, params('w'));
            
        end        
        
        function [grads, calc_cnt, gnorm] = calculate_grads(obj, params, indice)
            
            calc_cnt = 0;
            grads = containers.Map('KeyType','char','ValueType','any');
            
            grads('w') = obj.grad(params('w'), indice);
            calc_cnt = calc_cnt + length(indice);
            gnorm = norm(grads('w'));
            
        end
        
        
        function [grads, calc_cnt, gnorm] = calculate_full_grads(obj, params)
            
            [grads, calc_cnt, gnorm] = calculate_grads(obj, params, 1:obj.n_train);
            
        end
                
        
        function params = calculate_proximal_operator(obj, params, step)
            
            params('w') = obj.problem.prox(params('w'), step);
            
        end        
        
        
        function [hess, calc_cnt] = calculate_hess(obj, params, indice)
            
            calc_cnt = 0;  
            hess = containers.Map('KeyType','char','ValueType','any');
            
            hess('w') = obj.hess(params('w'), indice);
            calc_cnt = calc_cnt + length(indice);
        end 
                


     end
end

