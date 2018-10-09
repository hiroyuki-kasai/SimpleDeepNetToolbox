function grad = numerical_gradient(f, x)

    h = 1e-3;
    
    vec_x = x(:);
    x_len = length(vec_x);
    grad_vec = zeros(x_len, 1);
    
    for idx = 1:x_len
        % store original value
        tmp_val = vec_x(idx);
        
        % replace idx-th element with "vec_x(idx) + h" 
        vec_x(idx) = tmp_val + h;
        f_plus_h = f(vec_x); % f(x+h)
        
        % replace idx-th element with "vec_x(idx) - h"
        vec_x(idx) = tmp_val - h;
        f_minus_h = f(vec_x); % f(x+h)
        
        % calculate4 gradient
        grad_vec(idx) = (f_plus_h - f_minus_h) / (2*h);
        
        % recover the original value
        vec_x(idx) = tmp_val;
    end
    
    grad = reshape(grad_vec, size(x));
end



