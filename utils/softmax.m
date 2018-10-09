function y = softmax(x)

%     c = max(a);
%     exp_a = exp(a -c);
%     sum_exp_a = sum(exp_a);
%     
%     y = exp_a ./ sum_exp_a;
    
    

    if ismatrix(x)     
        
        x = x';
        x = x - max(x);
        y = exp(x) ./ sum(exp(x));
        
        y = y';
        
    else
        
        x = x - max(x); % to avoid overflow
        y = exp(x) ./ sum(exp(x));         
        
    end

end

