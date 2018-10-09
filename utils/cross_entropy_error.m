function ce = cross_entropy_error(y, t)

    % input
    %    y: (batch_size, class_num)
    
    if ndims(y) == 1
        t = t(:)';
        y = y(:)';
    end
    


%     if size(t) == size(y)
%         % extract correct label in each batch (row)
%         [~, t_max_idx] = max(t, [], 2);
%     else
%         t_max_idx = t;
%     end

    % if "t" (correct label) is one-hot-vector, it needs to be vectorized.
    if ismatrix(t)
        % extract correct label in each batch (row)
        [~, t_max_idx] = max(t, [], 2);        
    else
        t_max_idx = t;
    end
    
    batch_size = size(y, 1);

    delta = 1e-7;
    
    
%     ce = 0;
%     for i = 1 : batch_size
%         y_val = y(i, t_max_idx(i));
%         ce = ce - log(y_val + delta);
%     end
    
    y_val = zeros(batch_size, 1);
    for i = 1 : batch_size
        y_val(i) = y(i, t_max_idx(i));
    end   
    ce = sum(-log(y_val + delta));
    
    
    ce = ce/batch_size; 

end


