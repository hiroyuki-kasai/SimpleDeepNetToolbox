function label_mat  = convert_labelvec_to_mat(label_vec, num, class_num)
% This file defines a function to a vector-form label set to a matrix-form
% one.
%
% This file is part of SimpleDeepNetToolbox.
%
% Created by H.Kasai on Oct. 02, 2018
% Modified by H.Kasai on Oct. 05, 2018


    if min(label_vec) == 0
        offset = 1;
    elseif min(label_vec) == 1
        offset = 0;        
    else
        warning('Unexpected class label')
    end
        
    label_mat = zeros(class_num, num);
    for i=1:num
        label_mat(label_vec(i)+offset,i) = 1;
    end
end

