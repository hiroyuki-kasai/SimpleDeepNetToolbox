function return_img = col2im(col, input_shape, filter_h, filter_w, varargin)
%
%
%   Input: 
%       col         : 
%       input_shape : shape of input col (e.g. (10, 1, 28, 28))
%       filter_h    : filter height
%       filter_w    : filter width
%       stride      : stride
%       pad         : padding
%
%   Output:
%       return_img  : 2-dimensinal array
%



    if nargin < 6
        pad = 0;
    else
        pad = varargin{2};
    end             

    if nargin < 5
        stride = 1;
    else
        stride = varargin{1};
    end 
    
    
    N = input_shape(1);
    C = input_shape(2);
    H = input_shape(3);
    W = input_shape(4);
    out_h = fix(H + 2*pad - filter_h)/stride + 1;
    out_w = fix(W + 2*pad - filter_w)/stride + 1;
    col = reshape(col, N, out_h, out_w, C, filter_h, filter_w);
    col = permute(col, [1, 4, 5, 6, 2, 3]);
    
    
    img = zeros(N, C, H + 2*pad + stride - 2, W + 2*pad + stride - 2);
    %img_new = zeros(N, C, H + 2*pad + stride - 2, W + 2*pad + stride - 2);

            
%     for y = 1 : filter_h
%         y_max = y + stride*out_h - 1;
%         for x = 1 : filter_w
%             x_max = x + stride*out_w - 1;
% 
%             for k = 1 : N
%                 for j = 1 : C
%                       img(k, j, y:stride:y_max, x:stride:x_max) = squeeze(col(k, j, y, x, :, :));
%                 end
%             end
% 
%         end
%     end    
    
    
    for y = 1 : filter_h
        y_max = y + stride*out_h - 1;
        for x = 1 : filter_w
            x_max = x + stride*out_w - 1;
            %img(:, :, y:stride:y_max, x:stride:x_max) = squeeze(col(:, :, y, x, :, :));
            img(:, :, y:stride:y_max, x:stride:x_max) = col(:, :, y, x, :, :);
        end
    end       

    return_img = img(:, :, pad+1:H+pad, pad+1:W+pad);    



end




