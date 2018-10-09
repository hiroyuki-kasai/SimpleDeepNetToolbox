function col = im2col(input_data, filter_h, filter_w, varargin)
%
%
%   Input: 
%       input_data  : 4-dimensinal array (data num, channel, hight, widhth)
%       filter_h    : filter height
%       filter_w    : filter width
%       stride      : stride
%       pad         : padding
%
%   Output:
%       col         : 2-dimensinal array
%

    if nargin < 5
        pad = 0;
    else
        pad = varargin{2};
    end             

    if nargin < 4
        stride = 1;
    else
        stride = varargin{1};
    end 


    [N, C, H, W] = size(input_data);
    
    out_h = fix((H + 2*pad - filter_h)/stride) + 1;
    out_w = fix((W + 2*pad - filter_w)/stride) + 1;

    %img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant'); % To Do
    img = input_data;
    
    col = zeros(N, C, filter_h, filter_w, out_h, out_w);
  
%     for y = 1 : filter_h
%         y_max = y + stride*out_h - 1;
%         for x = 1 : filter_w
%             x_max = x + stride*out_w - 1;
% 
%             for k = 1 : N
%                 for j = 1 : C
%                       col(k, j, y, x, :, :) = squeeze(img(k, j, y:stride:y_max, x:stride:x_max));
%                 end
%             end
% 
%         end
%     end
    
    
    for y = 1 : filter_h
        y_max = y + stride*out_h - 1;
        for x = 1 : filter_w
            x_max = x + stride*out_w - 1;
            %col(:, :, y, x, :, :) = squeeze(img(:, :, y:stride:y_max, x:stride:x_max));
            col(:, :, y, x, :, :) = img(:, :, y:stride:y_max, x:stride:x_max);
        end
    end    
    
    
    col = permute(col, [1, 5, 6, 2, 3, 4]);
    col = reshape(col, [N*out_h*out_w, C*filter_h*filter_w]);
    

end




