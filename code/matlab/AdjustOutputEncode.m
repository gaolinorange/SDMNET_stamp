function [new_encode] = AdjustOutputEncode(encode)
    new_encode = zeros(size(encode));
    n = size(encode, 1);
    for i = 1:n
        for j = 1:2*n+1
            if encode(i, j) < 0.5
                new_encode(i, j) = 0;
            else
                new_encode(i, j) = 1;
            end
        end
        for j = 2*n+2:2*n+4
            new_encode(i, j) = encode(i, j);
        end
        for j = 2*n+5:2*n+9
            if encode(i, j) < 0.5
                new_encode(i, j) = 0;
            else
                new_encode(i, j) = 1;
            end
        end
    end
end