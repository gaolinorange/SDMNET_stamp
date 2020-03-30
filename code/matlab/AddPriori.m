function [code] = AddPriori(code, type, part_names)
    part_num = size(part_names, 2);
    if strcmp(type, 'chair') == 1
        code(8, 1+3) = 1;
        code(3, 1+8+part_num) = 1;
        for i = 4:7
            code(i, 1+8) = 1;
            code(8, 1+i+part_num) = 1;
        end
        symmetry = [
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    ];
        code(:, 2*part_num+5:2*part_num+9) = symmetry;
    elseif strcmp(type, 'knife') == 1
        code(2, 1+1) = 1;
        code(1, 1+2+part_num) = 1;
        symmetry = [
                    0, 0, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    ];
        code(:, 2*part_num+5:2*part_num+9) = symmetry;
    elseif strcmp(type, 'guitar') == 1
        supporting = [
                        0, 0, 0;
                        1, 0, 0;
                        0, 1, 0
                    ];
        code(:, 2:1+part_num) = supporting;
        code(:, 2+part_num:1+2*part_num) = supporting';
        symmetry = [
                    0, 0, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    ];
        code(:, 2*part_num+5:2*part_num+9) = symmetry;
    elseif strcmp(type, 'skateboard') == 1
        supporting = [
                        0, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 1, 1, 1, 1;
                        1, 0, 0, 1, 1, 1, 1;
                        1, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0;
                    ];
        code(:, 2:1+part_num) = supporting;
        code(:, 2+part_num:1+2*part_num) = supporting';
        symmetry = [
                    0, 0, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    ];
        code(:, 2*part_num+5:2*part_num+9) = symmetry;
    elseif strcmp(type, 'cup') == 1
        supporting = [
                        0, 0;
                        1, 0;
                    ];
        code(:, 2:1+part_num) = supporting;
        code(:, 2+part_num:1+2*part_num) = supporting';
        symmetry = [
                    0, 0, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    ];
        code(:, 2*part_num+5:2*part_num+9) = symmetry;
    elseif strcmp(type, 'car') == 1
        supporting = [
                        0, 0, 0, 0, 0, 1, 1;
                        1, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0;
                    ];
        code(:, 2:1+part_num) = supporting;
        code(:, 2+part_num:1+2*part_num) = supporting';
        symmetry = [
                    0, 0, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    ];
        code(:, 2*part_num+5:2*part_num+9) = symmetry;
    elseif strcmp(type, 'plane') == 1
        supporting = [
                        0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                    ];
        code(:, 2:1+part_num) = supporting;
        code(:, 2+part_num:1+2*part_num) = supporting';
        symmetry = [
                    0, 0, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    0, 0, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    ];
        code(:, 2*part_num+5:2*part_num+9) = symmetry;
    elseif strcmp(type, 'table') == 1
        supporting = [
                        0, 0, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0, 0;
                        1, 0, 0, 0, 0, 0, 0, 0;
                    ];
        code(:, 2:1+part_num) = supporting;
        code(:, 2+part_num:1+2*part_num) = supporting';
        symmetry = [
                    0, 0, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    1, 1, 0, 0, 0;
                    ];
        code(:, 2*part_num+5:2*part_num+9) = symmetry;
    end
end