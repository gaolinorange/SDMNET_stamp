function part_names = getlabel(cate)
    type = cate;
    if strcmp(type, 'chair') == 1
        part_names = {'armrest_1', 'armrest_2', 'back', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'seat'};
    elseif strcmp(type, 'knife') == 1
        part_names = {'part1', 'part2'};
    elseif strcmp(type, 'guitar') == 1
        part_names = {'part1', 'part2', 'part3'};
    elseif strcmp(type, 'monitor') == 1
        part_names = {'display', 'connector', 'base'};
    elseif strcmp(type, 'cup') == 1
        part_names = {'part1', 'part2'};
    elseif strcmp(type, 'car') == 1
        part_names = {'body', 'left_front_wheel', 'right_front_wheel', 'left_back_wheel', 'right_back_wheel', 'left_mirror', 'right_mirror'};
    elseif strcmp(type, 'plane') == 1
        part_names = {'body', 'left_wing', 'right_wing', 'left_tail', 'right_tail', 'upper_tail', 'down_tail', 'front_gear', 'left_gear', 'right_gear', 'left_engine1', 'right_engine1', 'left_engine2', 'right_engine2'};
    elseif strcmp(type, 'table') == 1
        part_names = {'surface', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'leg3_1', 'leg3_2', 'leg4_1', 'leg4_2'};
    end
end