function new_total_f = AddPartFace2TotalFace(total_f, part_f)
    max_point = max(total_f(:));
    if isempty(max_point)
        new_total_f = [total_f; part_f];
    else
        part_f = part_f + max_point;
        new_total_f = [total_f; part_f];
    end
end