function [direction] = get_direction(A, B)
    distances = pdist2(A, B);%calculate the distance between 2 datasets 
    minDistance = min(distances(:));%find the minimum value of distance
    [rowOfA, rowOfB] = find(distances == minDistance);
    A_coord_closest=A(rowOfA,:);
    B_coord_closest=B(rowOfB,:);
    direction = A_coord_closest - B_coord_closest;
end
