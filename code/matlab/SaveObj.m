function [ ] = SaveObj( inputname, ver, face)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[m,n]=size(ver);
if n==1
    newver=reshape(ver, 3,m/3);
else
    newver=ver;
end
[m,n]=size(newver);
fid=fopen(inputname,'w');
for i = 1 : n
    verline=['v ',num2str(newver(1,i)),' ',num2str(newver(2,i)),' ',num2str(newver(3,i))];
    fprintf(fid,'%s\n',verline);
end
[m,n]=size(face);
for i = 1:n
    faceline=['f ',num2str(face(1,i)),' ',num2str(face(2,i)),' ',num2str(face(3,i))];
    fprintf(fid,'%s\n',faceline);
end
fclose(fid);
end

