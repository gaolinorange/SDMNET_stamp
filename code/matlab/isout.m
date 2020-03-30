function all=isout(data)
% c=mean(data);
% data=abs(data-c);
all=zeros(length(data),1);
[data_sort,id]=sort(data);
grad=diff(data_sort);

idx = find(grad > 0.02);
if ~isempty(idx)
if idx<length(data)/2
    id_out=id(1:idx);
all(id_out)=1;
else

id_out=id(idx+1:end);
all(id_out)=1;
end
end
end