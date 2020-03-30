function pcout=removeoutliner(pc)
x=pc(:,1);
y=pc(:,2);
z=pc(:,3);
tfx=isout(x);
tfy=isout(y);
tfz=isout(z);
isouta=any([tfx,tfy,tfz],2);
pcout=pc(~isouta,:);
end