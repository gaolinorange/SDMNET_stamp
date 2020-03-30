function bbxnew=changebbxvert(cornerpoints)


n1=cornerpoints(2,:)-cornerpoints(1,:);
n2=cornerpoints(4,:)-cornerpoints(1,:);
n3=cornerpoints(5,:)-cornerpoints(1,:);
n1=n1/norm(n1);
n2=n2/norm(n2);
n3=n3/norm(n3);

dot1=n1-[1 1 1];
dot2=n2-[1 1 1];
dot3=n3-[1 1 1];

[~,id1]=min(abs(dot1));
[~,id2]=min(abs(dot2));
[~,id3]=min(abs(dot3));

assert(length(unique([id1,id2,id3]))==3)

[n1,id11]=makedirect(n1,id1);
[n2,id22]=makedirect(n2,id2);
[n3,id33]=makedirect(n3,id3);

h(id11,:)=n1;
h(id22,:)=n2;
h(id33,:)=n3;

bbxc=cornerpoints-mean(cornerpoints);
bbxsign=sign((h*bbxc')');
% bbxsign=sign(bbxc);
for i=1:8
    
    if bbxsign(i,1)<0&&bbxsign(i,2)<0&&bbxsign(i,3)<0
        bbxnew(1,:)=cornerpoints(i,:);
        continue;
    end
    if bbxsign(i,1)>0&&bbxsign(i,2)<0&&bbxsign(i,3)<0
        bbxnew(2,:)=cornerpoints(i,:);
        continue;
    end
    if bbxsign(i,1)>0&&bbxsign(i,2)<0&&bbxsign(i,3)>0
        bbxnew(3,:)=cornerpoints(i,:);
        continue;
    end
    if bbxsign(i,1)<0&&bbxsign(i,2)<0&&bbxsign(i,3)>0
        bbxnew(4,:)=cornerpoints(i,:);
        continue;
    end
    if bbxsign(i,1)<0&&bbxsign(i,2)>0&&bbxsign(i,3)<0
        bbxnew(5,:)=cornerpoints(i,:);
        continue;
    end
    if bbxsign(i,1)>0&&bbxsign(i,2)>0&&bbxsign(i,3)<0
        bbxnew(6,:)=cornerpoints(i,:);
        continue;
    end
    if bbxsign(i,1)>0&&bbxsign(i,2)>0&&bbxsign(i,3)>0
        bbxnew(7,:)=cornerpoints(i,:);
        continue;
    end
    if bbxsign(i,1)<0&&bbxsign(i,2)>0&&bbxsign(i,3)>0
        bbxnew(8,:)=cornerpoints(i,:);
        continue;
    end
end
end


function [n1,id]=makedirect(n1,id1)
if (id1==1)
    if n1(1)<0
        n1=-n1;
    end
    id=1;
elseif id1==2
    if n1(2)<0
        n1=-n1;
    end
    id=1;
elseif id1==3
    if n1(3)<0
        n1=-n1;
    end
    id=1;
else
    error('error')
end
end