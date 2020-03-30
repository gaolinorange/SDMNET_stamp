function  NonRigidAlignment3Dnew(srcfile,tarfile,iter,savefile,W1,W2,W3,W4)
global w1;w1 = W1;
global w2;w2 = W2;
global w3;w3 = W3;
global w4;w4 = W4;
demo(srcfile,tarfile);
run(iter,savefile);
end
%----------------------------- Optimization -------------------------------
function run(iter,savefile)
global stopnow;
stopnow=false;
global type;
global X;
global XF;
global YF;
global Y;
global Z;
global Ex;
% global NY;
global NFY;
% global NX;
global w1;
global w2;
global w3;
global w4;
%w3 = 3;
%w4 = 10000;
global MX;
% global MY;
global Ni;
% global Nr;
%w1 = 1.0;
%w2 = 0.1;
Z = X;
Zo = Z;
type =2;

% Initialize linear system ||D^0.5(Av - b)||_2^2 + ||W^0.5v||_2^2
dim = size(Z,1)*size(Z,2);
edim = size(MX,1);
Nvec = size(Z,1);
Ex = MX*X;
A = sparse(size(Z,1)+2*dim+3*edim, dim+dim+6);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A((1+size(Z,1)):(size(Z,1)+dim),1:dim) = speye(dim,dim);
A((1+size(Z,1)+dim):(size(Z,1)+2*dim),1:dim) = speye(dim,dim);
A((1+size(Z,1)  +dim):(size(Z,1)*2+dim), dim+4) = -ones(size(Z,1),1);
A((1+size(Z,1)*2+dim):(size(Z,1)*3+dim), dim+5) = -ones(size(Z,1),1);
A((1+size(Z,1)*3+dim):(size(Z,1)*4+dim), dim+6) = -ones(size(Z,1),1);

A((size(Z,1)+2*dim+1):(size(Z,1)+2*dim+edim),1:size(Z,1)) = MX;
A((size(Z,1)+2*dim+edim+1):(size(Z,1)+2*dim+2*edim),(size(Z,1)+1):(2*size(Z,1))) = MX;
A((size(Z,1)+2*dim+2*edim+1):end,(2*size(Z,1)+1):(dim)) = MX;
% b = zeros(size(Z,1)+2*dim+3*edim,1);
D = sparse(size(Z,1)+2*dim+3*edim, size(Z,1)+2*dim+3*edim);
D(1:size(Z,1),1:size(Z,1)) = w1*speye(size(Z,1),size(Z,1));
% D((size(Z,1)+1):(size(Z,1)+dim),(size(Z,1)+1):(size(Z,1)+dim)) = w2*speye(dim,dim);
W = sparse(dim+dim+6, dim+dim+6);
W((dim+1):(dim+6), (dim+1):(dim+6)) = 0.1*speye(6, 6);
W((dim+7):end,(dim+7):end) = 1*speye(dim, dim);

scale_w4=0.80;
scale_w2=1.0;
pBar=TimedProgressBar(iter,30,'',' Percent complete ',' Finish ');
for it=1:iter
    
    if it<20
        if it<10
            w3=0.0;
        else
            w3=0.0;
        end        
        [sqrDz,P,NP]=search_nn_bidirector(Z,XF,Y,YF,NFY);        
        w2=0.01*10000;
%         w2=repmat(exp(50*sqrDz)-0.9,3,1)*10;
    else     
        w3=0.0;
        scale_w4=0.97;        
        if (iter - it < 5)
            [sqrDz,P,NP]=search_nn_bidirector(Z,XF,Y,YF,NFY);
            %             sqrDz(sqrDz<1e-4)=0;
            w2=0.01*2000;%repmat(exp(50*sqrDz)-0.9,3,1)*10;
%             w2=repmat(exp(50*sqrDz)-0.8,3,1)*10;
        else
            [sqrDz,P,NP]=search_nn_bidirector(Z,XF,Y,YF,NFY);
            %                                 sqrDz(sqrDz<1e-4)=0;
            w2=0.01*2000;%repmat(exp(100*sqrDz)-0.9999,3,1)*10;
%             w2=repmat(exp(50*sqrDz)-0.8,3,1)*10;
        end        
    end
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Build linear system
    N = [spdiags(NP(:,1),0,size(NP,1),size(NP,1)), ...
        spdiags(NP(:,2),0,size(NP,1),size(NP,1)), ...
        spdiags(NP(:,3),0,size(NP,1),size(NP,1))];
    A(1:size(Z,1),1:dim) = N;

    
    A(sub2ind(size(A),(size(Z,1)+(1:3*size(Z,1))+dim)',kron([dim+1;dim+3;dim+2],ones(size(Z,1),1)))) = reshape(X(:,[2,3,1]),[],1);
    A(sub2ind(size(A),(size(Z,1)+(1:3*size(Z,1))+dim)',kron([dim+2;dim+1;dim+3],ones(size(Z,1),1)))) = reshape(-X(:,[3,1,2]),[],1);
%     A((1+size(Z,1)  +dim):(size(Z,1)*2+dim),dim+1) =  X(:,2);
%     A((1+size(Z,1)*2+dim):(size(Z,1)*3+dim),dim+1) = -X(:,1);
%     A((1+size(Z,1)  +dim):(size(Z,1)*2+dim),dim+2) = -X(:,3);
%     A((1+size(Z,1)*3+dim):(size(Z,1)+2*dim),dim+2) =  X(:,1);
%     A((1+size(Z,1)*2+dim):(size(Z,1)*3+dim),dim+3) =  X(:,3);
%     A((1+size(Z,1)*3+dim):(size(Z,1)+2*dim),dim+3) = -X(:,2);
    
    %         tic
    A(sub2ind(size(A), (size(Z,1)+2*dim+       (1:3*edim))', [dim+6+       Ni;dim+6+2*Nvec+Ni;dim+6+  Nvec+Ni]))=reshape(Ex(:,[2,3,1]),[],1);
    A(sub2ind(size(A), (size(Z,1)+2*dim+       (1:3*edim))', [dim+6+  Nvec+Ni;dim+6+      +Ni;dim+6+2*Nvec+Ni]))=reshape(-Ex(:,[3,1,2]),[],1);
%     A(sub2ind(size(A), [size(Z,1)+2*dim+       (1:edim)]', dim+6+       Ni))=Ex(:,2);
%     A(sub2ind(size(A), [size(Z,1)+2*dim+       (1:edim)]', dim+6+  Nvec+Ni))=-Ex(:,3);
%     A(sub2ind(size(A), [size(Z,1)+2*dim+  edim+(1:edim)]', dim+6+       Ni))=-Ex(:,1);
%     A(sub2ind(size(A), [size(Z,1)+2*dim+  edim+(1:edim)]', dim+6+2*Nvec+Ni))=Ex(:,3);
%     A(sub2ind(size(A), [size(Z,1)+2*dim+2*edim+(1:edim)]', dim+6+  Nvec+Ni))=Ex(:,1);
%     A(sub2ind(size(A), [size(Z,1)+2*dim+2*edim+(1:edim)]', dim+6+2*Nvec+Ni))=-Ex(:,2);
%     for i=1:edim
%         numI = Ni(i);
%         A(size(Z,1)+2*dim       +i, dim+6+       numI) = Ex(i,2);
%         A(size(Z,1)+2*dim       +i, dim+6+  Nvec+numI) = -Ex(i,3);
%         A(size(Z,1)+2*dim+  edim+i, dim+6+       numI) = -Ex(i,1);
%         A(size(Z,1)+2*dim+  edim+i, dim+6+2*Nvec+numI) = Ex(i,3);
%         A(size(Z,1)+2*dim+2*edim+i, dim+6+  Nvec+numI) = Ex(i,1);
%         A(size(Z,1)+2*dim+2*edim+i, dim+6+2*Nvec+numI) = -Ex(i,2);
%     end
    %         toc
    b=[N*reshape(P,dim,1);reshape(P,dim,1);reshape(X,dim,1);reshape(Ex,edim*3,1)];
%     b(1:size(Z,1)) = N*reshape(P,dim,1);
%     b((1+size(Z,1)):(size(Z,1)+dim)) = reshape(P,dim,1);
%     b((1+size(Z,1)+dim):(size(Z,1)+2*dim)) = reshape(X,dim,1);
%     b((size(Z,1)+2*dim+1):end) = reshape(Ex,edim*3,1);
%     D((size(Z,1)+1):(size(Z,1)+dim),(size(Z,1)+1):(size(Z,1)+dim)) = w2.*speye(dim,dim);
    D(sub2ind(size(D),(size(Z,1)+1):size(D,1),(size(Z,1)+1):size(D,2)))=[w2.*ones(dim,1);w3*ones(dim,1);w4*ones(edim*3,1)];
%     D((size(Z,1)+dim+1):(size(Z,1)+2*dim),(size(Z,1)+dim+1):(size(Z,1)+2*dim)) =w3*speye(dim,dim);
%     D((size(Z,1)+2*dim+1):end,(size(Z,1)+2*dim+1):end) = w4*speye(edim*3,edim*3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Solve
    v = (A'*D*A + W)\(A'*D*b);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Extract solution
    Z = reshape(v(1:dim), size(X,1), size(X,2));
    R = tranR( -v( dim+1 ), v( dim+2 ), -v( dim+3 ));
    C = v((dim+4):(dim+6))';
    X = X*R' + repmat(C, [size(X,1),1]);
    
%     a1=reshape(v,3,[])';
    
%     a1(1:(dim+6)/3,:)=[];
%     a1=v;
    v(1:dim+6)=[];
    a2=reshape(v,[],3);
    Rset=permute(eul2rotm(a2(Ni,:)),[3,2,1]);
    Ex=squeeze(sum(Ex.*Rset,2));

   
%     for i=1:edim
%         numI = Ni(i);
%         R = tranR( v( dim+6+numI ),v( dim+6+Nvec+numI ), v( dim+6+2*Nvec+numI ));
%         Ex(i,:) = Ex(i,:)*R';
%     end
% %     
    
    if( (norm(Z-Zo)/size(Z,1) < 1e-6) && type == 1)
        %         return
    elseif( (norm(Z-Zo)/size(Z,1) < 1e-4) && type == 2)
        w3 = w3*0.4;
        w4 = w4*0.9;
    end
    Zo = Z;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    w2 = w2* scale_w2;
    w4 = w4* scale_w4;
    pBar.progress();
    %         SaveObjT( [savefile,num2str(it),'.obj'], Z', XF');
end
pBar.stop();

SaveObjT( savefile , Z', XF');

end
%==========================================================================
%----------------------------- SUBROUTINES --------------------------------
%==========================================================================
function X = corrupt_data(X, offset, gamma,beta,alpha, noise)
s = RandStream('mt19937ar','Seed',10);
Rot = [cos(alpha) -sin(gamma) sin(beta); sin(gamma) cos(beta) -sin(alpha); -sin(beta) sin(alpha) cos(gamma)];
X = X*Rot';
X = X + repmat(offset, [size(X,1),1]);
X = X+noise*randn(s,size(X));
end
function [X Y] = mean_center_and_normalize(X,Y)
RX = min(X(:,1));
RY = min(X(:,2));
RZ = min(X(:,3));
X = X - repmat([RX,RY,RZ], [size(X,1),1]);
RX = min(Y(:,1));
RY = min(Y(:,2));
RZ = min(Y(:,3));
Y = Y - repmat([RX,RY,RZ], [size(Y,1),1]);
RR = (max(Y(:,3))-min(Y(:,3)))/4;
Y = Y + repmat([0,0,RR], [size(Y,1),1]);
%XnY = [X;Y];
%avg = mean(XnY);
%X = X - repmat(avg, [size(X,1),1]);
%Y = Y - repmat(avg, [size(Y,1),1]);
%XnY = [X;Y];
%d = sqrt(max(sum(XnY.^2,2)));
%X = X./d;
%Y = Y./d;
end

function show_contour(X,Y)
cla;
%     rotm=axang2rotm([0,0,-1,pi/2]);
%     X=(rotm*X')';
%     Y=(rotm*Y')';
global XF;
global YF;


%     c = 10*ones(size(X,1),1);
%     subplot(1,3,1)
%     trimesh( XF, X(:,1), X(:,2), X(:,3),c);
% % plot3(X(:,1), X(:,2), X(:,3),'r*')
%     hold on;
%     %figure
%     c = 70*ones(size(Y,1),1);
% %     trimesh( YF, Y(:,1), Y(:,2), Y(:,3),c);
%     plot3(Y(:,1), Y(:,2), Y(:,3),'r*')
%     %show_closest();
%     axis equal
%     view(-90,0)
%         c = 10*ones(size(X,1),1);
%     subplot(1,3,2)
%     trimesh( XF, X(:,1), X(:,2), X(:,3),c);
% % plot3(X(:,1), X(:,2), X(:,3),'r*')
%     hold on;
%     %figure
%     c = 70*ones(size(Y,1),1);
% %     trimesh( YF, Y(:,1), Y(:,2), Y(:,3),c);
%     plot3(Y(:,1), Y(:,2), Y(:,3),'r*')
%     %show_closest();
%     axis equal
%     view(0,0)
%
%         subplot(1,3,3)
%          c = 10*ones(size(X,1),1);
%     trimesh( XF, X(:,1), X(:,2), X(:,3),c);
% % plot3(X(:,1), X(:,2), X(:,3),'r*')
%     hold on;
%     %figure
%     c = 70*ones(size(Y,1),1);
% %     trimesh( YF, Y(:,1), Y(:,2), Y(:,3),c);
%     plot3(Y(:,1), Y(:,2), Y(:,3),'r*')
%     %show_closest();
%        axis equal
%     view(0,90)
% ttl={'主视图','左视图','俯视图','三维图'};
% angle={[0,0],[-90,0],[0 90],[-37.5,30]};
% for i=1:4
%      subplot(2,2,i)
%     c = 10*ones(size(X,1),1);
%     trimesh( XF, X(:,1), X(:,2), X(:,3),c);hold on;
%         plot3(Y(:,1), Y(:,2), Y(:,3),'r*');
% % plot3(X(:,1), X(:,2), X(:,3),'r*')
% %     hold on;
%     %figure
% %     c = 70*ones(size(Y,1),1);
% %   trimesh( YF, Y(:,1), Y(:,2), Y(:,3),c);
%
%     %show_closest();
%         axis equal
% view(angle{i});title(ttl{i});
%    pause(0.1)
%    drawnow;
%
% end

end
function [xx,p,cor]=draw_pre(X,Y,XF)
clf;
ttl={'主视图','左视图','俯视图','三维图'};
angle={[0,0],[-90,0],[0 90],[-37.5,30]};
for i=1:4
    subplot(2,3,i)
    c = 10*ones(size(X,1),1);
    plot3(Y(:,1), Y(:,2), Y(:,3),'r*');
    
    hold on;
    p{i}=trimesh( XF, X(:,1), X(:,2), X(:,3),c);
    axis equal
    view(angle{i});title(ttl{i});
    xx{i}=get(p{i},'Vertices');
end

subplot(2,3,5)

plot3(X(:,1)+1,X(:,2),X(:,3),'g.')
hold on
plot3(Y(:,1),Y(:,2),Y(:,3),'bo')
hold on
axis equal
step=1:50:min(size(Y,1),size(X,1));
for i=step
    cor{i}=plot3([X(i,1)+1 Y(i,1)], [X(i,2) Y(i,2)],[X(i,3) Y(i,3)], '-', 'color',[1  0 0]);
end

% subplot(1,2,1),plot(rand(10,1),rand(10,1),'.'),hold on,p1=plot(rand(1),rand(1),'.r')
% subplot(1,2,2),plot(rand(10,1),rand(10,1),'.'),hold on,p2=plot(rand(1),rand(1),'.r')
%
% %# read the red coordinates - I should have stored them before plotting :)
% x(1) = get(p1,'xdata');y(1)=get(p1,'ydata');x(2)=get(p2,'xdata');y(2)=get(p2,'ydata');
end
function show_pic(xx,pp,X,idz,cor)
xx{1}=X;
xx{2}=X;
xx{3}=X;
xx{4}=X;

set(pp{1},'Vertices',xx{1});
set(pp{2},'Vertices',xx{2});
set(pp{3},'Vertices',xx{3});
set(pp{4},'Vertices',xx{4});

show_closest(cor,idz)
pause(0.1)
drawnow

end

function show_closest(cor,YY)
global Y;
global Z;
global X;
% kd = KDTreeSearcher(Y);
% idz = knnsearch(kd,Z);
% subplot(2,3,5)
% P = Y(idz,:);
%
% plot3(X(:,1)+1,X(:,2),X(:,3),'g.')
% hold on
% plot3(P(:,1),P(:,2),P(:,3),'bo')
% hold on
step=1:50:min(size(Y,1),size(Z,1));
for i=step
    %     plot3([X(i,1)+1 Y(idz(i),1)], [X(i,2) Y(idz(i),2)],[X(i,3) Y(idz(i),3)], '-', 'color',[1  0 0]);
    set(cor{i},'XData',[X(i,1)+1 YY(i,1)]);
    set(cor{i},'YData',[X(i,2) YY(i,2)]);
    set(cor{i},'ZData',[X(i,3) YY(i,3)]);
end

% drawnow;
end
%==========================================================================
%--------------------------------- DEMOS ----------------------------------
%==========================================================================
function R = tranR(t01,t02,t12)
cx = cos(t12);
cy = cos(t02);
cz = cos(t01);
sx = sin(t12);
sy = sin(t02);
sz = sin(t01);
R=[ cy*cz,  cz*sx*sy-cx*sz , cx*cz*sy+sx*sz;cy*sz,cx*cz+sx*sy*sz,cx*sy*sz-cz*sx; -sy , cy*sx , cx*cy];
end
function demo(srcfile,tarfile)
global X;
global Y;
global Z;
[X, Y] = demo1(srcfile,tarfile);
%[X Y] = mean_center_and_normalize(X,Y);
Z = X;
% show_contour(X,Y);
end
function [X, Y] = demo1(srcfile,tarfile)
% global Ex;
global MX;
% global MY;
global NY;
global NFY;
global NX;
global XF;
global YF;
global Ni;
% global Nr;
[v, f, n, ~] = cotlpvf(srcfile);
% size(v);
X = v;
NX = n;
dim = size(f',1);
F = f';
MX = sparse(dim*6 , size(X,1));
% Nr = sparse(dim*6 , size(X,1));
% Ni = sparse(dim*6 , 1);
ni=[F(:,[2,3,1]),F(:,[3,1,2])];
Ni=reshape(ni',[],1);
MX(sub2ind(size(MX),1:size(MX,1),Ni'))=-1;
ni=reshape(kron(F,[1,1])',[],1);
MX(sub2ind(size(MX),1:size(MX,1),ni'))=1;

% for i = 0:dim-1
%     x = F(i+1,1);
%     y = F(i+1,2);
%     z = F(i+1,3);
%     MX(i*6+1,x) = 1;
%     MX(i*6+1,y) = -1;
% %     Ni(i*6+1  ) = y;
% %     Nr(i*6+1,y  ) = 1;
%     
%     MX(i*6+2,x) = 1;
%     MX(i*6+2,z) = -1;
% %     Ni(i*6+2  ) = z;
% %     Nr(i*6+2,z) = 1;
%     
%     MX(i*6+3,y) = 1;
%     MX(i*6+3,x) = -1;
% %     Ni(i*6+3  ) = x;
% %     Nr(i*6+3,x) = 1;
%     
%     MX(i*6+4,y) = 1;
%     MX(i*6+4,z) = -1;
% %     Ni(i*6+4  ) = z;
% %     Nr(i*6+4,z) = 1;
%     
%     MX(i*6+5,z) = 1;
%     MX(i*6+5,x) = -1;
% %     Ni(i*6+5  ) = x;
% %     Nr(i*6+5,x) = 1;
%     
%     MX(i*6+6,z) = 1;
%     MX(i*6+6,y) = -1;
% %     Ni(i*6+6  ) = y;
% %     Nr(i*6+6,y) = 1;
% end
XF = f';
% fn=[];
% [v, f, vn] = read_obj(tarfile);
[v, f, vn, fn] = cotlpvf(tarfile);
size(v);
Y = v;
% F = f';
% dim = size(f',1);
%{
    MY = sparse(dim*6 , size(X,1));
    for i = 0:dim-1
        x = F(i+1,1);
        y = F(i+1,2);
        z = F(i+1,3);
        MY(i*6+1,x) = 1;
        MY(i*6+1,y) = -1;
        MY(i*6+2,x) = 1;
        MY(i*6+2,z) = -1;
        MY(i*6+3,y) = 1;
        MY(i*6+3,x) = -1;
        MY(i*6+4,y) = 1;
        MY(i*6+4,z) = -1;
        MY(i*6+5,z) = 1;
        MY(i*6+5,x) = -1;
        MY(i*6+6,z) = 1;
        MY(i*6+6,y) = -1;
    end
%}
%R = [0,-1,1;1,0,-1;-1,1,0];
%NY = n*R';
NY = vn;
NFY = fn;
YF = f';
%X = corrupt_data(X, [0 0 0],0,0, 0, 0);
%Y = corrupt_data(Y, [0 0 0 ],0,0, 0, 0);
end
%==========================================================================