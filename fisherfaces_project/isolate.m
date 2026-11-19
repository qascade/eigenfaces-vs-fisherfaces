% Does PCA do better on Yale once illumination is held near-constant?
% Use the first 32 Yale subjects so c matches ORL-32 exactly.
rng(0);
yaleRoot='../CroppedYale/'; dY=192*168; orlRoot='../ORL/'; dO=112*92;
dy=dir(yaleRoot); YF={};
for i=1:numel(dy), if dy(i).isdir && dy(i).name(1)~='.', YF{end+1}=[yaleRoot dy(i).name '/']; end, end
YF=YF(1:32);                              % 32 subjects, same c as ORL-32

runs = { 'ORL-32  all 10 imgs (pose varies)',      'orl',  6, 999; ...
         'Yale-32 light <=12 deg  (7 imgs)',       'yale', 4,  12; ...
         'Yale-32 light <=25 deg  (19 imgs)',      'yale', 6,  25; ...
         'Yale-32 light <=50 deg  (29 imgs)',      'yale', 6,  50; ...
         'Yale-32 ALL lighting    (64 imgs)',      'yale', 6, 999 };

fprintf('%-36s %4s %5s %5s  %6s %8s\n','configuration','n_tr','N','test','eig%','fisher%');
for r=1:size(runs,1)
  nm=runs{r,1}; kind=runs{r,2}; nTr=runs{r,3}; maxAng=runs{r,4};
  if strcmp(kind,'orl'), d=dO; else d=dY; end
  rng(0); Xtr=[];ytr=[];Xte=[];yte=[];
  for p=1:32
    if strcmp(kind,'orl')
      f=dir([orlRoot 's' num2str(p) '/*.pgm']); base=[orlRoot 's' num2str(p) '/']; keep=1:numel(f);
    else
      f=dir([YF{p} '*.pgm']); base=YF{p}; keep=[];
      for j=1:numel(f)
        t=f(j).name; ia=strfind(t,'A'); ie=strfind(t,'E'); ia=ia(end); ie=ie(end);
        A=str2double(t(ia+1:ie-1)); E=str2double(t(ie+1:end-4));
        ang=acosd(cosd(A)*cosd(E));
        if ang<=maxAng, keep(end+1)=j; end
      end
    end
    if strcmp(kind,'yale'), keep=keep(randperm(numel(keep))); end
    for q=1:numel(keep)
      im=double(imread([base f(keep(q)).name]));
      if q<=nTr, Xtr=[Xtr im(:)]; ytr(end+1)=p; else Xte=[Xte im(:)]; yte(end+1)=p; end
    end
  end
  N=size(Xtr,2); c=32;
  xbar=mean(Xtr,2); X=Xtr-xbar; Xt=Xte-xbar;
  L=X'*X;[W,D]=eig(L);[ev,ix]=sort(diag(D),'descend');W=W(:,ix);
  V=X*W; s1=norm(V(:,1));
  for l=1:size(V,2), nr=norm(V(:,l)); if nr>1e-10*s1, V(:,l)=V(:,l)/nr; else V(:,l)=0; end, end
  be=0;
  for k=[10 20 30 50 75 100 150]
    if k<=size(V,2)
      rr=nnc(V(:,1:k)'*X,ytr,V(:,1:k)'*Xt,yte); if rr>be, be=rr; end
    end
  end
  m=N-c; Vm=V(:,1:m); Al=Vm'*X; ab=mean(Al,2);
  Sw=zeros(m);Sb=zeros(m);
  for j=1:c
    Aj=Al(:,ytr==j);nj=size(Aj,2);mj=mean(Aj,2);Dj=Aj-mj;
    Sw=Sw+Dj*Dj';Sb=Sb+nj*(mj-ab)*(mj-ab)';
  end
  Sw=(Sw+Sw')/2;Sb=(Sb+Sb')/2;
  [R,pp]=chol(Sw); if pp~=0, Sw=Sw+1e-6*trace(Sw)/m*eye(m); R=chol(Sw); end
  M=(R')\(Sb/R);M=(M+M')/2;[Z,Dz]=eig(M);[~,iz]=sort(diag(Dz),'descend');
  Wf=Vm*(R\Z(:,iz(1:c-1)));
  for l=1:size(Wf,2), nr=norm(Wf(:,l)); if nr>1e-10, Wf(:,l)=Wf(:,l)/nr; end, end
  rf=nnc(Wf'*X,ytr,Wf'*Xt,yte);
  fprintf('%-36s %4d %5d %5d  %6.1f %8.1f\n', nm, nTr, N, size(Xte,2), be, rf);
end

function r=nnc(A,ya,B,yb)
  cr=0; for i=1:size(B,2)
    [~,k]=min(sum((A-B(:,i)).^2,1)); if ya(k)==yb(i), cr=cr+1; end
  end
  r=100*cr/size(B,2);
end
