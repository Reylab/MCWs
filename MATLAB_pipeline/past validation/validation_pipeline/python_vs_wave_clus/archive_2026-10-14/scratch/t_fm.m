rng(0); L=64; F=10;
t1=-sin(linspace(0,pi,L))*50; t2=-sin(linspace(0,2*pi,L))*40;
f_in=[t1+5*randn(200,L); t2+5*randn(200,L)]; cls=[ones(1,200) 2*ones(1,200)];
f_out=[t1+6*randn(100,L); t2+6*randn(100,L); 30*randn(50,L)];
P=randn(L,F); fi=f_in*P; fo=f_out*P;
par.template_type='center'; par.template_sdnum=3;
a=force_membership_wc(f_in,cls,f_out,par);
b=force_membership_wc(f_in,cls,f_out,par,fi,fo);
c=force_membership_wc(f_in,cls,f_out,par,fi(1:10,:),fo);
fprintf('wave: %d acc | feat: %d acc | same accept set: %d | differ labels: %d | fallback==wave: %d\n', ...
  nnz(a),nnz(b),isequal(a>0,b>0),nnz(a~=b),isequal(a,c));
