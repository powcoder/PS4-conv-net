function [W1,bvec_1,W2,bvec_2]=compute_preset_weights(xmin,xmax,Npreset_nodes,I_input)
%make a network that maps input value into Npreset_nodes ranges
%Npresent nodes is the number of categories, e.g. integers: 0,1,2 or 3
%I_input is dimension of input vector

%use this trick to recognize Npreset_nodes categories for a range of scalar inputs, xmin to xmax
dx=(xmax-xmin)/(Npreset_nodes-1);
nC = Npreset_nodes; %synonym: number of categories for each input
%set weights on inputs manually
w_scale = 2/dx;
W1kernel = w_scale*ones(Npreset_nodes,1)
bkvec_1 = w_scale*(-xmin*ones(Npreset_nodes,1)-dx*(0:Npreset_nodes-1)')
%kvec_1 = w_scale*ones(nC:1)
%now, stack bkvec_1 values I_input times:
bvec_1 = [];
for i=1:I_input;
  bvec_1 = [bvec_1;bkvec_1];
end

%now, build W1 using W1kernel as a template:
W1 = zeros(Npreset_nodes*I_input,I_input);

for icol = 1:I_input
   jrow = 1+(icol-1)*nC;
     W1((jrow:jrow+nC-1),icol)=W1kernel;

 end
% W1
%now have fanned out I_input values into nC*I_input values;
%do a second layer of processing on these values, in groups of 4, to make an RBF-like response of nC categories per input
%thus, this matrix W2 is square, but it is comprised of replications of W_kernel nCxnC


w_scale2 = 2;
W2_kernel = eye(Npreset_nodes,Npreset_nodes)
for i=1:Npreset_nodes-1
    W2_kernel(i,i+1)=-2;
end
W2_kernel = 20*W2_kernel;
%pause

bkvec_2 = -4*ones(Npreset_nodes,1)

%build the bias vector as replications of bkvec_2:
bvec_2 = [];
for i=1:I_input;
  bvec_2 = [bvec_2;bkvec_2];
end
%now populate W2 as block diagonal
W2 = zeros(I_input*nC,I_input*nC);
W2(1:nC,1:nC)=W2_kernel;
%keep populating W2:
for icol=nC+1:nC:nC*I_input
  jrow=icol;
  %jrow = (icol-1)*nC
    W2((jrow:jrow+nC-1),(icol:icol+nC-1))=W2_kernel;
end



