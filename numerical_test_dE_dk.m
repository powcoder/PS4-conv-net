function [dE_dkvec,dE_dbias_param] = numerical_test_dE_dk(kmaps,kvec,bmaps,b_conv,phi_code,patterns,targets)
%vary kernel components 1 at a time and estimate derivatives numerically
%start w/ given W's and b's:
%compute W's and b's from kernels:
[W,b_vec] = W_and_b_from_kernel(kmaps,kvec,bmaps,b_conv);

%eval total error penalty for these weights over all training patterns
[rmserr,esqd0] = err_eval(W,b_vec,phi_code,patterns,targets);

%now perturb individual components
eps = 0.000001;

[dim_b_conv,dummy]=size(b_conv) %start w/ bias params, layer 1
dE_dbias_param = zeros(dim_b_conv,1);
for kk=1:dim_b_conv
    kvec_temp = b_conv;
    kvec_temp(kk)=kvec_temp(kk)+eps;
    [W,b_vec] = W_and_b_from_kernel(kmaps,kvec,bmaps,kvec_temp);
    [rmserr,esqd] = err_eval(W,b_vec,phi_code,patterns,targets);
    dE_dbias_param(kk) = (esqd-esqd0)/eps;
end
%dE_dbias_param
%now compute sensitivities of bias params for layer 2:
[W,b_vec] = W_and_b_from_kernel(kmaps,kvec,bmaps,b_conv);


%compute the sensitivies of the kernel params:
[dim_k_conv,dummy]=size(kvec); 
dE_dkvec = zeros(dim_k_conv,1);
for kk=1:dim_k_conv
    kvec_temp = kvec;
    kvec_temp(kk)=kvec_temp(kk)+eps;
    [W,b_vec] = W_and_b_from_kernel(kmaps,kvec_temp,bmaps,b_conv);
    [rmserr,esqd] = err_eval(W,b_vec,phi_code,patterns,targets);
    dE_dkvec(kk) = (esqd-esqd0)/eps;
end
%dE_dkvec
