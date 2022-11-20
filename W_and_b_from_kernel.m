function [ W,b_vec ] = W_and_b_from_kernel( kernel_maps,kvec,b_maps,b_kernel)
%given kernel values and kernel maps, expand these into W and b_vec
%size W and b appropriately and init to 0's
[W_kernel_dim,dummy]=size(kvec);
[b_kernel_dim,dummy] = size(b_kernel);
[n_rows,n_cols] = size(kernel_maps{1});
W = zeros(n_rows,n_cols);
b_vec=0*b_maps{1};
%size(kernel_maps);
%size(b_maps)
for kk=1:W_kernel_dim
    W = W+kernel_maps{kk}*kvec(kk);
end
 for kk=1:b_kernel_dim
    b_vec = b_vec + b_maps{kk}*b_kernel(kk);
 end


end

