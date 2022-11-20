%use synapse sensitiviy matrix, dE_dW, and kernal maps to derive sensitivity of kernel params, dE_dkvec
function [dE_dkvec] = compute_dE_dkvec(kmaps,dE_dW)
mapmat =[];
[n_kmaps,dummy] = size(kmaps);
for kk=1:n_kmaps
    %kmaps{kk}
    SOH(kmaps{kk}); %get a kernel map and string it out horizontally
    mapmat = [mapmat;SOH(kmaps{kk})]; %construct a matrix in which the rows are SOH's of kernel maps
end

dE_dkvec = 0; %FIX ME!!  compute this with a 1-liner using the above


