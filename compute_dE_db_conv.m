function [dE_db_conv] = compute_dE_db_conv(bmaps,delta_cum)
%given cumulative delta vector, compute sensitivities with respect to
%kernel parameters of b_vec by using b_vec mappings
mapmat =[];
[n_bmaps,dummy] = size(bmaps);
for kk=1:n_bmaps
    %bmaps{kk}
    %SOH(kmaps{kk})
    mapmat = [mapmat;bmaps{kk}'];
end
%mapmat
dE_db_conv = mapmat*delta_cum;
end

