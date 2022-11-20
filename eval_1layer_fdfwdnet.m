%helper fnc to evaluate outputs of a 2-layer fdfwd net;
%this version treats the bias inputs as a separate vector (instead of
%including a dummy neuron of output "1") 
function [outputs]=eval_1layer_fdfwdnet(W,b_vec,phi1_code,stimuli)
[ninputs,npats] = size(stimuli);
u_vecs = W*stimuli+b_vec*ones(1,npats);
outputs= fnc_phi(phi1_code,u_vecs); 
