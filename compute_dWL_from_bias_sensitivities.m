%computes derivatives w/rt weights for BP; SINGLE LAYER
function  [dWL,delta_L] = compute_dWL_from_bias_sensitivities(W,b_vec,phi_code,training_patterns,targets)

[K,P] =size(targets); %dim of output vec and num training patterns
[J,I] = size(W); %input vector dim I and num interneurons, J

delta_L = zeros(K,1);
dWL = W*0; 

%[y] = eval_1layer_fdfwdnet(W3,bvec_3,sigmoid_code,outputs_k)
%W
%b_vec
%training_patterns
[y] = eval_1layer_fdfwdnet(W,b_vec,phi_code,training_patterns);
err_vecs = y - targets;
phi_prime_vecs = fnc_phi_prime(phi_code,y); 

deltas_L = phi_prime_vecs.*err_vecs;
delta_L = sum(deltas_L,2);

W_L_of_p = deltas_L(:,1)*training_patterns(:,1)';
dWL = W_L_of_p;
    for p=2:P
        W_L_of_p = deltas_L(:,p)*training_patterns(:,p)';
        dWL = dWL + W_L_of_p;
    end