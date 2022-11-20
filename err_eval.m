%evaluate rms error for all training patterns (rows in training_patterns)
% and targets
% this is specialized for a single (output) layer;
%  may need to compute earlier layers separately before running this
function [rmserr,esqd] = err_eval(W,b_vec,phi_code,training_patterns,targets)
[noutputs,npats] =size(targets);
%esqd=0;
%evaluate all output errors
 [y_vecs] = eval_1layer_fdfwdnet(W,b_vec,phi_code,training_patterns);

   errvecs=y_vecs-targets; %column vectors of y-t
   sqd_errs = errvecs.*errvecs;
   esqd= 0.5*sum(sum(sqd_errs));  %1/2 sum squared errors; 
   rmserr=sqrt(esqd/npats);
