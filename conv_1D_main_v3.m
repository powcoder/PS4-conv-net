clear all
sigmoid_code = 1; %code=1 implies use sigmoid, i.e. logsig

% use two layers of preprocessing with static synapses:
N_categories = 4  %considering only integer values 0,1,2,3; could make this larger
xmin=0;  %make preset layers recognize digits 0 through 3, so xmin=0, xmax=3
xmax=3;  %if change number of categories, change this too

phi1_code=sigmoid_code; %preset layers use sigmoids
phi2_code=sigmoid_code;

I_pattern_size=5  %dimension of a stimulus vector; change this as desired; number of digits in an input
I = I_pattern_size; %synonym
K = I_pattern_size-1 %number of output dimensions of y_vec chosen to be 1 less than input dim
        %desired outputs are 1 for y(i) if p(i),p(i+1) match desired
        %sequence
        
N = 8; %number of training patterns= batch size; can change this
[patterns,targets] = get_training_data(N,I_pattern_size,K)

[W1_presets,bvec_1_presets,W2_presets,bvec_2_presets]=compute_preset_weights(xmin,xmax,N_categories,I_pattern_size)

%the next set of inputs is merely to illustrate the receptive fields of the preset network
test_inputs_x=ones(I_pattern_size,1)*(xmin:0.02:xmax);
%evaluate the outputs of the first two layers, which use static synapses
[outputs_j,outputs_k]=eval_2layer_fdfwdnet(W1_presets,bvec_1_presets,phi1_code,W2_presets,bvec_2_presets,phi2_code,test_inputs_x);
%plot out responses of preset layers--just for visualization; not needed for training the network
figure(1)
for i=1:N_categories
 plot(test_inputs_x(1,:),outputs_j(i,:))
 hold on
 end
title('premapped nodal outputs j')
xlabel('x')
ylabel('node output')
figure(2)
for i=1:N_categories
 plot(test_inputs_x(1,:),outputs_k(i,:))
 hold on
 end
title('premapped nodal outputs k')
xlabel('x')
ylabel('node output')



[outputs_j,outputs_k]=eval_2layer_fdfwdnet(W1_presets,bvec_1_presets,phi1_code,W2_presets,bvec_2_presets,phi2_code,patterns)
kvec_3 = 2*rand(2*N_categories,1)-1; %consider two groups of N_categories to interpret 2 sequential values    
[kvec_dim,dummy] = size(kvec_3);% = 2*N_categories; 
bkvec_3 = 2*rand(1,1)-1; %every output neuron gets the same bias
[N_inputs_3,dummy] = size(outputs_k) %number of inputs to adaptive layer3 (last layer)

[W3,bvec_3]=set_weights_from_kernel(N_categories,N_inputs_3,kvec_3,bkvec_3) 
%this computes the outputs of layer 3, as fed by outputs from layer2 (preset network outputs, outputs_k)
[y] = eval_1layer_fdfwdnet(W3,bvec_3,sigmoid_code,outputs_k) 
%compute sensitivities of layer-3 synapses using bias-sensitivity vector recursive computation
[dWL,delta_L] = compute_dWL_from_bias_sensitivities(W3,bvec_3,sigmoid_code,outputs_k,targets)

[y_dim,N_inputs_layer3] = size(W3);  %the preprocessing layers result in this many neural outputs
[kvec_dim,dummy]=size(kvec_3)
%compute the map matrices that identify locations were kernel parameters are repeated in the output layer
[kmaps,bmaps] = compute_conv_maps(kvec_dim,N_inputs_layer3,y_dim)

%get training data: 
[patterns,targets] = get_training_data(N,I,K)

%compute sensitivities w/rt kernel params:
%next line is redundant with above; should delete it
[dWL,delta_L] = compute_dWL_from_bias_sensitivities(W3,bvec_3,sigmoid_code,outputs_k,targets)
%use dWL to compute sensitivities with respect to kernel parameters 
[dE_dkvec] = compute_dE_dkvec(kmaps,dWL)
%do the same with respect to templated bias offsets
[dE_db_conv_L] = compute_dE_db_conv(bmaps,delta_L)
%numerical estimates of the above sensitivities:
%use these to debug compute_dE_dkvec and compute_dE_db_conv
[est_dE_dkvec,est_dE_dbias_param]=numerical_test_dE_dk(kmaps,kvec_3,bmaps,bkvec_3,sigmoid_code,outputs_k,targets)

display("paused")
pause  %get rid of this pause after debugging


%here is the loop where learning takes place:
eta = 0.0005 %CHOOSE ME

N_target_iterations = 100  %arbitrary; number of BP iterations on a given batch of training data
for iter=1:500  %arbitrary; number of outer loops, i.e. number of batches considered
  
   [patterns,targets] = get_training_data(N,I,K); %get some new training data
   %re-use these targets for N_target_iter iterations of BP
   %responses of first two layers will not change until there are new input patterns
   [outputs_j,outputs_k]=eval_2layer_fdfwdnet(W1_presets,bvec_1_presets,phi1_code,W2_presets,bvec_2_presets,phi2_code,patterns);

    %but W3 values will change every iteration of BP
   [W3,bvec_3]=set_weights_from_kernel(N_categories,N_inputs_3,kvec_3,bkvec_3); 
   display("max err before iterations on this pattern:")   
   [y_vec] = eval_1layer_fdfwdnet(W3,bvec_3,sigmoid_code,outputs_k);
   errs = y_vec -targets; 
   max_err=max(max(abs(errs)))
    %compute all sensitivities, first w/ respect to full W and bias
    for iBP=1:N_target_iterations
      [W3,bvec_3]=set_weights_from_kernel(N_categories,N_inputs_3,kvec_3,bkvec_3); 
      [dWL,delta_L] = compute_dWL_from_bias_sensitivities(W3,bvec_3,sigmoid_code,outputs_k,targets);
      [dE_dkvec] = compute_dE_dkvec(kmaps,dWL);
      [dE_db_conv_L] = compute_dE_db_conv(bmaps,delta_L);
     
       %learning: update the kernel params for both L-layer biases and L-layer synapses:
       bvec_3 = bvec_3 - eta*dE_db_conv_L;
       kvec_3 = kvec_3 -eta*dE_dkvec;
     end
     display("errors after N iterations on this input pattern: ");
       [rmserr,esqd] = err_eval(W3,bvec_3,sigmoid_code,outputs_k,targets) % uses output of 2-layer as input to third layer
       [y_vec] = eval_1layer_fdfwdnet(W3,bvec_3,sigmoid_code,outputs_k)
       errs = y_vec -targets 
       max_err=max(max(abs(errs)))  %worst-case error will detect any misclassifications of identifying sequences of interest
       kvec_3'
       bvec_3'
end

%done with learning; now evaluate the network on various inputs:
max_of_max_err =0
for i=1:100
  display("test result: ")
  [patterns,targets] = get_training_data(N,I,K)

    [outputs_j,outputs_k]=eval_2layer_fdfwdnet(W1_presets,bvec_1_presets,phi1_code,W2_presets,bvec_2_presets,phi2_code,patterns);
    [W3,bvec_3]=set_weights_from_kernel(N_categories,N_inputs_3,kvec_3,bkvec_3); 
    [rmserr,esqd] = err_eval(W3,bvec_3,sigmoid_code,outputs_k,targets); % uses output of 2-layer as input to third layer
    [y_vec] = eval_1layer_fdfwdnet(W3,bvec_3,sigmoid_code,outputs_k);
       errs = y_vec -targets 
       max_err=max(max(abs(errs)))
  
       patterns
       targets
       y_vec
       errs = y_vec -targets 
       max_err=max(max(abs(errs)))
       if (max_err> max_of_max_err) 
         max_of_max_err =max_err
       end
       %display("paused: ");
       %pause
end
max_of_max_err  %for all of the above, check if ANY sequences were mis-identified


