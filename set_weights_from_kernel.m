%make a network that maps input vector of dimension N_inputs onto
% next layer with N_output_neurons outputs
%  W3 and bvec_3 are templated using kvec_3 and bkvec_3
function [W3,bvec_3]=set_weights_from_kernel(N_categories,N_inputs,kvec_3,bkvec_3)
N_output_neurons = N_inputs/N_categories-1; %outputs indicate "found sequence here" or "did not find sequence here"
%potential locations for a 2-component sequence are 1 through N_inputs-1
%number of inputs to this W3 layer will be number of inputs * number of categories
W3kernel = kvec_3'; %make this a row vector
W3 = zeros(N_output_neurons,N_inputs);

bvec_3 = bkvec_3*ones(N_output_neurons,1); %for this example, all output neurons share the same bias value

%build W1 using W1kernel as a template:
for irow = 1:N_output_neurons
  %FIX ME!!  this is a nuisance; need to get all the kvec_3 parameters in the right locations in W3(i,j)
  %can be done in 1 or 2 lines...but can be frustrating

end

return

