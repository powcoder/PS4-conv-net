%computes selection/mapping matrices to help compute a Wc matrix that
%performs convolution.  Lots of assumptions embedded re/ architecture
%choice

%nC categories; kvec will be dim 2*nC; num outputs = num inputs-1 (dim patterns)
%kvec_dim = nC*2;
%creating selection maps for convolution is specific to the convolution pattern
function [kmaps,bmaps] = compute_conv_maps(kvec_dim,W3_input_dim,y_dim)
kmap = zeros(y_dim,W3_input_dim);
bmap = ones(y_dim,1); %all output neurons have same bias
%set up cells to hold all of the kernel injection maps
kmaps = cell(kvec_dim,1); %number of cells = number of kernel params

%more generally, may have multiple components to bk
bmaps = cell(1,1); %all outputs get the same bias
bmaps{1} = bmap; %all outputs get the same bias; only a single map is needed
%if more components to convolutional biases, loop through and define more bmaps

%fill in the maps for convolutional vector that defines synapse matrix
%init the cells:
for kk=1:kvec_dim
    kmaps{kk}=kmap; %so far, this is a properly dimensioned matrix, but just contains zeros
end

%construct the maps from kvec
nC = kvec_dim/2; %number of categories from kvec, assuming seeking 2-digit sequence
%populate a new map matrix

for imap = 1:kvec_dim
  kmap = zeros(y_dim,W3_input_dim);  %reset map to all zeros
  for irow=1:y_dim
      %FIX ME!!  place 1's in key locations for each map matrix, indicating the locations of kvec[imap]
  end
  kmaps{imap}=kmap; %store the new map in a cell, and repeat for the next kernel map
end

return
