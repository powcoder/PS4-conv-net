%generates random sequences and sets targets corresponding to specific
%sequence of interest; should generate these once and dump them to file,
%but this is useful for testing/visualization with small data sets
function [patterns,targets] = get_training_data(N,I,K)
%these pattern elements are integers 0 through 3
patterns = round(3*rand(I,N)); %set up training patterns and targets
targets = zeros(K,N); %must make targets consistent w/ training patterns
%target(i) is "true" if p(i),p(i+1) = (3,1)
%i.e., look for sequence (3,1)
for n=1:N
    pattern = patterns(:,n);
    target = zeros(K,1);
    for j=1:I-1
        if (pattern(j)==3)&&(pattern(j+1)==1)
            target(j)=1;
        end
    end
    targets(:,n)=target;
end