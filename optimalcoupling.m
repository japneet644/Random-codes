%  W= [0.1400    0.5000    0.0400    0.3200;
%     0.3500    0.3500    0.0300    0.2700;
%     0.2700    0.5300    0.0400    0.1600;
%     0.1100    0.2900    0.5600    0.0400;];
% 
% W= [ 0.3593    0.0833    0.4618    0.0956;
%     0.1858    0.0979    0.0956    0.6207;
%     0.3750    0.2072    0.3590    0.0588;
%     0.0836    0.2326    0.5309    0.1529;];


n = 3;
m = 4;
newW = true;
% [n, m] = size(W); % Determine the number of distributions


for iteration = 1:1
iteration 
if newW == true
    W = zeros(n, m); % Initialize the matrix
    for i = 1:n
        % Generate m-1 random weights for a row
        weights = log(rand(1, m)); 
    
        % Normalize the weights
        total = sum(weights);
        W(i, 1:m) = weights / total;
    end
end
    M = zeros(m*n, m^n);
    
    % Iterate through each column
    for col = 1:m^n
    % Convert column index to base m and pad with zeros
    baseMIndex = convertToBase(col-1, m, n);

    % Iterate through rows based on significant bits
    for bitPosition = 1:n
        row = (bitPosition - 1) * m + str2double(baseMIndex(bitPosition)) + 1;                            
        M(row, col) = 1;
    end
    end


    B = zeros(m, m^n); % Initialize the matrix with zeros

% Iterate through each columnfor k = 0:m-1
for k = 0:m-1
    for col = 1:m^n
        baseMIndex = convertToBase(col-1, m, n);
        % baseMIndex 
        %any(baseMIndex == num2str(k)) 
        % Optimized check if 'k' is present in baseMIndex
        if any(baseMIndex == num2str(k))  
            B(k + 1, col) = 1;
        end
    end
end




z= reshape(W', [], 1);
a = ones(1,m);
vecf = a*B;
geq = zeros(m^n,1);

% given a matrix W of size nxm and using the function convertFromBase(baseNumStr, m, n) set geq[convertfrombase('aaaa...aa') ] to mininum of ath column W(:,a) for all a in 1,...m
for a = 0:m-1
        baseNumStr = repmat(num2str(a), 1, n); % Create strings like "111...", "222..."

        % Convert to integer and adjust for indexing
        index = convertFromBase(baseNumStr, m) + 1; 

        % Find the minimum value in the corresponding column and store in geq
        geq(index) = min(W(:, a+1)); 
end
% geq = 0.*geq;

cvx_begin
    variables x(m^n,1) 
    % Define the objective function
    minimize vecf*x % + 0*sum(x)
    subject to
        M * x == z;
        x >= geq;
    cvx_end

    % disp([cvx_optval, maxdoeblin(W),max2doeblin(W),max3doeblin(W), mindoeblin(W)]) 
    [maxdoeblin(W), 2 - mindoeblin(W) ]
    
if abs(cvx_optval - max(maxdoeblin(W), 2 -mindoeblin(W) )) > 0.001
        disp('Condition failed!')
        disp(W)
        disp([cvx_optval, maxdoeblin(W),max2doeblin(W),max3doeblin(W), mindoeblin(W)]) 
        [maxdoeblin(W), 2 - mindoeblin(W) ]

        % break; % Stop the loop if the condition fails
end

  % Reshape the vector into a 4D array
  y = zeros(m*ones(1,n));

  for i = 1:length(x)
    baseNum = convertToBase(i - 1, m, n); % Decrement i for zero-based indexing
    indices = zeros(1, n);
    for j = 1:n
        indices(j) = str2num(baseNum(j)) + 1; % Convert character to number, add 1 
    end
    subscripts = num2cell(indices); % Convert indices to a cell array
    y(subscripts{:}) = x(i); % Use cell array for dynamic indexing
  end

sum1 =0;
for i = 1:m        
    for j = 1:m
        indices = i*ones(1, n);
        indices(n) = j;
        subscripts = num2cell(indices); % Convert indices to a cell array
        sum1 = sum1 + y(subscripts{:}); 
    end
end

sum2 =0;
for i = 1:m        
    for j = 1:m
        indices = i*ones(1, n);
        indices(n-1) = j;
        subscripts = num2cell(indices); % Convert indices to a cell array
        sum2 = sum2 + y(subscripts{:}); 
    end
end

% sum_a x(a,a,a,a)
sum4 = 0;
for i = 1:m        
    indices = i*ones(1, n);
    subscripts = num2cell(indices); % Convert indices to a cell array
    sum4 = sum4 + y(subscripts{:}); 
end


% Print the results (optional)
fprintf('sum_a(x(a,a,a,.)) = %f\n', sum1);
fprintf('sum_a(x(a,a,.,a)) = %f\n', sum2);
fprintf('last sum = %f\n', sum4);

 disp([cvx_optval, maxdoeblin(W),max2doeblin(W),max3doeblin(W), mindoeblin(W)]) 
        [maxdoeblin(W), 2 - mindoeblin(W) ]

end



% optimal_coupling = zeros(3);

function max_doeblin = maxdoeblin(W)
    max_colums_sums = max(W, [], 1);  % Find maximum value in each column
    max_doeblin = sum(max_colums_sums); % Sum of the maximum values 
end

% disp(optimal_coupling);
function baseNum = convertToBase(num, m, n)
    % Convert the number to base m notation
    baseNum = dec2base(num, m);
    
    % Pad with zeros if necessary
    baseNum = ['0' * ones(1, n - length(baseNum)), baseNum];
end
function originalNum = convertFromBase(baseNumStr, m)
    % Remove any leading zeros (added by convertToBase)
    while startsWith(baseNumStr, '0') % Check if the string starts with '0'
       baseNumStr = baseNumStr(2:end); % Remove the leading '0'
    end 

    % Convert each character digit back to its numerical value
    numDigits = str2num(baseNumStr(:))'; % Convert to column vector

    % Calculate the original decimal number using base conversion
    originalNum = sum(numDigits .* (m .^ (length(numDigits) - 1:-1:0))); 
end




function min_doeblin = mindoeblin(W)
    min_colums_sums = min(W, [], 1);  % Find maximum value in each column
    min_doeblin = sum(min_colums_sums); % Sum of the maximum values 
end


function max2_doeblin = max2doeblin(W)
    % Calculates the second-max-Doeblin norm of a matrix W

    [~, sorted_indices] = sort(W, 1, 'descend'); % Sort each column in descending order

    second_max_values = W(sub2ind(size(W), sorted_indices(2,:), 1:size(W,2)));
    max2_doeblin = sum(second_max_values); % Sum of the second maximum values
end

function max3_doeblin = max3doeblin(W)
    % Calculates the third-max-Doeblin norm of a matrix W

    [~, sorted_indices] = sort(W, 1, 'descend'); % Sort each column in descending order

    third_max_values = W(sub2ind(size(W), sorted_indices(3,:), 1:size(W,2)));
    max3_doeblin = sum(third_max_values); % Sum of the third maximum values
end
function max4_doeblin = max4doeblin(W)
    % Calculates the fourth-max-Doeblin norm of a matrix W

    [~, sorted_indices] = sort(W, 1, 'descend'); % Sort each column in descending order

    fourth_max_values = W(sub2ind(size(W), sorted_indices(4,:), 1:size(W,2)));
    max4_doeblin = sum(fourth_max_values); % Sum of the fourth maximum values
end



% A

% % Define optimization variables (dynamically)
% cvx_begin
%     variables x(n*columns^n)
%     maximize(sum(x)) 
%     subject to
%         A * x == W(:) % Reshape W into a column vector
%         x >= 0
% cvx_end
% 
% % Reshape solution for clarity
% optimal_coupling = reshape(x, n, columns^n);



