%% NORDIC implementation for line-scanning with percentage signal removed (psr)
function [denoised_data,removed_perc] = nordic_psr(removing,data)

    data = squeeze(data);
    % singular values to be set to zero
    % amount = (1+round((10-removing)*size(data,2)/10):size(data,2)); 
    if removing>1
        rem_comp = (removing/100)*size(data,2);
    else
        rem_comp = removing*size(data,2);
    end

    % enforce integer
    rem_comp = int16(rem_comp);

    amount = (1+size(data,2)-rem_comp:size(data,2)); % with percentage
    % amount is an array which stores the indexes of the components to be
    % removed. If you want to remove a number of components = rem_comp you
    % have to calculate the amout indexes from total_comp-rem_comp+1 till the
    % total_comp. 

    removed_perc = rem_comp/size(data,2)*100;

    % singular value decomposition
    [U,S,V] = svd(data);
    svalues = diag(S);

    denoised_svalues = svalues;
    removed_comp = length(amount);
    denoised_svalues(amount) = 0; % set some of the singular values to zero
    % denoised_svalues(end-amount+1:end) = 0;

    denoised_S = diag(denoised_svalues);
    
    % add some zeros to the end of S matrix to keep the correct dimensions
    %denoised_S(:,size(denoised_svalues,1):end) = 0; %Catarina's way
    %Luisa's way
    if size(denoised_S,1)<=size(S,1) && size(denoised_S,2)<=size(S,2)
    new_denoised_S = zeros(size(S));
    new_denoised_S(1:size(denoised_S,1), 1:size(denoised_S,2)) = denoised_S;
    elseif size(denoised_S,1)>size(S,1)
        new_denoised_S = denoised_S;
        new_denoised_S(size(S,1)+1:end,:) = [];
    elseif size(denoised_S,2)>size(S,2)
        new_denoised_S = denoised_S;
        new_denoised_S(:,size(S,2)+1:end) = [];
    else
        disp('nessun caso trovato, scema')
    end
        
    % recalculating the Y data matrix with the thresholded singular values
    denoised_data = U*new_denoised_S*V';

end
