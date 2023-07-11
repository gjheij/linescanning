%% NORDIC implementation for line-scanning with percentage signal removed (psr)
function [denoised_data] = nordic_psr(removing,data)

    data = squeeze(data);
    % singular values to be set to zero
    amount = (1+round((10-removing)*size(data,2)/10):size(data,2)); 
    
    % singular value decomposition
    [U,S,V] = svd(data);
    svalues = diag(S);

    denoised_svalues = svalues;
    denoised_svalues(amount) = 0; % set some of the singular values to zero

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

