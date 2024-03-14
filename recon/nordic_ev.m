%% NORDIC implementation for line-scanning with explained variance or wirth eigenvalues plot
function [denoised_data, removed_perc] = nordic_ev(data)

    data = squeeze(data);
   
    % singular value decomposition
    [U,S,V] = svd(data);
    svalues = diag(S); %the diagonal contains the eigenvalues 
    eig_elbow = find_elbow(svalues);
    %figure, plot(svalues); xlabel('componets'); ylabel('eigenvalues'); title('scree plot'); set(gca,'box','off');

    var_exp = cumsum(svalues)/sum(svalues);
    elbow_idx = find_elbow(var_exp);
    %figure, plot(var_exp); xlabel('componets'); ylabel('explained variance');
        
    % singular values to be set to zero
    amount = (eig_elbow:size(data,2)); %with eigenvalues elbow
    removed_comp = length(amount);
    removed_perc = removed_comp/size(data,2)*100;
    %amount = (length(svalues) - elbow_idx):size(data,2); %with explained
    %variance elbow
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

