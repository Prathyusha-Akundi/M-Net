deep_structures = [58,26,54,18,50,11,53,17,52,13,51,12,49,10];
filepath = 'D:\My_files\IIITH\Research\M-Net\dataset\';
for i=1:18
    file_name_train = [filepath 'brain\nii_normalized\' num2str(i)];
    %file_name_train
    volume_nii_train = load_nifti(file_name_train);
    volume_train = volume_nii_train.img;
    volume_train = permute(volume_train,[1,3,2]);
    volume_train = volume_train(54:205,54:205,:);
    
    file_name_label = [filepath 'preprocessed_labels\nii\' num2str(i)];
    %file_name_label
    volume_nii_label = load_nifti(file_name_label);
    volume_label = volume_nii_label.img;
    volume_label = permute(volume_label,[1,3,2]);
    volume_label = volume_label(54:205,54:205,:);
    
    
    for j=25:103
        s = volume_train(:,:,j);
        s_l = volume_train(:,:,j-24:j);
        s_h = volume_train(:,:,j+1:j+25);
        s_v = cat(3, s_l,s,s_h);
        gt = volume_label(:,:,j);
        max(max(max(gt)))
        
        save(strcat('preprocessed_data\volumes\itrain',num2str(i),'_',num2str(j),'.mat'),'s_v');
        save(strcat('preprocessed_data\ground_truth\igt',num2str(i),'_',num2str(j),'.mat'),'gt');
    end
end




