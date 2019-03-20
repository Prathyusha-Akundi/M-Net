deep_structures = [23,30,31,32,36,37,47,48,56,57,58,59,60,55];
filepath = 'D:\My_files\IIITH\Research\M-Net\dataset\MICCAI\';
for i=1:15
    file_name_train = [filepath 'brain\nii\' num2str(i)];
    file_name_train
    volume_nii_train = load_nifti(file_name_train);
    volume_train = volume_nii_train.img;
    %volume_train = permute(volume_train,[1,3,2]);
    
    file_name_label = [filepath 'labels_new\new\' num2str(i)];
    file_name_label
    volume_nii_label = load_nifti(file_name_label);
    volume_label = volume_nii_label.img;
    %volume_label = permute(volume_label,[1,3,2]);
    
    s = size(volume_train);
    len_s = s(3)-25;
    
    for j=25:len_s
        s = volume_train(:,:,j);
        s_l = volume_train(:,:,j-24:j);
        s_h = volume_train(:,:,j+1:j+25);
        s_v = cat(3, s_l,s,s_h);
        gt = volume_label(:,:,j);
        save(strcat('preprocessed_data\volumes\mtrain',num2str(i),'_',num2str(j),'.mat'),'s_v');
        save(strcat('preprocessed_data\ground_truth\mgt',num2str(i),'_',num2str(j),'.mat'),'gt');
    end
end





