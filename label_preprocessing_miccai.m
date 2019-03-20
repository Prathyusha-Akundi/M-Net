deep_structures_list_miccai = [23,30,31,32,36,37,47,48,55,56,57,58,59,60];
deep_structures_list_isbr = [58,26,54,18,50,11,53,17,52,13,51,12,49,10];


filepath = 'D:\My_files\IIITH\Research\M-Net\dataset\MICCAI\labels\nii\';
list_files_dir = dir(filepath);

for i=1:15
   file = [filepath num2str(i)];
   
   volume_nii = load_nifti(file);
   volume = volume_nii.img;
   
   for j=1:length(deep_structures_list_miccai)
      iseq = volume == deep_structures_list_miccai(j);
      volume(iseq) = deep_structures_list_isbr(j);
   end
   
   output_path = ['D:\My_files\IIITH\Research\M-Net\dataset\MICCAI\labels_new\new\' num2str(i)];
   volume_nii.img = volume;
   save_nifti(volume_nii, output_path);
   
end