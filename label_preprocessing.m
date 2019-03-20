deep_structures_list = [58,26,54,18,50,11,53,17,52,13,51,12,49,10];
%deep_structures_list = [23,30,31,32,36,37,47,48,55,56,57,58,59,60];
filepath = 'D:\My_files\IIITH\Research\M-Net\dataset\MICCAI\labels_new\new\';
list_files_dir = dir(filepath);
map = 1:14;
for i=1:15
   file = [filepath num2str(i)];
   
   volume_nii = load_nifti(file);
   volume = volume_nii.img;
   ismem = ~ones(size(volume));
   for j=1:length(deep_structures_list)
      ismem = or(ismem,volume==deep_structures_list(j)); 
   end
   volume(~ismem) = 0;
   volume = map_intensities(volume,deep_structures_list,map);
   output_path = ['D:\My_files\IIITH\Research\M-Net\dataset\MICCAI\preprocessed_labels\' num2str(i)];
   volume_nii.img = volume;
   save_nifti(volume_nii, output_path);
   
end


