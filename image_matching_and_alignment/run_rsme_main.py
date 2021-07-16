from sliding_window import rsme_main

error_threshold = 5 # average number of pixels...
target_image_path = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/targets/6ft_target/cropped_6_normal.jpg'

dist = 6

#FUSION
source_image_directory = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/outputs_6ft_target/'+str(dist)+' ft/fusion/coarse_warped_image_fusion/'
save_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/save_dir/6ft_target/fusion/'
rsme_main(source_image_directory, target_image_path, save_dir, error_threshold)


# COARSE
source_image_directory = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/outputs_6ft_target/'+str(dist)+' ft/coarse_warped_image_'+str(dist)+' ft/'
save_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/save_dir/6ft_target/coarse/'
rsme_main(source_image_directory, target_image_path, save_dir, error_threshold)


# HOMOGRAPHY
source_image_directory = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/outputs_6ft_target/'+str(dist)+' ft/homography_warped_image_'+str(dist)+' ft/'
save_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/save_dir/6ft_target/homography/'
rsme_main(source_image_directory, target_image_path, save_dir, error_threshold)

