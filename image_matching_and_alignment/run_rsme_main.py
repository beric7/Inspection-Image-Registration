from sliding_window_select import rsme_main

error_threshold = 3 # average number of pixels...
dist = 10
dist_2 = 4
target_image_path = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/targets/'+str(dist_2)+'ft_target/cropped_'+str(dist_2)+'_normal.jpg'

'''
#FUSION
source_image_directory = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/outputs_6ft_target/'+str(dist)+' ft/fusion/coarse_warped_image_fusion/'
save_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/save_dir/6ft_target/fusion/'
rsme_main(source_image_directory, target_image_path, save_dir, error_threshold, resize=True, resize_shape=(1500,840), set_min_n=12, set_max_step=375)


# COARSE
source_image_directory = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/outputs_6ft_target/'+str(dist)+' ft/coarse_warped_image_'+str(dist)+' ft/'
save_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/save_dir/6ft_target/coarse/'
rsme_main(source_image_directory, target_image_path, save_dir, error_threshold, resize=True, resize_shape=(1500,840), set_min_n=12, set_max_step=375)
'''

# HOMOGRAPHY
source_image_directory = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/outputs_'+str(dist_2)+'ft_target/'+str(dist)+' ft/homography_warped_image_'+str(dist)+' ft/'
save_dir = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/6-18-2020-lab_test/save_dir/'+str(dist_2)+'ft_target/homography/'
rsme_main(source_image_directory, target_image_path, save_dir, error_threshold, resize=True, resize_shape=(1500,840), number_x=4)

