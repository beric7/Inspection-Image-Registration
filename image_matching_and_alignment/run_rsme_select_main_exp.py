from sliding_window_select import rsme_main
import os

error_threshold = 5 # average number of pixels...

beam_iteration = 'beam_2_frames_source_many_output_l2/'
base = 'C://Users/Admin/OneDrive - Virginia Tech/Documents/data_image_registration/ZED_captures_outdoor_beams/'

target_dir = base + 'beam_2_targets_reversed_many_frames/'
target_files = os.listdir(base + 'beam_2_targets_reversed_many_frames/')
count = 0

for folder in os.listdir(base + 'every_1_frame/' + beam_iteration):
    print (count)
    print
    target_image_path = target_dir + target_files[count]
    '''
    #FUSION
    source_image_directory = base + 'every_1_frame/' + beam_iteration + folder + '/fusion_warped_image/'
    save_dir = base + 'every_1_frame/' + beam_iteration + folder + '/fusion_error/'
    rsme_main(source_image_directory, target_image_path, save_dir, error_threshold, resize=True, resize_shape=(1080,600), number_x=12)
    
    
    # COARSE
    source_image_directory = base + 'every_1_frame/' + beam_iteration + folder + '/coarse_warped_image/' 
    save_dir = base + 'every_1_frame/' + beam_iteration+folder + '/coarse_error/'
    rsme_main(source_image_directory, target_image_path, save_dir, error_threshold, resize=True, resize_shape=(1080,600), number_x=12)
    '''
    
    # HOMOGRAPHY
    source_image_directory = base + 'every_1_frame/' + beam_iteration + folder + '/homography_warped_image/'
    save_dir = base + 'every_1_frame/' + beam_iteration + '/' + folder + '/homography_error/'
    rsme_main(source_image_directory, target_image_path, save_dir, error_threshold, resize=True, resize_shape=(1080,600), number_x=12)
    
    count += 1


