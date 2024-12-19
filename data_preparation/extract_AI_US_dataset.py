import glob
import os

import SimpleITK as sitk
import cv2
import numpy as np
from natsort import natsorted


def subsample_segmentation(in_data, dim_size=None):
    """Takes an input image (no color channel) and resamples it to a desired size. It can be expanded to work with
    any number of dimensions. """
    dims = np.shape(in_data)
    x_val, y_val = np.mgrid[0:dims[0] - 1:complex(0, dim_size[0]),
                   :dims[1] - 1:complex(0, dim_size[1])].astype(int)
    return in_data[x_val, y_val]


def load_img(img_path, img_out_size=None):
    # print(img_path)
    if img_path.split('.')[-1]=='png':
        image_array = cv2.imread(img_path)[:, :, 0]
    else:
        image_array = np.load(img_path)
    return subsample_segmentation(image_array, dim_size=img_out_size)


def generate_save_path(save_folder_path, folder_type, save_patient_name, save_selected_view_name, save_train_test_type):
    images_save_folder = os.path.join(save_folder_path, folder_type, save_train_test_type)
    os.makedirs(images_save_folder, exist_ok=True)
    out_save_path = os.path.join(images_save_folder,
                                 '_'.join([save_patient_name, save_selected_view_name, save_train_test_type]) + '.png')
    return out_save_path

def match_CT_US_file_list(CT_list, US_list, record):
    new_US_list = []
    new_CT_list = []
    for i in range(len(CT_list)):
        id = CT_list[i].split('CT_slice_')[-1].split('raw')[0]
        US_file = os.path.join(record, 'UltrasoundImages', id + '.png')
        if US_file in US_list:
            new_US_list.append(US_file)
            new_CT_list.append(CT_list[i])
    return new_US_list, new_CT_list
        
        
def save_dataset_list(in_data_folder, save_folder, img_save_size, 
                        data_list, list_name):
    # iterate over patient
    for patient in data_list:
        patient_folder = os.path.join(in_data_folder, patient)
        records = natsorted(glob.glob(os.path.join(patient_folder, 'Linear18', 'record*')))
        for record in records:
            # print(record)
            US_files = glob.glob(os.path.join(record, 'UltrasoundImages', '*.png'))
            CT_files = glob.glob(os.path.join(record, 'US_simulation', 'CT_slice', '*.npy'))
            # match us with ct
            matched_US_files, matched_CT_files = match_CT_US_file_list(CT_files, US_files, record)
            
            record_name = record.split('/')[-1]
            for i in range(len(matched_US_files)):
                loaded_img = load_img(matched_US_files[i], img_save_size)
                loaded_gt_img = load_img(matched_CT_files[i], img_save_size)
                
                id = matched_US_files[i].split('/')[-1].split('.')[0]
                # patient_img_file_name = patient + '_' + record_name + id + '.png'
                # patient_gt_file_name = patient + '_' + record_name + id + '_gt.png'


                img_save_path = generate_save_path(save_folder, 'images', patient, record_name + '_' + id,
                                                list_name)
                annotations_save_path = generate_save_path(save_folder, 'annotations', patient, record_name + '_' + id,
                                                        list_name)

                cv2.imwrite(img_save_path, loaded_img)
                cv2.imwrite(annotations_save_path, loaded_gt_img)
            print('Done with {} patients for record {}'.format(patient, record_name))


def save_all_imgs(in_data_folder, save_folder, img_save_size,
                        train_list, val_list, test_list):
    
    save_dataset_list(in_data_folder, save_folder, img_save_size,train_list,'training')
    save_dataset_list(in_data_folder, save_folder, img_save_size,val_list,'val')
    save_dataset_list(in_data_folder, save_folder, img_save_size,val_list,'test')


if __name__ == '__main__':

    camus_data_folder = r'data_preparation/AI_Ultrasound_dataset'
    save_folder_path = r'data_preparation/AI_Ultrasound_processed'
    save_img_size = (256, 256)
    
    train_list = ['cadaver01_F231091',
                  'cadaver03_S231783',
                  'cadaver05_S232132L',
                  'cadaver06_S231987',
                  'cadaver10_S232098L',
                  'cadaver11_S232110',
                  'cadaver12_S240174',
                  'cadaver13_S232110L',
                  'cadaver14_S240280',
                  'cadaver04_F231091',
                    'cadaver09_S231989R']
    val_list = ['cadaver02_F231218',
                  'cadaver07_S232132R',
                  'cadaver08_S231989L',]
    test_list = [
                  'cadaver02_F231218',
                  'cadaver07_S232132R',
                  'cadaver08_S231989L',
                ]
    
    data_folder = camus_data_folder

    save_all_imgs(data_folder, save_folder_path, save_img_size,
                        train_list=train_list, val_list=val_list, test_list=test_list)