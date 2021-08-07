import os
import shutil
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from collections import Iterable


def is_dicom(file_path):
    
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
        
    except:
        return False


def get_modality(dicom_data, key=('0008', '0060')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return ''

    
def get_body_part(dicom_data, key=('0018', '0015')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 


def get_study_desc(dicom_data, key=('0008', '1030')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 


def get_series_desc(dicom_data, key=('0008', '103e')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 
    
    
def get_protocol_name(dicom_data, key=('0018', '1030')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 
    

def get_plane(dicom_data, key=('0020', '0037')):
    if key in dicom_data:
        iop = dicom_data[key].value
        z = np.cross(iop[0:3], iop[3:6])
        arg_max = np.argmax(np.abs(z))
        if arg_max == 0:
            return "Sagittal"
        elif arg_max == 1:
            return "Coronal"
        elif arg_max == 2:
            return "Axial"
    else:
        return ''


def get_wc(dicom_data, key=('0028','1050')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 

    
def get_ww(dicom_data, key=('0028','1051')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 


def get_intercept(dicom_data, key=('0028','1052')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 

    
def get_slope(dicom_data, key=('0028','1053')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 


def get_thickness(dicom_data, key=('0018','0050')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 
    
    
def get_instance_number(dicom_data, key=('0020','0013')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 
    

def get_institution_name(dicom_data, key=('0008','0080')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 
    
    
def get_institution_address(dicom_data, key=('0008','0081')):
    if key in dicom_data:
        return dicom_data[key].value
    else:
        return '' 


def get_patient_name(dicom_data):
    return dicom_data[('0010', '0010')].value


def get_patient_id(dicom_data):
    return dicom_data[('0010', '0020')].value


def get_study_date(dicom_data):
    return dicom_data[('0008', '0020')].value


def get_study_time(dicom_data):
    return dicom_data[('0008', '0030')].value


def find_phase(file_paths, phase):
    
    results = []
    instance_num = {}
    
    for f in file_paths:
        
        data = pydicom.dcmread(f)
        instance_num[f] = get_instance_number(data)
        data_phase = get_series_desc(data)
        
        if data_phase == phase:
            results.append(f)
            
    results = sorted(results, key = lambda x: instance_num[x])
            
    return results


def window_image(img, window_center,window_width, intercept, slope):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    img = ((img - img_min) * (255/(img_max - img_min))).astype('uint8')
    return img 


def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if isinstance(x, Iterable):
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def extract_multiwindow_image(data, windows=((40, 80), (40, 40), (60, 40))):
    image = data.pixel_array
    instance_num = int(get_instance_number(data))
    _ , _, intercept, slope = get_windowing(data)
    image_windowed = np.stack(
                              [
                                window_image(image, center, width, intercept, slope) 
                                for center, width in windows
                              ], 
                              axis=2
                             )
    return image_windowed, instance_num


def save_img(img_array, save_path, mode='RGB'):
    im = Image.fromarray(img_array).convert(mode)
    im.save(save_path)


def clean_dir(dir_path):
    for f in os.listdir(dir_path):
        p = os.path.join(dir_path, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
