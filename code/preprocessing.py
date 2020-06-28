# %%
# import libraries
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os

import nibabel as nib

from skimage import img_as_float
from skimage import io

# %%


def get_lung_cnts(contours, threshold):
    contours_p = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > threshold:
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w)/h

            if aspect_ratio > 2.5:
                continue
            contours_p.append(cnt)
    return contours_p

# %%


def get_cropped_img(input_img):
    # Make copy
    img = input_img.copy()

    # Perform median filtering
    median = cv.medianBlur(img, 3)

    # Perform adaptive histogram equalization
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(median)

    # Apply threshold
    _, thresh = cv.threshold(
        img_cl, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # Get dimensions of image
    img_hgt, img_wdt = thresh.shape
    # Set background to 0
    f = 0.1
    mask = np.zeros((img_hgt+2, img_wdt+2), np.uint8)
    cv.floodFill(thresh, mask, (int(img_wdt*f), int(img_hgt*f)), 0)
    cv.floodFill(thresh, mask, (int(img_wdt*(1-f)), int(img_hgt*f)), 0)
    cv.floodFill(thresh, mask, (int(img_wdt*f), int(img_hgt*(1-f))), 0)
    cv.floodFill(thresh, mask, (int(img_wdt*(1-f)), int(img_hgt*(1-f))), 0)

    f = 0.01
    cv.floodFill(thresh, mask, (int(img_wdt*f), int(img_hgt*f)), 0)
    cv.floodFill(thresh, mask, (int(img_wdt*(1-f)), int(img_hgt*f)), 0)
    cv.floodFill(thresh, mask, (int(img_wdt*f), int(img_hgt*(1-f))), 0)
    cv.floodFill(thresh, mask, (int(img_wdt*(1-f)), int(img_hgt*(1-f))), 0)

    # Perform dilation
    imgBW = cv.dilate(thresh, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv.findContours(imgBW.copy(), cv.RETR_LIST,
                                  cv.CHAIN_APPROX_SIMPLE)
    radius = 2
    for cnt in contours:
        isEdge = False
        for pt in cnt:
            pt_y = pt[0][1]
            pt_x = pt[0][0]

            # Check if within radius of border
            check_y = (pt_y >= 0 and pt_y < radius) or (
                pt_y >= img_hgt-1-radius and pt_y < img_hgt)
            check_x = (pt_x >= 0 and pt_x < radius) or (
                pt_x >= img_wdt-1-radius and pt_x < img_wdt)

            if check_y or check_x:
                isEdge = True
                cv.fillPoly(imgBW, pts=[cnt], color=(0, 0, 0))
                break
        if not isEdge:
            cv.fillPoly(imgBW, pts=[cnt], color=(255, 255, 255))

    contours, _ = cv.findContours(imgBW.copy(), cv.RETR_LIST,
                                  cv.CHAIN_APPROX_SIMPLE)

    threshold = 1500
    contours_p = get_lung_cnts(contours, threshold)

    count = 0
    while(len(contours_p) != 2):
        if len(contours_p) > 2:
            threshold += 100
        if len(contours_p) < 2:
            threshold -= 100
            count += 1
            if count == 10:
                break

        contours_p = get_lung_cnts(contours, threshold)

     min_y, min_x = int(img.shape[0]/2), int(img.shape[1]/2)
    max_y, max_x = int(img.shape[0]/2), int(img.shape[1]/2)

    for cnt in contours_p:
        for pt in cnt:
            if pt[0][1] < min_y:
                min_y = pt[0][1]
            if pt[0][1] > max_y:
                max_y = pt[0][1]
            if pt[0][0] < min_x:
                min_x = pt[0][0]
            if pt[0][0] > max_x:
                max_x = pt[0][0]

    min_x -= 5
    min_y -= 5
    max_x += 5
    max_y += 5

    if min_x < 0:
        min_x = 0
    if min_y < 0:
        min_y = 0
    if max_x > img.shape[1]:
        max_x = img.shape[1]
    if max_y > img.shape[0]:
        max_y = img.shape[0]

    img_crop = img[min_y:max_y, min_x:max_x]

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_crop = clahe.apply(img_crop)

    img_pad = cv.copyMakeBorder(img_crop,
                                int((max(img_crop.shape)-img_crop.shape[0])/2),
                                int((max(img_crop.shape)-img_crop.shape[0])/2),
                                int((max(img_crop.shape)-img_crop.shape[1])/2),
                                int((max(img_crop.shape)-img_crop.shape[1])/2),
                                cv.BORDER_CONSTANT,
                                value=[0])

    output_img_size = 280

    if img_pad.shape[0] > output_img_size:
        # Zoom out
        img_rs = cv.resize(
            img_pad, (output_img_size, output_img_size), interpolation=cv.INTER_AREA)
    else:
        # Zoom in
        img_rs = cv.resize(
            img_pad, (output_img_size, output_img_size), interpolation=cv.INTER_CUBIC)
    # 224 by 280
    img_rs = img_rs[28:252, 0:280]

    return img_rs


# %%
# Define source 1 Directory
s1_dir = '..\source\S1_covid19-ct-scans\ct_scans'

# extract files
files = []
for _, _, filenames in os.walk(s1_dir):
    for filename in filenames:
        files.append(filename)

S1_patient_list = []
S1_filename_new = []
S1_filepath = []
S1_isCovid = []
patient_count = 1

for file in files:
    data = nib.load(os.path.join(s1_dir, file))
    middle_slice = int(data.shape[2]/2)
    list_slices = [*range(middle_slice-4, middle_slice+7, 2)]

    count = 1
    for img_slice in list_slices:
        print(f'Processing {file}, slice {count}')
        print('===============================')
        # Get slice from file
        img_slice16 = data.get_fdata()[:, :, img_slice]
        img = np.rot90(np.uint8(cv.normalize(
            img_slice16, None, 0, 255, cv.NORM_MINMAX)), 1)
        crop_img = get_cropped_img(img)
        filename = f"S1_{file.split('.')[0]}_{str(count)}.jpg"
        cv.imwrite(
            f"..\data\CT_Covid\{filename}", crop_img)

        count += 1
        S1_patient_list.append(f"S1_{patient_count}")
        S1_filename_new.append(filename)
        S1_isCovid.append(1)
        S1_filepath.append(f".\data\CT_Covid\{filename}")

    patient_count += 1

df_S1 = pd.DataFrame({'filename': S1_filename_new, 'patientID': S1_patient_list,
                      'isCovid': S1_isCovid, 'filepath': S1_filepath})

# %%
# Define source 2
s2_dir = '..\source\S2_COVID-CT-master'

# extract COVID files
files = []
for _, _, filenames in os.walk(os.path.join(s2_dir, 'Images-processed', 'CT_COVID')):
    for filename in filenames:
        files.append(filename)

S2_patient_list = []
S2_filename_new = []
S2_filepath = []
S2_isCovid = []

S2_meta_covid = pd.read_csv(s2_dir+"\\COVID-CT-MetaInfo.csv")

for file in files:
    print(f'Processing {file}')
    print('=================')
    img = cv.imread(os.path.join(s2_dir, 'Images-processed', 'CT_COVID', file))
    img = np.uint8(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    num_px = 5
    px_top_edge = img[0:num_px, 0:img.shape[1]].flatten()
    px_btm_edge = img[img.shape[0] -
                      num_px:img.shape[0], 0:img.shape[1]].flatten()
    px_lft_edge = img[0:img.shape[0], 0:num_px].flatten()
    px_rgt_edge = img[0:img.shape[0],
                      img.shape[1]-num_px:img.shape[1]].flatten()
    edge_px_value = int(np.median(np.concatenate(
        [px_top_edge, px_btm_edge, px_lft_edge, px_rgt_edge])))
    img = cv.copyMakeBorder(img, 20, 20, 20, 20,
                            cv.BORDER_CONSTANT, value=[edge_px_value])
    crop_img = get_cropped_img(img)
    filename = f"S2_{'_'.join(file.split('.')[:-1])}.jpg"
    cv.imwrite(
        f"..\data\CT_Covid\{filename}", crop_img)

    patient_id = S2_meta_covid.loc[S2_meta_covid['File name']
                                   == file, 'Patient ID'].to_numpy()
    if len(patient_id) != 0:
        patient_id = patient_id[0]
    else:
        patient_id = 'unknown'

    S2_patient_list.append(f"S2_{patient_id.replace(' ','_')}")
    S2_filename_new.append(filename)
    S2_filepath.append(f".\data\CT_Covid\{filename}")
    S2_isCovid.append(1)

# extract nonCOVID files
files = []
for _, _, filenames in os.walk(os.path.join(s2_dir, 'Images-processed', 'CT_NonCOVID')):
    for filename in filenames:
        files.append(filename)

S2_meta_noncovid = pd.read_csv(s2_dir+"\\NonCOVID-CT-MetaInfo.csv")

for file in files:
    print(f'Processing {file}')
    print('=================')
    img = cv.imread(os.path.join(
        s2_dir, 'Images-processed', 'CT_NonCOVID', file))
    img = np.uint8(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    num_px = 5
    px_top_edge = img[0:num_px, 0:img.shape[1]].flatten()
    px_btm_edge = img[img.shape[0] -
                      num_px:img.shape[0], 0:img.shape[1]].flatten()
    px_lft_edge = img[0:img.shape[0], 0:num_px].flatten()
    px_rgt_edge = img[0:img.shape[0],
                      img.shape[1]-num_px:img.shape[1]].flatten()
    edge_px_value = int(np.median(np.concatenate(
        [px_top_edge, px_btm_edge, px_lft_edge, px_rgt_edge])))
    img = cv.copyMakeBorder(img, 20, 20, 20, 20,
                            cv.BORDER_CONSTANT, value=[edge_px_value])
    crop_img = get_cropped_img(img)
    filename = f"S2_{'_'.join(file.split('.')[:-1])}.jpg"
    cv.imwrite(
        f"..\data\CT_NonCovid\{filename}", crop_img)

    patient_id = S2_meta_noncovid.loc[S2_meta_noncovid['image name']
                                      == file, 'patient id'].to_numpy()
    if len(patient_id) != 0:
        patient_id = patient_id[0]
    else:
        patient_id = 'unknown'

    S2_patient_list.append(f"S2_{patient_id.replace(' ','_')}")
    S2_filename_new.append(filename)
    S2_filepath.append(f".\data\CT_NonCovid\{filename}")
    S2_isCovid.append(0)

df_S2 = pd.DataFrame({'filename': S2_filename_new, 'patientID': S2_patient_list,
                      'isCovid': S2_isCovid, 'filepath': S2_filepath})

# %%

# Define Source 3
s3_dir = '..\source\S3_covid-chestxray-dataset-master'

s3_sourceMeta = pd.read_csv(os.path.join(s3_dir, 'metadata.csv'))
s3_covid_df = s3_sourceMeta.loc[(s3_sourceMeta['view'] == 'Axial') &
                                (s3_sourceMeta['finding'].str.contains('COVID')) &
                                (s3_sourceMeta['folder'].str.contains(
                                    'images')),
                                ['patientid', 'filename']]

s3_noncovid_df = s3_sourceMeta.loc[(s3_sourceMeta['view'] == 'Axial') &
                                   (~s3_sourceMeta['finding'].str.contains('COVID')) &
                                   (s3_sourceMeta['folder'].str.contains(
                                       'images')),
                                   ['patientid', 'filename']]

S3_patient_list = []
S3_filename_new = []
S3_filepath = []
S3_isCovid = []

# extract COVID files
for file in list(s3_covid_df['filename']):
    print(f'Processing {file}')
    print('=================')
    img = cv.imread(os.path.join(s3_dir, 'images', file))
    img = np.uint8(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    num_px = 5
    px_top_edge = img[0:num_px, 0:img.shape[1]].flatten()
    px_btm_edge = img[img.shape[0] -
                      num_px:img.shape[0], 0:img.shape[1]].flatten()
    px_lft_edge = img[0:img.shape[0], 0:num_px].flatten()
    px_rgt_edge = img[0:img.shape[0],
                      img.shape[1]-num_px:img.shape[1]].flatten()
    edge_px_value = int(np.median(np.concatenate(
        [px_top_edge, px_btm_edge, px_lft_edge, px_rgt_edge])))
    img = cv.copyMakeBorder(img, 20, 20, 20, 20,
                            cv.BORDER_CONSTANT, value=[edge_px_value])
    crop_img = get_cropped_img(img)
    filename = f"S3_{file.split('.')[0]}.jpg"
    cv.imwrite(
        f"..\data\CT_Covid\{filename}", crop_img)
    patient_id = s3_covid_df.loc[s3_covid_df['filename']
                                 == file, 'patientid'].to_numpy()

    if len(patient_id) != 0:
        patient_id = patient_id[0]
    else:
        patient_id = 'unknown'
    S3_patient_list.append(f"S3_{patient_id}")
    S3_filename_new.append(filename)
    S3_filepath.append(f".\data\CT_Covid\{filename}")
    S3_isCovid.append(1)

# extract nonCOVID files
for file in list(s3_noncovid_df['filename']):
    print(f'Processing {file}')
    print('=================')
    img = cv.imread(os.path.join(s3_dir, 'images', file))
    img = np.uint8(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    num_px = 5
    px_top_edge = img[0:num_px, 0:img.shape[1]].flatten()
    px_btm_edge = img[img.shape[0] -
                      num_px:img.shape[0], 0:img.shape[1]].flatten()
    px_lft_edge = img[0:img.shape[0], 0:num_px].flatten()
    px_rgt_edge = img[0:img.shape[0],
                      img.shape[1]-num_px:img.shape[1]].flatten()
    edge_px_value = int(np.median(np.concatenate(
        [px_top_edge, px_btm_edge, px_lft_edge, px_rgt_edge])))
    img = cv.copyMakeBorder(img, 20, 20, 20, 20,
                            cv.BORDER_CONSTANT, value=[edge_px_value])
    crop_img = get_cropped_img(img)
    filename = f"S3_{file.split('.')[0]}.jpg"
    cv.imwrite(f"..\data\CT_nonCovid\{filename}", crop_img)
    patient_id = s3_noncovid_df.loc[s3_noncovid_df['filename']
                                    == file, 'patientid'].to_numpy()

    if len(patient_id) != 0:
        patient_id = patient_id[0]
    else:
        patient_id = 'unknown'

    S3_patient_list.append(f"S3_{patient_id}")
    S3_filename_new.append(filename)
    S3_filepath.append(f".\data\CT_nonCovid\{filename}")
    S3_isCovid.append(0)

df_S3 = pd.DataFrame({'filename': S3_filename_new, 'patientID': S3_patient_list,
                      'isCovid': S3_isCovid, 'filepath': S3_filepath})

# %%
# Define source 4 Directory
s4_dir = '..\source\S4_covid_19_public_data\synthetic_data'

files = []
for _, _, filenames in os.walk(s4_dir):
    for filename in filenames:
        files.append(filename)

S4_patient_list = []
S4_filename_new = []
S4_filepath = []
S4_isCovid = []
patient_count = 1

files_vol = [file for file in files if 'vol' in file]

for file in files_vol:
    data = nib.load(os.path.join(s4_dir, file))
    middle_slice = int(data.shape[2]/2)
    list_slices = [*range(middle_slice-4, middle_slice+7, 2)]

    count = 1
    for img_slice in list_slices:
        print(f'Processing {file}, slice {count}')
        print('===============================')
        # Get slice from file
        img_slice16 = data.get_fdata()[:, :, img_slice]
        img = np.rot90(np.uint8(cv.normalize(
            img_slice16, None, 0, 255, cv.NORM_MINMAX)), 1)
        crop_img = get_cropped_img(img)
        filename = f"S4_{file.split('.')[0]}_{str(count)}.jpg"
        cv.imwrite(
            f"..\data\CT_Covid\{filename}", crop_img)
        count += 1
        S4_patient_list.append(f"S4_{patient_count}")
        S4_filename_new.append(filename)
        S4_isCovid.append(1)
        S4_filepath.append(f".\data\CT_Covid\{filename}")

    patient_count += 1

df_S4 = pd.DataFrame({'filename': S4_filename_new, 'patientID': S4_patient_list,
                      'isCovid': S4_isCovid, 'filepath': S4_filepath})

# %%
# Define source 5
s5_dir = '..\source\S5_radiopedia'

# extract file list
files = []
for _, _, filenames in os.walk(s5_dir):
    for filename in filenames:
        files.append(filename)

S5_patient_list = []
S5_filename_new = []
S5_filepath = []
S5_isCovid = []

for file in files:
    print(f'Processing {file}')
    print('=================')
    img = cv.imread(os.path.join(s5_dir, file))
    img = np.uint8(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    num_px = 5
    px_top_edge = img[0:num_px, 0:img.shape[1]].flatten()
    px_btm_edge = img[img.shape[0] -
                      num_px:img.shape[0], 0:img.shape[1]].flatten()
    px_lft_edge = img[0:img.shape[0], 0:num_px].flatten()
    px_rgt_edge = img[0:img.shape[0],
                      img.shape[1]-num_px:img.shape[1]].flatten()
    edge_px_value = int(np.median(np.concatenate(
        [px_top_edge, px_btm_edge, px_lft_edge, px_rgt_edge])))
    img = cv.copyMakeBorder(img, 20, 20, 20, 20,
                            cv.BORDER_CONSTANT, value=[edge_px_value])
    crop_img = get_cropped_img(img)
    filename = f"S5_{'_'.join(file.split('.')[:-1])}.jpg"
    cv.imwrite(
        f"..\data\CT_nonCovid\{filename}", crop_img)

    S5_patient_list.append(f"S5_{file.split('v')[0][1:]}")
    S5_filename_new.append(filename)
    S5_isCovid.append(0)
    S5_filepath.append(f".\data\CT_nonCovid\{filename}")

df_S5 = pd.DataFrame({'filename': S5_filename_new, 'patientID': S5_patient_list,
                      'isCovid': S5_isCovid, 'filepath': S5_filepath})
# %%
# Define source 6 Directory
s6_dir = '..\source\S6_mosmed\studies'

S6_patient_list = []
S6_filename_new = []
S6_filepath = []
S6_isCovid = []
patient_count = 1

for root, _, filenames in os.walk(s6_dir):
    for file in filenames:
        data = nib.load(os.path.join(root, file))
        middle_slice = int(data.shape[2]/2)
        list_slices = [*range(middle_slice-4, middle_slice+7, 2)]

        count = 1
        for img_slice in list_slices:
            print(f'Processing {file}, slice {count}')
            print('===============================')
            # Get slice from file
            img_slice16 = data.get_fdata()[:, :, img_slice]
            img = np.rot90(np.uint8(cv.normalize(
                img_slice16, None, 0, 255, cv.NORM_MINMAX)), 1)
            crop_img = get_cropped_img(img)
            filename = f"S6_{file.split('.')[0]}_{str(count)}.jpg"
            if root.split('\\')[-1] == 'CT-0':
                cv.imwrite(f"..\data\CT_nonCovid\{filename}", crop_img)
                S6_isCovid.append(0)
                S6_filepath.append(f".\data\CT_nonCovid\{filename}")
            else:
                cv.imwrite(f"..\data\CT_Covid\{filename}", crop_img)
                S6_isCovid.append(1)
                S6_filepath.append(f".\data\CT_Covid\{filename}")
            count += 1
            S6_patient_list.append(f"S6_{patient_count}")
            S6_filename_new.append(filename)

        patient_count += 1

df_S6 = pd.DataFrame({'filename': S6_filename_new, 'patientID': S6_patient_list,
                      'isCovid': S6_isCovid, 'filepath': S6_filepath})

# %%
""" Generate metadata """
dataframes = [df_S1, df_S2, df_S3, df_S4, df_S5, df_S6]

df_metadata = pd.concat(dataframes)

df_metadata.to_csv('../metadata.csv', index=False)
