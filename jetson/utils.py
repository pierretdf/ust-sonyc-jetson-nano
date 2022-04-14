#MIT License
#
#Copyright (c) 2019 Sounds of New York City (SONYC)
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in 
#the Software without restriction, including without limitation the rights to 
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
#of the Software, and to permit persons to whom the Software is furnished to do 
#so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all 
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
#SOFTWARE.

"""

These functions have been taken and potentially modified from 

https://github.com/sonyc-project/urban-sound-tagging-baseline

"""

import csv
import os
import numpy as np


def get_subset_split(annotation_data):
    """
    Get indices for train and validation subsets

    Parameters
    ----------
    annotation_data

    Returns
    -------
    train_idxs
    valid_idxs

    """

    # Get the audio filenames and the splits without duplicates
    data = annotation_data[['split', 'audio_filename']].drop_duplicates().sort_values('audio_filename')

    train_idxs = []
    valid_idxs = []
    test_idx = []
    for idx, (_, row) in enumerate(data.iterrows()):
        if row['split'] == 'train':
            train_idxs.append(idx)
        elif row['split'] == 'validate':
            valid_idxs.append(idx)
        elif row['split'] == 'test':
            test_idx.append(idx)

    return np.array(train_idxs), np.array(valid_idxs), np.array(test_idx)


def get_file_targets(annotation_data, labels):
    """
    Get file target annotation vector for the given set of labels

    Parameters
    ----------
    annotation_data
    labels

    Returns
    -------
    target_list

    """
    target_list = []
    file_list = annotation_data['audio_filename'].unique().tolist()

    for filename in file_list:
        file_df = annotation_data[annotation_data['audio_filename'] == filename]
        target = []

        for label in labels:
            count = 0

            for _, row in file_df.iterrows():
                if int(row['annotator_id']) == 0:
                    # If we have a validated annotation, just use that
                    count = row[label + '_presence']
                    break
                else:
                    count += row[label + '_presence']

            if count > 0:
                target.append(1.0)
            else:
                target.append(0.0)

        target_list.append(target)

    return np.array(target_list)

def generate_output_file(y_pred, test_file_idxs, results_dir, file_list,
                         aggregation_type, label_mode, taxonomy):
    """
    Write the output file containing model predictions
    
    reproduced from  https://github.com/sonyc-project/urban-sound-tagging-baseline/blob/master/urban-sound-tagging-baseline/classify.py

    Parameters
    ----------
    y_pred
    test_file_idxs
    results_dir
    file_list
    aggregation_type
    label_mode
    taxonomy

    Returns
    -------

    """
    output_path = os.path.join(results_dir, "output_{}.csv".format(aggregation_type))
    test_file_list = [file_list[idx] for idx in test_file_idxs]

    coarse_fine_labels = [["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                             for fine_id, fine_label in fine_dict.items()]
                           for coarse_id, fine_dict in taxonomy['fine'].items()]

    full_fine_target_labels = [fine_label for fine_list in coarse_fine_labels
                                          for fine_label in fine_list]
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]


    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename"] + full_fine_target_labels + coarse_target_labels
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(test_file_list, y_pred):
            row = [filename]

            if label_mode == "fine":
                fine_values = []
                coarse_values = [0 for _ in range(len(coarse_target_labels))]
                coarse_idx = 0
                fine_idx = 0
                for coarse_label, fine_label_list in zip(coarse_target_labels,
                                                         coarse_fine_labels):
                    for fine_label in fine_label_list:
                        if 'X' in fine_label.split('_')[0].split('-')[1]:
                            # Put a 0 for other, since the baseline doesn't
                            # account for it
                            fine_values.append(0.0)
                            continue

                        # Append the next fine prediction
                        fine_values.append(y[fine_idx])

                        # Add coarse level labels corresponding to fine level
                        # predictions. Obtain by taking the maximum from the
                        # fine level labels
                        coarse_values[coarse_idx] = max(coarse_values[coarse_idx],
                                                        y[fine_idx])
                        fine_idx += 1
                    coarse_idx += 1

                row += fine_values + coarse_values

            else:
                # Add placeholder values for fine level
                row += [0.0 for _ in range(len(full_fine_target_labels))]
                # Add coarse level labels
                row += list(y)

            csvwriter.writerow(row)

