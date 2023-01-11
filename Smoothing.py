"""
Get subsets of Spectral Index and smooth them to export npz files.
These files will later be used in bitemporal indices calculation

e.g (bfast_env) E:\Publications\BFAST_Monitor\code>python Smoothing.py -j ..\param\smoothing_nbr.json
"""
import os
from pathlib import Path
import argparse
import json

import pandas as pd
import numpy as np
from datetime import datetime

from tqdm import tqdm


def get_sys_argv():
    """
    Turn the input json file from the command line to a variable
    :return: a dictionary with the contents of the json file
    """
    parser = argparse.ArgumentParser(description="Parse required arguments for the BFAST Monitor analysis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-j", "--json-file", help="Point to json file that contains required parameters", required=True)

    args = parser.parse_args()
    config = vars(args)
    return config


def smoothing(index, dates, time_series_resampling, output, suffix, si):
    """
    Turns an irregular dataset of a spectral index to a smoothed one using linear interpolation
    and resample it on a given frequency (i.e. '16D'), which is given as a string compatible with pandas.
    :param index: raw dataset of a spectral index with NA values and irregular frequency
    :param dates: the dates responding to the input raw dataset
    :param time_series_resampling: The frequency of the new resampled dataset
    :return: The smoothed dataset, the new date range corresponding to the resampling frequency
    and the metadata of the dataset (rows and columns).
    """
    index = np.where(index == -32768, np.nan, index)
    rows, cols = index.shape[1], index.shape[2]
    index_df = pd.DataFrame(index.reshape([index.shape[0], -1]), index=dates)
    del index
    smoothed = index_df.resample('D').median()
    smoothed = smoothed.interpolate(method='linear', axis=0)
    smoothed = smoothed.resample(time_series_resampling).median()
    smoothed = smoothed.interpolate(method='linear', axis=0)
    smoothed = smoothed.replace(np.nan, -32768)
    smoothed_np = smoothed.to_numpy().astype('int16')
    new_dates_range = smoothed.index.to_pydatetime().tolist()[1:]
    del smoothed
    bands = smoothed_np.shape[0]
    smoothed_np.shape = (bands, rows, cols)
    smoothed_np = np.ascontiguousarray(smoothed_np.reshape((bands, rows, cols)))

    np.savez_compressed(Path(output, f'sm_{si}_{time_series_resampling}_{suffix}'), smoothed_np)

    return smoothed_np, new_dates_range, rows, cols


def main():
    # Assign variables from json
    args = get_sys_argv()
    arg_json_file = args['json_file']
    with open(arg_json_file) as f:
        parameters_dict = json.load(f)

    inpath = parameters_dict['input']

    index = parameters_dict['index']

    dates_file = parameters_dict['dates']
    time_series_resampling = parameters_dict['time_series_resampling']

    # Set list of dates
    with open(dates_file) as f:
        lines = f.read().split('\n')
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in lines if len(d) > 0]

    subs = os.listdir(inpath)
    bar = tqdm(range(len(subs)))

    for i in bar:
        bar.set_description('Loading')
        sub = subs[i]
        tmp = Path(inpath, sub, 'data')

        flist = os.listdir(tmp)
        flist.sort()
        npz_paths = []
        for f in flist:
            if f.endswith('.npz'):
                npz_paths.append(os.path.join(tmp, f))
        npz_paths.sort()

        f_stack = []
        for f in npz_paths:
            im = np.load(f)
            arr = im[im.files[0]]
            f_stack.append(arr)

        nbr_raw = np.stack(f_stack, axis=0)

        output = Path(inpath, sub)
        bar.set_description('Smoothing')
        smoothing(nbr_raw, dates, time_series_resampling, output, sub, index)


if __name__ == "__main__":
    main()
