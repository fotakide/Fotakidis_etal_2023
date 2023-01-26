"""

"""

# packages for system interactions
import argparse
import glob
import json
import os
from pathlib import Path

# packages for array manipulation
import rasterio as rio
import pandas as pd
import numpy as np
from natsort import natsort_keygen

from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from PyAstronomy import pyasl

# package from bfast monitor and utilities
from bfast import BFASTMonitor
from bfast.monitor import utils

# package ro keep track of processing
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


def smoothing(index, dates, time_series_resampling):
    """
    Turns an irregular dataset of a spectral index to a smoothed one using linear interpolation
    and resample it on a given frequency (i.e. '16D'), which is given as a string compatible with pandas.
    :param index: Raw dataset of a spectral index with NA values and irregular frequency.
    :param dates: the dates responding to the input raw dataset
    :param time_series_resampling: The frequency of the new resampled dataset
    :return: The smoothed dataset, the new date range corresponding to the resampling frequency
    and the metadata of the dataset (rows and columns).
    """
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

    return smoothed_np, new_dates_range, rows, cols


def get_bitemporal_data(index, smoothed_stack):
    """
    After completing the smoothing function compute the bitemporal dataset. Currently from dNBR, RBR.
    :param index: A string of the spectral index name from the list ['nbr', 'dnbr', 'rbr']
    :param smoothed_stack: The resulted dataset after interpolation and resampling
    :return: A numpy stack of the index
    """

    if index in ['rbr', 'dnbr']:
        f_stack = []
        if index == 'rbr':
            for post in range(1, smoothed_stack.shape[0]):
                pre = post - 1

                arr_pre = smoothed_stack[pre]
                arr_post = smoothed_stack[post]

                # RBR = np.where((((arr_pre == -32768) & (arr_post == -32768)) | (
                #         ((arr_pre == -32768) & (arr_post != -32768)) | ((arr_pre != -32768) & (arr_post == -32768)))),
                #                -3.2768,
                #                (arr_pre - arr_post) / (arr_pre + 10010))
                # RBR = np.where(RBR < -1, -3.2768, RBR)
                # RBR = np.round(RBR, decimals=4).astype('float32')
                RBR = (arr_pre - arr_post) / (arr_pre + 1.001)
                # RBR = (np.multiply(RBR, 10000)).astype('int16')

                f_stack.append(RBR)
            data = np.stack(f_stack, axis=0)
        elif index == 'dnbr':
            for post in range(1, smoothed_stack.shape[0]):
                pre = post - 1

                arr_pre = smoothed_stack[pre]
                arr_post = smoothed_stack[post]

                # dNBR = np.where((((arr_pre == -32768) & (arr_post == -32768)) | (
                #         ((arr_pre == -32768) & (arr_post != -32768)) | ((arr_pre != -32768) & (arr_post == -32768)))),
                #                 -32768,
                #                 (arr_pre - arr_post)
                #                 )

                dNBR = arr_pre - arr_post

                # dNBR = np.where((dNBR > 10000) | (dNBR < -10000), -32768, dNBR)

                f_stack.append(dNBR)
            data = np.stack(f_stack, axis=0)

    return data


def get_geotransformation(metadata_src):
    """
    :param metadata_src: Path to a file from which to read metadata
    :return: A list of metadata in [width, height, transform, crs] order
    """
    with rio.open(metadata_src) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
    metadata = [width, height, transform, crs]
    if not src.closed: src.close()
    return metadata


def get_monitor_periods_ranges(start, dates):
    starts_list = [s for s in pd.date_range(start=start, end=dates[-1], freq='6MS', tz=None).to_pydatetime()]
    if (dates[-1] - starts_list[-1]) < timedelta(days=+360):
        starts_list[-1] = (dates[-1] - timedelta(days=+360)).replace(day=1)
    ends_list = [e + relativedelta(months=+12) for e in starts_list]
    ends_list[-1] = dates[-1]
    return starts_list, ends_list


def assemble_results(model, rows, cols, data, dates2, dates, start, end, output, name, sub):
    dates_monitor = []
    dates_array = []

    # collect dates for monitor period
    for i in range(len(dates2)):
        if start <= dates2[i]:
            dates_monitor.append(dates2[i])
    dates_array = np.array(dates_monitor)

    breaks_dec = np.empty([rows, cols], dtype=float)
    magnitudes = np.empty([rows, cols], dtype=float)

    # Extract the relevant dates from the dates_array using the indices in the breaks array
    if model.breaks.max() >= 0:
        valid_break = model.breaks >= 0
        dates_decimal = np.array([round(pyasl.decimalYear(d), 3) for d in dates_array])
        breaks_dec = np.where(valid_break, dates_decimal[model.breaks], model.breaks)
    else:
        breaks_dec = model.breaks

    breaks = model.breaks
    means = model.means
    valids = model.valids
    magnitudes = model.magnitudes

    data = data[int(data.shape[0] - len(dates_array)):data.shape[0], :, :]
    flags = np.expand_dims(model.breaks, axis=0)
    values_of_breaks = np.choose(flags, data, mode='clip')
    values_of_breaks = values_of_breaks.squeeze()

    output = Path(output, f'{start.year}-{end.year}-{start.month}')
    output.mkdir(parents=True, exist_ok=True)

    result_names = ['breaks', 'breaks_dec', 'magnitudes', 'means', 'valids', 'values_of_breaks']
    list(map(lambda subdir_name: (output.joinpath(subdir_name)).mkdir(parents=True, exist_ok=True), result_names))

    np.savez_compressed(f'{str(output)}/breaks/{name}_breaks_{sub}_{start.year}-{end.year}-{start.month}', breaks)
    np.savez_compressed(f'{str(output)}/breaks_dec/{name}_breaks_dec_{sub}_{start.year}-{end.year}-{start.month}',
                        breaks_dec)
    np.savez_compressed(f'{str(output)}/magnitudes/{name}_magnitudes_{sub}_{start.year}-{end.year}-{start.month}',
                        magnitudes)
    np.savez_compressed(f'{str(output)}/means/{name}_means_{sub}_{start.year}-{end.year}-{start.month}', means)
    np.savez_compressed(f'{str(output)}/valids/{name}_valids_{sub}_{start.year}-{end.year}-{start.month}', valids)
    np.savez_compressed(
        f'{str(output)}/values_of_breaks/{name}_values_of_breaks_{sub}_{start.year}-{end.year}-{start.month}',
        values_of_breaks)


def join_results(input_path, output_path, metadata, name, prefix):
    width, height, transform, crs = metadata

    periods = os.listdir(input_path)
    bar = tqdm(range(len(periods)))
    for task in bar:
        period = periods[task]
        bar.set_description(f'Joining Results {period}')
        result_types = os.listdir(Path(input_path, period))

        for result in result_types:
            bar.set_description(f'Joining Results {period} - {result}')
            tmp_output_dir = Path(output_path, period)
            tmp_output_dir.mkdir(parents=True, exist_ok=True)

            npz_paths = glob.glob(str(Path(input_path, period, str(result), '*sub[0-9]*.npz')))
            key = natsort_keygen(key=lambda y: y.lower())
            npz_paths.sort(key=key)

            joined = np.empty([0, width], dtype=float)

            for npz in npz_paths:
                im = np.load(npz)
                arr = im[im.files[0]]

                joined = np.append(joined, arr, axis=0)

            joined = np.reshape(joined, (1, height, width))

            res_name = f'{name}_{prefix}_{result}_{period}.tif'

            WritePath = Path(tmp_output_dir, res_name)

            with rio.open(WritePath, 'w',
                          driver='GTiff',
                          height=height,
                          width=width,
                          count=1,
                          dtype=rio.float32,
                          crs=crs,
                          transform=transform) as dst:
                dst.write(joined)


def do_bfast_monitor_6_month_sequential(model, starts, ends, data, dates, output, name, sub):
    history = dates[0]
    rows, cols = data.shape[1], data.shape[2]
    i = 0
    for period in starts:
        data2, dates2 = utils.crop_data_dates(data, dates, start=history, end=ends[i])
        model.set_params(start_monitor=period)
        model.fit(data=data2.astype('int16'), dates=dates2, nan_value=-32768)
        period_start, period_end = period, ends[i]
        assemble_results(model, rows, cols, data2, dates2, dates, period_start,
                         period_end, output, name, sub)
        i += 1


def main():
    """

    :return:
    """

    # Assign variables from json
    args = get_sys_argv()
    arg_json_file = args['json_file']
    with open(arg_json_file) as f:
        parameters_dict = json.load(f)

    input_files = parameters_dict['input']
    supplying_16D_smoothed_npz = parameters_dict['supplying_16D_smoothed_npz']

    output = parameters_dict['output']
    output_prejoin = parameters_dict['output_prejoin']
    prefix = parameters_dict['prefix']

    index = parameters_dict['index']

    metadata_source_path = parameters_dict['metadata_source_path']
    metadata = get_geotransformation(metadata_source_path)

    bfast_param = parameters_dict['bfast_monitor_param']

    dates_file = parameters_dict['dates']
    start_monitor = datetime.strptime(bfast_param['start_monitor'], '%Y-%m-%d')
    time_series_resampling = parameters_dict['time_series_resampling']
    only_join = parameters_dict['only_join']

    # Set list of dates
    with open(dates_file) as f:
        lines = f.read().split('\n')
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in lines if len(d) > 0]

    starts, ends = get_monitor_periods_ranges(start_monitor, dates)

    model = BFASTMonitor(
        start_monitor=starts[0],
        freq=bfast_param['freq'],
        k=bfast_param['k'],
        hfrac=bfast_param['hfrac'],
        trend=bfast_param['trend'],
        level=bfast_param['level'],
        backend=bfast_param['backend'],
        verbose=bfast_param['verbose'],
        device_id=bfast_param['device_id'],
        detailed_results=bfast_param['detailed_results'],
        find_magnitudes=bfast_param['find_magnitudes']
    )

    if supplying_16D_smoothed_npz and not only_join:
        dates = [d for d in pd.date_range(start=dates[0], end=dates[-1], freq='16D', tz=None).to_pydatetime()]
        if index in ['dnbr', 'rbr']:
            subs = os.listdir(input_files)
            bar = tqdm(range(len(subs)))
            dates = dates[1:len(dates)]

            for task in bar:
                bar.set_description('Loading...')
                sub = subs[task]
                npz = np.load(str(Path(input_files, sub, f'sm_nbr_16D_{sub}.npz')))
                sm_index = np.array(npz[npz.files[0]])
                print(Path(input_files, sub, f'sm_nbr_16D_{sub}.npz'))
                data = get_bitemporal_data(index, sm_index)
                del sm_index

                output_path = Path(output_prejoin, prefix, index)
                output_path.mkdir(parents=True, exist_ok=True)

                bar.set_description('Monitoring...')
                do_bfast_monitor_6_month_sequential(model=model,
                                                    starts=starts,
                                                    ends=ends,
                                                    data=data,
                                                    dates=dates,
                                                    output=str(output_path),
                                                    name=index,
                                                    sub=sub
                                                    )
        else:
            subs = os.listdir(input_files)
            bar = tqdm(range(len(subs)))
            dates = dates[1:len(dates)]

            for task in bar:
                bar.set_description('Loading...')
                sub = subs[task]
                npz = np.load(str(Path(input_files, sub, f'sm_nbr_{time_series_resampling}_{sub}.npz')))
                data = np.array(npz[npz.files[0]])

                output_path = Path(output_prejoin, prefix, index)
                output_path.mkdir(parents=True, exist_ok=True)

                bar.set_description('Monitoring...')
                do_bfast_monitor_6_month_sequential(model=model,
                                                    starts=starts,
                                                    ends=ends,
                                                    data=data,
                                                    dates=dates,
                                                    output=str(output_path),
                                                    name=index,
                                                    sub=sub
                                                    )
        bar.set_description('Joining Results...')
        join_results(input_path=str(Path(output_prejoin, prefix, index)),
                     output_path=output,
                     metadata=metadata,
                     name=index,
                     prefix=prefix)
        bar.set_description('Done')
    elif only_join:
        output_path = Path(output_prejoin, prefix, index)
        join_results(input_path=str(Path(output_prejoin, prefix, index)),
                     output_path=output,
                     metadata=metadata,
                     name=index,
                     prefix=prefix)
    else:
        raise Exception("Provide 16-day smoothed npz files")


if __name__ == "__main__":
    main()
