import argparse
import json

import re
import rasterio as rio
import numpy as np
import pandas as pd


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


def compute_error_matrix(ref, brp, emsr, idx, output, ref_dates):
    ref_d = np.where(ref > 0, ref_dates.loc[ref_dates['emsr'] == emsr, 'activation'].item(), ref)

    # All studied pixels
    burned_burned = np.where((ref > 0) & (brp > 0),1,0)
    count_bb = burned_burned.sum()

    diff_bb = np.where(burned_burned, brp - ref_d, np.nan).flatten()
    diff_bb = diff_bb[~np.isnan(diff_bb)]
    diff_bb = diff_bb.round(3)
    bins, hist = np.unique(diff_bb, return_counts=True)

    time_lag = round((bins * hist).sum() / burned_burned.sum(), 3)

    df = pd.DataFrame({'emsr': emsr,
                       'index': idx,
                       'burned_burned': count_bb,
                       'time-lag': time_lag,
                       }, index=[0])

    for j, b in enumerate(bins):
        df[b] = hist[j]

    df.to_csv(output, sep=',', index=False)

    return df.to_dict('records')


def main():
    # args = get_sys_argv()
    # arg_json_file = args['json_file']

    ref_dates = pd.DataFrame(columns=['emsr', 'activation'],
                             data={'emsr':['300-01','300-02','306-01','369-01','369-02','369-03','380-01','389-01','447-01','510-01','527-01','527-02','540-01','542-01'],
                                   'activation':[2018.561,2018.561,2018.614,2019.508,2019.508,2019.508,2019.617,2019.703,2020.558,2021.386,2021.589,2021.589,2021.625,2021.625]})

    arg_json_file = "E:/Publications/BFAST_Monitor/results/emsr/zonal_stats_ras2ras/rasta_code18_dnbr_rbr_nbr360_timelag.json"

    with open(arg_json_file) as f:
        parameters_dict = json.load(f)

    df_all = pd.DataFrame(columns=['emsr', 'index', 'burned_burned', 'time-lag'])

    for x in range(len(parameters_dict)):
        xfiles = parameters_dict[x]
        breaks_dec_path = xfiles['PARAMETERS']['INPUT'].replace("'", "")
        referenced_path = xfiles['PARAMETERS']['ZONES'].replace("'", "")
        output_path = xfiles['OUTPUTS']['OUTPUT_TABLE'].replace("'", "")

        emsr = re.search(r'(\d+-\d+)', breaks_dec_path).group(1)
        idx = re.search(r'(rbr|nbr|dnbr)', breaks_dec_path).group()

        with rio.open(breaks_dec_path, 'r') as b:
            brp = b.read()
        with rio.open(referenced_path, 'r') as r:
            ref = r.read()

        width, height, transform, crs = get_geotransformation(breaks_dec_path)

        brp = brp[0]
        ref = ref[0]

        results_dict = compute_error_matrix(ref=ref, brp=brp, idx=idx, emsr=emsr,
                                            output=output_path, ref_dates=ref_dates)

        # df_all = pd.concat([df_all, results_dict])
        df_all = df_all.append(results_dict, ignore_index=True)

    df_all.to_csv(arg_json_file.replace('json', 'csv'), index=False, sep=',')
