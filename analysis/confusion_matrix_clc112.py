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


def save_classified(ref, brp, width, height, transform, crs, output_path):
    cla = brp

    brp = np.where(ref > 112, -1, brp)
    ref = np.where(ref != 112, 0, ref)

    cla = np.where((ref == 0) & (brp == -1), 1, cla)
    cla = np.where((ref > 0) & (brp == -1), -1, cla)
    cla = np.where((ref > 0) & (brp > 0), 2, cla)
    cla = np.where((ref == 0) & (brp > 0), -2, cla)

    cla = cla.reshape([1, height, width])

    out = '/'.join(output_path.split('/')[:-1] + [output_path.split('/')[-1].replace('csv', 'tif')])

    with rio.open(out, 'w',
                  driver='GTiff',
                  height=height,
                  width=width,
                  count=1,
                  dtype=rio.float32,
                  crs=crs,
                  transform=transform) as dst:
        dst.write(cla)


def compute_error_matrix(ref, brp, emsr, idx, output):
    # All studied pixels
    # unburned_unburned = np.count_nonzero((ref == 0) & (brp == -1))
    # burned_unburned = np.count_nonzero((ref > 0) & (brp == -1))
    # burned_burned = np.count_nonzero((ref > 0) & (brp > 0))
    # unburned_burned = np.count_nonzero((ref == 0) & (brp > 0))

    # Only for specific class of pixels (ref|clc==112)
    brp = np.where(ref > 112, -1, brp)
    ref = np.where(ref != 112, 0, ref)

    unburned_unburned = np.count_nonzero((ref == 0) & (brp == -1))
    burned_unburned = np.count_nonzero((ref > 0) & (brp == -1))
    burned_burned = np.count_nonzero((ref > 0) & (brp > 0))
    unburned_burned = np.count_nonzero((ref == 0) & (brp > 0))

    # Classes and Reference
    class_burned = burned_burned + unburned_burned
    class_unburned = unburned_unburned + burned_unburned
    ref_burned = burned_unburned + burned_burned
    ref_unburned = unburned_unburned + unburned_burned

    total_pixels = ref_unburned + ref_burned

    print(class_burned, class_unburned, ref_burned, ref_unburned)
    if ref_burned>0:
        try:
            PA_burned = np.round(burned_burned / ref_burned, 3)
            PA_unburned = np.round(unburned_unburned / ref_unburned, 3)
            UA_burned = np.round(burned_burned / class_burned, 3)
            UA_unburned = np.round(unburned_unburned / class_unburned, 3)

            OA = np.round((burned_burned + unburned_unburned) / total_pixels, 3)
            AC = np.round(ref_burned / total_pixels * class_burned / total_pixels +
                          ref_unburned / total_pixels * class_unburned / total_pixels, 3)
            Kappa = np.round((OA - AC) / (1 - AC), 3)
        except ZeroDivisionError:
            PA_burned = np.round(burned_burned / ref_burned, 3)
            PA_unburned = np.round(unburned_unburned / ref_unburned, 3)
            UA_burned = 0
            UA_unburned = np.round(unburned_unburned / class_unburned, 3)

            OA = np.round((burned_burned + unburned_unburned) / total_pixels, 3)
            AC = np.round(ref_burned / total_pixels * class_burned / total_pixels +
                          ref_unburned / total_pixels * class_unburned / total_pixels, 3)
            Kappa = np.round((OA - AC) / (1 - AC), 3)

        df = pd.DataFrame({'emsr': emsr,
                           'index': idx,
                           'burned_burned': burned_burned,
                           'burned_unburned': burned_unburned,
                           'unburned_burned': unburned_burned,
                           'unburned_unburned': unburned_unburned,
                           'class_burned': class_burned,
                           'class_unburned': class_unburned,
                           'ref_burned': ref_burned,
                           'ref_unburned': ref_unburned,
                           'total_pixels': total_pixels,
                           'PA_b': PA_burned,
                           'UA_b': UA_burned,
                           'PA_u': PA_unburned,
                           'UA_u': UA_unburned,
                           'OA': OA,
                           'Kappa': Kappa}, index=[0])

        df.to_csv(output, sep=',', index=False)

        return df.to_dict('records')
    else:
        df = pd.DataFrame({'emsr': emsr,
                           'index': idx,
                           'burned_burned': burned_burned,
                           'burned_unburned': burned_unburned,
                           'unburned_burned': unburned_burned,
                           'unburned_unburned': unburned_unburned,
                           'class_burned': '#NA',
                           'class_unburned': '#NA',
                           'ref_burned': '#NA',
                           'ref_unburned': '#NA',
                           'total_pixels': '#NA',
                           'PA_b': '#NA',
                           'UA_b': '#NA',
                           'PA_u': '#NA',
                           'UA_u': '#NA',
                           'OA': '#NA',
                           'Kappa': '#NA'}, index=[0])

        df.to_csv(output, sep=',', index=False)

        return df.to_dict('records')


def main():
    # args = get_sys_argv()
    # arg_json_file = args['json_file']
    arg_json_file = "D:/Publications/BFAST_Monitor/results/emsr/zonal_stats_ras2ras/rasta_code18_dnbr_rbr_nbr360_urban.json"

    with open(arg_json_file) as f:
        parameters_dict = json.load(f)

    df_all = pd.DataFrame(columns=['emsr', 'index', 'burned_burned', 'burned_unburned', 'unburned_burned',
                                   'unburned_unburned', 'class_burned', 'class_unburned', 'ref_burned', 'ref_unburned',
                                   'total_pixels', 'PA_b', 'UA_b', 'PA_u', 'UA_u', 'OA', 'Kappa'])

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

        save_classified(ref, brp, width, height, transform, crs, output_path)

        results_dict = compute_error_matrix(ref=ref, brp=brp, idx=idx, emsr=emsr, output=output_path)

        # df_all = pd.concat([df_all, results_dict])
        df_all = df_all.append(results_dict, ignore_index=True)

    df_all.to_csv(arg_json_file.replace('json', 'csv'), index=False, sep=',')
