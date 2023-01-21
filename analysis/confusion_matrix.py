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


def get_classified(ref, brp):
    unburned_unburned = np.count_nonzero((ref == 0) & (brp == -1))
    burned_unburned = np.count_nonzero((ref > 0) & (brp == -1))
    burned_burned = np.count_nonzero((ref > 0) & (brp > 0))
    unburned_burned = np.count_nonzero((ref == 0) & (brp > 0))


def compute_error_matrix(ref, brp, emsr, idx, output):
    unburned_unburned = np.count_nonzero((ref == 0) & (brp == -1))
    burned_unburned = np.count_nonzero((ref > 0) & (brp == -1))
    burned_burned = np.count_nonzero((ref > 0) & (brp > 0))
    unburned_burned = np.count_nonzero((ref == 0) & (brp > 0))

    class_burned = burned_burned + unburned_burned
    class_unburned = unburned_unburned + burned_unburned
    ref_burned = burned_unburned + burned_burned
    ref_unburned = unburned_unburned + unburned_burned

    total_pixels = ref_unburned + ref_burned

    PA_burned = np.round(burned_burned / ref_burned, 3)
    PA_unburned = np.round(unburned_unburned / ref_unburned, 3)
    UA_burned = np.round(burned_burned / class_burned, 3)
    UA_unburned = np.round(unburned_unburned / class_unburned, 3)

    OA = np.round((burned_burned + unburned_unburned) / total_pixels, 3)
    AC = np.round(ref_burned / total_pixels * class_burned / total_pixels +
                  ref_unburned / total_pixels * class_unburned / total_pixels, 3)
    Kappa = np.round((OA - AC) / (1 - AC), 3)

    df = pd.DataFrame({'emsr': emsr,
                       'index': idx,
                       'burned_burned':burned_burned,
                       'burned_unburned':burned_unburned,
                       'unburned_burned':unburned_burned,
                       'unburned_unburned':unburned_unburned,
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





def main():
    # args = get_sys_argv()
    # arg_json_file = args['json_file']
    arg_json_file = "E:/Publications/BFAST_Monitor/results/emsr/zonal_stats_ras2ras/rasta_code18.json"

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

        brp = brp[0]
        ref = ref[0]

        results_dict = compute_error_matrix(ref=ref, brp=brp, idx=idx, emsr=emsr, output=output_path)

        # df_all = pd.concat([df_all, results_dict])
        df_all = df_all.append(results_dict, ignore_index=True)

    df_all.to_csv(arg_json_file.replace('json', 'csv'), index=False, sep=',')
