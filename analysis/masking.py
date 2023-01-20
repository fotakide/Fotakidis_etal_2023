import argparse
import json

import rasterio as rio
import numpy as np


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


def main():
    """

    :return:
    """

    # Assign variables from json

    # args = get_sys_argv()
    # arg_json_file = args['json_file']
    arg_json_file = "E:/Publications/BFAST_Monitor/results/emsr/rasters/breaks_dec/thres/nbr_thres.json"
    # arg_json_file = "E:/Publications/BFAST_Monitor/results/emsr/rasters/breaks_dec/thres/dnbr_thres.json"
    with open(arg_json_file) as f:
        parameters_dict = json.load(f)

    for x in range(len(parameters_dict)):
        xfiles = parameters_dict[x]
        breaks_dec_path = xfiles['PARAMETERS']['breaksdecimal'].replace("'", "")
        magnitudes_path = xfiles['PARAMETERS']['magnitudes'].replace("'", "")
        output_path = xfiles['OUTPUTS']['qgis:rastercalculator_1:breaks_with_valid_magnitude'].replace("'", "")

        width, height, transform, crs = get_geotransformation(breaks_dec_path)

        with rio.open(breaks_dec_path, 'r') as b:
            breaks_dec = b.read()
        with rio.open(magnitudes_path, 'r') as m:
            magnitudes = m.read()

        breaks_dec_masked = breaks_dec[0]
        # breaks_dec_masked = np.where((breaks_dec[0]>0)&(magnitudes[0] < 1000), -1, breaks_dec_masked) # dNBR value

        breaks_dec_masked = np.where((breaks_dec[0]>0)&(magnitudes[0] > -360.255), -1, breaks_dec_masked) # NBR value

        breaks_dec_masked = breaks_dec_masked.reshape([1,height,width])
        with rio.open(output_path, 'w',
                      driver='GTiff',
                      height=height,
                      width=width,
                      count=1,
                      dtype=rio.float32,
                      crs=crs,
                      transform=transform) as dst:
            dst.write(breaks_dec_masked)
