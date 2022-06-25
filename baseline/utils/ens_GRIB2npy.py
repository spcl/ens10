# Preprocess the ENS10 dataaset and convert to .npy files [We save ensembles + statistics (mean/std)]

import argparse
import multiprocessing as mp
import os.path
from functools import partial
import numpy as np
from eccodes import *

sfc_params = ['34.128', '136.128', '137.128', '143.128', '151.128', '164.128', '165.128', '166.128', '167.128', '228.128', '235.128']
# Sea surface temperature (SST): 34.128
# Total Column Water (TCW): 136.128
# Total column water vapour (TCWV): 137.128
# Convective precipitation (CP): 143.128
# Mean sea level pressure (MSL): 151.128
# Total cloud cover (TCC): 164.128
# 10m U wind component (U10): 165.128
# 10m V wind component (V10): 166.128
# 2m temperature (T2m): 167.128
# Total precipitation (TP): 228.128
# Skin temperature at the surface (SKT): 235.128


pl_params = ['130.128', '133.128', '135.128', '131.128', '132.128', '129.128', '155.128']  # Parameters used
# 248.128 = fraction cloud cover
# 157.128 = relative humidity
# 131.128 = U component of wind
# 132.128 = V component of wind
# 129.128 = Geopotential
# 130.128 = Temperature
# 155.128 = Divergence
# 133.128 = Specific humidity
# 135.128 = Vertical velocity


def prep_parser():

    parser = argparse.ArgumentParser(description='ENS-10 Preprocessing (GRIB -> .npy)')

    parser.add_argument('-pl', '--pressure-level', default=850, type=int,
                        metavar='PL',
                        choices=[0, 10, 50, 100, 200, 300, 400, 500, 700, 850, 925, 1000],
                        help='Pressure level. 0 means surface level  (Default:850)')

    parser.add_argument('-l', '--lead-time', default=[0, 24, 48], type=int,
                        nargs='+',
                        help='Lead-times for the ensembles (default: [0, 24, 48)')

    parser.add_argument('-n', '--ens-num', default=10, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='Number of Ensembles. -1 will save mean and std of the variables (default: 10)')

    parser.add_argument('-d', '--grib-dir', default='./', type=str,
                        help='The grib files directory (default: ./)')

    parser.add_argument('-o', '--output-dir', default='./', type=str,
                        help='The .npy files directory (default: ./)')

    parser.add_argument('-w', '--workers', default=4, type=int,
                        help='Number of Worders (default: 4)')

    args = parser.parse_args()

    # Remove duplicates
    assert len(set(args.lead_time)) == len(args.lead_time), 'Duplicate Lead-Times!'

    # Check the validity of the lead-times
    for i in args.lead_time:
        assert i in [0, 24, 48], 'Lead time should be 0/24/48.'

    # Sort the lead-times
    args.lead_time.sort()

    return args


def calc(i, args):
    file_indexes = ['0101', '0104', '0108', '0111', '0115', '0118', '0122', '0125', '0129', '0201', '0205', '0208',
                    '0212', '0215', '0219', '0222', '0226', '0301', '0305', '0308', '0312', '0315', '0319', '0322',
                    '0326', '0329', '0402', '0405', '0409', '0412', '0416', '0419', '0423', '0426', '0430', '0503',
                    '0507', '0510', '0514', '0517', '0521', '0524', '0528', '0531', '0604', '0607', '0611', '0614',
                    '0618', '0621', '0625', '0628', '0702', '0705', '0709', '0712', '0716', '0719', '0723', '0726',
                    '0730', '0802', '0806', '0809', '0813', '0816', '0820', '0823', '0827', '0830', '0903', '0906',
                    '0910', '0913', '0917', '0920', '0924', '0927', '1001', '1004', '1008', '1011', '1015', '1018',
                    '1022', '1025', '1029', '1101', '1105', '1108', '1112', '1115', '1119', '1122', '1126', '1129',
                    '1203', '1206', '1210', '1213', '1217', '1220', '1224', '1227', '1231']

    if args.pressure_level == 0:
        params = sfc_params
        grid_num = 6600
    else:
        params = pl_params
        grid_num = 46179

    # Perturbations during the preprocessing phase
    used_pert = [i+1 for i in range(10)]
    if args.ens_num > 0:
        used_pert = [i+1 for i in range(args.ens_num)]

    Nhours = 105
    assert Nhours == len(file_indexes)
    Nparam = len(params)
    Nheight = 1 #for now, we only extract one pressure level
    Nlatitude = 361
    Nlongitude = 720
    npx = np.empty([len(args.lead_time), len(used_pert), Nparam, Nheight, Nlatitude, Nlongitude], dtype='float32')

    str_ = f'{args.ens_num}ENS_PL{args.pressure_level}_'
    if args.pressure_level == 0:
        str_ = f'{args.ens_num}ENS_sfc_'

    for lt in args.lead_time:
        str_ += f'{lt}'

    OUTPUT = os.path.join(args.output_dir,  str_ + '_' + str(i))

    if args.pressure_level == 0:
        prefix = 'output.sfc.2018'
    else:
        prefix = 'output.pl.2018'

    for hi, ind in enumerate(file_indexes):
        FILE = os.path.join(args.grib_dir, prefix+ind)
        if not os.path.exists(FILE):
            continue

        with open(FILE, "r") as fd:
            print(str(i) + ' Year and index ' + str(hi + 1) + '/' + str(Nhours))

            for ise in range(0, grid_num):  # hardcoded the number of GRIB messages per file, to avoid an initial iteration through the whole file
                gidt = codes_grib_new_from_file(fd)  # Get the next GRIB message
                y = codes_get(gidt, "year")
                lvl = codes_get(gidt, "ls.level")

                if i == y and lvl == args.pressure_level:  # only look at it if we're in the correct year, only 500 and 850hpa for now
                    pl_idx = 0
                    P1t = codes_get(gidt, "P1")
                    if P1t not in args.lead_time:
                        continue
                    pnr = codes_get(gidt, "perturbationNumber")
                    Paramt = codes_get(gidt, "param")
                    values = codes_get_values(gidt)

                    if pnr in used_pert:
                        pert = used_pert.index(pnr)
                        paramit = params.index(Paramt)

                    ens_idx = args.lead_time.index(P1t)
                    npx[ens_idx, pert, paramit, pl_idx, :, :] = np.reshape(values, (Nlatitude, Nlongitude))
                codes_release(gidt)
            cstddev = np.std(npx, axis=1, keepdims=True)
            cmean = np.mean(npx, axis=1, keepdims=True)
            cat_mean_std = np.concatenate((cmean, cstddev), axis=1)
            np.save(OUTPUT+'_stat_'+str(hi), cat_mean_std)
            np.save(OUTPUT + str(hi), npx)

    print("done with " + str(i))


if __name__ == '__main__':
    args = prep_parser()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Preprocess GRIB files in parallel, set the amount of workers here
    pool = mp.Pool(args.workers)  # Uses a lot of RAM per worker
    pool.map(partial(calc, args=args), list(range(1999,2018)))