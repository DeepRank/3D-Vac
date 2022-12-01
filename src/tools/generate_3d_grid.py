#!/usr/bin/env python

import os
import subprocess as sp

import h5py
import numpy as np
import pdb2sql
import random

from deeprank.tools import sparse


def visualize3Ddata(hdf5=None, out=None, n_mols=20):
    """This function can be used to generate cube files for the visualization
    of the mapped data in VMD.
    Usage
    python generate_cube_files.py <mol_dir_name>
    e.g. python generate_cube_files.py 1AK4
    or within a python script
    import deeprank.map
    deeprank.map.generate_viz_files(mol_dir_name)
    A new subfolder data_viz will be created in <mol_dir_name>
    with all the cube files representing the features contained in
    the files <mol_dir_name>/input/*.npy
    A script called <feature_name>.vmd is also outputed et allow for
    quick vizualisation of the data by typing
    vmd -e <feature_name>.vmd
    """

    outdir = out

    if outdir is None:
        outdir = mol_name

    if outdir[-1] != '/':
        outdir = outdir + '/'

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    try:
        f5 = h5py.File(hdf5, 'r')
    except BaseException:
        raise FileNotFoundError('HDF5 file %s could not be opened' % hdf5)

    try:
        mols = list(f5.keys())
        mol_names = random.sample(mols, n_mols)
    except BaseException:
        raise LookupError('Molecule %s not found in %s' % (mol_name, hdf5))

    # create the pdb file
    cube_data = []
    grids = []
    for i,mol_name in enumerate(mol_names):
        molgrp = f5[mol_name]
        sqldb = pdb2sql.pdb2sql(molgrp['complex'][:])
        sqldb.exportpdb(outdir + f'/complex_{i}.pdb')
        sqldb._close()

        # get the grid
        grid = {}
        grid['x'] = molgrp['grid_points/x'][:]
        grid['y'] = molgrp['grid_points/y'][:]
        grid['z'] = molgrp['grid_points/z'][:]
        shape = (len(grid['x']), len(grid['y']), len(grid['z']))
        grids.append(grid)

        # deals with the features
        mapgrp = molgrp['mapped_features']

        # loop through all the features
        data_name = list(mapgrp)[0]
        # create a dict of the feature {name: value}
        featgrp = mapgrp[data_name]
        data_dict = {}
        ff = list(featgrp.keys())[0]
        subgrp = featgrp[ff]
        if not subgrp.attrs['sparse']:
            data_dict[ff] = subgrp['value'][:]
        else:
            spg = sparse.FLANgrid(
                sparse=True,
                index=subgrp['index'][:],
                value=subgrp['value'][:],
                shape=shape)
            data_dict[ff] = spg.to_dense()
        cube_data.append(data_dict)
    f5.close()

    # export the cube file
    export_cube_files(cube_data, "grid", grids, outdir)



def export_cube_files(data_arr, data_name, grids, export_path):
    """Generates a cube file per case given in data_arr containing one cube file
    per case which should be visualize at the same time.

    Args:
        data_arr (array): array containing cube info for each case 
        data_name (string): name of the vmd file
        grids (array): grids of data_arr
        export_path (str): destination path of the final file
    """

    print(f"-- Exporting file {data_name}..")
    bohr2ang = 0.52918

    # export files for visualization
    # export VMD script if cube format is required
    vmd_fname = export_path + data_name + '.vmd'
    vmd_f = open(vmd_fname, 'w')
    vmd_f.write('# can be executed with vmd -e viz_mol.vmd\n\n')
    for data_i, grid in enumerate(grids):
        x, y, z = grid['x'], grid['y'], grid['z']

        # extract grid_info
        npts = np.array([len(x), len(y), len(z)])
        res = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])

        # the cuve file is apparently give in bohr
        xmin, ymin, zmin = np.min(x) / bohr2ang, np.min(y) / \
            bohr2ang, np.min(z) / bohr2ang
        scale_res = res / bohr2ang

        for key, values in data_arr[data_i].items(): # at this point only one feature is used to make the cube
            data_name = f"cube_{data_i}"

            cube_fname = export_path + data_name + '_%s' % (key) + '.cube'
            f = open(cube_fname, 'w')
            f.write('CUBE FILE\n')
            f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")

            f.write("%5i %11.6f %11.6f %11.6f\n" % (1, xmin, ymin, zmin))
            f.write("%5i %11.6f %11.6f %11.6f\n" % (npts[0], scale_res[0], 0, 0))
            f.write("%5i %11.6f %11.6f %11.6f\n" % (npts[1], 0, scale_res[1], 0))
            f.write("%5i %11.6f %11.6f %11.6f\n" % (npts[2], 0, 0, scale_res[2]))

            # the cube file require 1 atom
            f.write("%5i %11.6f %11.6f %11.6f %11.6f\n" % (0, 0, 0, 0, 0))

            last_char_check = True
            for i in range(npts[0]):
                for j in range(npts[1]):
                    for k in range(npts[2]):
                        f.write(" %11.5e" % values[i, j, k])
                        last_char_check = True
                        if k % 6 == 5:
                            f.write("\n")
                            last_char_check = False
                    if last_char_check:
                        f.write("\n")
            f.close()


        # write all the cube file in one given molecule
        write_molspec_vmd(
            vmd_f,
            f"cube_{data_i}_{key}.cube",
            'Isosurface',
            'Volume')
        vmd_f.write('mol addfile ' + data_name + '_%s.cube\n' % (key))
        vmd_f.write('mol rename top ' + data_name)

        # load the complex
        write_molspec_vmd(vmd_f, f'complex_{data_i}.pdb', 'NewCartoon', 'Chain')

    vmd_f.close()


# quick shortcut for writting the vmd file
def write_molspec_vmd(f, name, rep, color):
    f.write('\nmol new %s\n' % name)
    f.write('mol delrep 0 top\nmol representation %s\n' % rep)
    if color is not None:
        f.write('mol color %s \n' % color)
    f.write('mol addrep top\n\n')


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='export the grid data in cube format')
    parser.add_argument(
        '--hdf5', "-f",
        help="hdf5 file storing the data set",
        default="/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/000_hla_a_02_01_9_length_peptide.hdf5")
    parser.add_argument(
        '--n_mols', "-n",
        help="name of the molecule in the hdf5",
        default=1,
        type=int)
    parser.add_argument(
        "--out", "-o",
        help="Path to dump the generated cube files. Default:/home/lepikhovd/cubes",
        default="/home/lepikhovd/cubes"
    )
    args = parser.parse_args()

    # lauch the tool

    visualize3Ddata(hdf5=args.hdf5, out=args.out, n_mols=args.n_mols)