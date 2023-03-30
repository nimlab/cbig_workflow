from nimlab import functions as fn
from nimlab import datasets as ds
from nimlab import connectomics as cs
import os
from nilearn import image, maskers
from tqdm import tqdm
from scipy import stats
from multiprocessing import Pool
import sys
import numpy as np

def adjacent_voxels(x, y, z, shape):
    def _inbounds(coord, shape):
        if coord[0] > 0 and coord[0] < shape[0] and coord[1] > 0 and coord[1] < shape[1] and coord[2] > 0 and coord[2] < shape[2]:
            return True
        return False
    adj = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                adj_candidate = (x+i, y+j, z+k)
                if _inbounds(adj_candidate, shape):
                    adj.append(adj_candidate)
    return adj


def searchlight_vox(args):
    brain_img = args[0]
    masker = args[1]
    allfx = args[2]
    tc = args[3]
    c = args[4]
    world_c = image.coord_transform(c[0], c[1], c[2], brain_img.affine)
        #print(world_c)
    cone = masker.transform(fn.make_tms_cone(brain_img, world_c[0], world_c[1], world_c[2]))
    cone_avgtc = cs.extract_avg_signal(tc, cone)
    cone_map = cs._vec2mat_corr(cone_avgtc, tc)
    # np.nan_to_num(cone_map,copy=False)
    # np.nan_to_num(allfx,copy=False)
    # print(allfx.max())
    # print(cone_map.max())
    try:
        corr = stats.pearsonr(cone_map, allfx[0,:])[0]
    except:
        return 0
    return corr

if __name__=="__main__":
    # The dlpfc mask is slightly larger than the brain mask\
    print("loading files")
    dlpfc = image.load_img("standard_data/spalm_dlPFC_GMmasked_03272023.nii.gz").get_fdata()
    brain_img = ds.get_img("MNI152_T1_2mm_brain_mask")
    brain = brain_img.get_fdata()
    
    masker = maskers.NiftiMasker(ds.get_img("MNI152_T1_2mm_brain_mask_dil")).fit()
    allfx = masker.transform("standard_data/AllFX_wmean.nii.gz")
    tc = masker.transform(sys.argv[1])
    print("files loaded")
    surface_dat = np.zeros(brain.shape)
    searchlight_result = np.zeros(brain.shape)
    surface_coords = []
    for c in np.ndindex(brain.shape):
        if brain[c] == 0 or dlpfc[c] == 0:
            continue
        adj = adjacent_voxels(c[0], c[1], c[2], brain.shape)
        for a in adj:
            if brain[a] == 0:
                surface_dat[c] = 1
                surface_coords.append(c)
                break
    surface_img = image.new_img_like(brain_img, surface_dat)
    surface_masker = maskers.NiftiMasker(surface_img).fit()
    args = [[brain_img, masker, allfx, tc, k] for k in surface_coords]
    os.nice(15)
    with Pool(8) as p:
        searchlight_result = np.asarray(p.map(searchlight_vox,args))
        print(searchlight_result.shape)
        searchlight_result = surface_masker.inverse_transform(searchlight_result)
    max_coord = np.unravel_index(searchlight_result.get_fdata().argmax(), searchlight_result.shape)
    searchlight_result.to_filename(sys.argv[2])