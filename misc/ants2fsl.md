# Convert ANTs to FSL

Most of the steps here are described in [this post](https://github.com/ANTsX/ANTs/issues/1293#issuecomment-1020242837), which shows how to convert ANTs non-linear warp fields to FSL-compatible warp fields. The output of these steps is a `reg` folder that you can immediately copy into `feat` folder. Please ALWAYS check the output of subsequent steps to make sure the transformations make sense! 

```bash
# define stuff
subID="002"
anatSes=1
boldSes=2

out_dir=${DIR_DATA_DERIV}/ants/sub-${subID}/ses-${anatSes}/reg
if [[ ! -d ${out_dir} ]]; then
    mkdir -p ${out_dir}
fi  
```

## T1w >> MNI
We can use `c3d_affine_tool` to tranlate the ANTs matrices to something that FSL understands. We start with the linear part, something that will be called `_0GenericAffine.mat` if you've used ANTs without fancy renaming.

### ANTs outside fMRIprep
If you've run module `05b` from the line-scanning repository, you'll have these files below. If you want to use the composite transform from fmriprep, scroll one section down :

```bash
src="${DIR_DATA_DERIV}/cat12/sub-${subID}/ses-${anatSes}/sub-${subID}_ses-${anatSes}_acq-MP2RAGE_T1w.nii.gz"
wrp="${DIR_DATA_DERIV}/ants/sub-${subID}/ses-${anatSes}/sub-${subID}_ses-${anatSes}_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.mat"
```

### ANTs from fMRIprep
We can also decompose the composite transform from fmriprep using `CompositeTransformUtil` (see also [this post](https://neurostars.org/t/extracting-individual-transforms-from-composite-h5-files-fmriprep/2215/12)):

```bash
h5_file=`find -L ${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${anatSes}/anat -type f -name "*from-T1w_to-MNI152NLin6Asym*" -and -name "*.h5"`

out_base=${out_dir}/highres2standard
CompositeTransformUtil --disassemble ${h5_file} ${out_base}
mv ${out_base}.nii.gz ${out_base}_warp.nii.gz
``` 

### Continue as is
```bash
ref="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
cp ${src} ${out_dir}/highres.nii.gz
cp ${ref} ${out_dir}/template.nii.gz

mat_fsl="${out_dir}/highres2standard.mat"

# run c3d
c3d_affine_tool -ref ${out_dir}/template.nii.gz -src ${out_dir}/highres.nii.gz -itk ${wrp} -ras2fsl -o ${mat_fsl}
```

Now we need to translate the non-linear warp file using the workbench `wb_command` command. This file will be called something like `_1Warp.nii.gz`
```bash
# define warp
nlin_wrp="${DIR_DATA_DERIV}/ants/sub-${subID}/ses-${anatSes}/sub-${subID}_ses-${anatSes}_from-MNI152NLin6Asym_to-T1w_mode-image_warp.nii.gz"
nlin_fsl="${out_dir}/tmp_warp.nii.gz"

# run wb command
wb_command -convert-warpfield -from-itk ${nlin_wrp} -to-fnirt ${nlin_fsl} ${out_dir}/template.nii.gz

# combine
convertwarp --ref=${out_dir}/template.nii.gz --premat=${mat_fsl} --warp1=${nlin_fsl} --out=${out_dir}/highres2standard_warp.nii.gz

# apply to create highres2standard.nii.gz
applywarp -i ${out_dir}/highres.nii.gz -r ${out_dir}/template.nii.gz -w ${out_dir}/highres2standard_warp.nii.gz -o ${out_dir}/highres2standard.nii.gz
```

## BOLD >> T1w
If you've run fmriprep with the line-scanning repository, you'll have a `from-bold_to-T1w` registration file. If not, this file is the output from `bbregister` and lives in the `fmriprep_wf`-folder.

```bash
wrp=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${boldSes}/func/sub-${subID}_ses-${boldSes}_task-RBL_acq-3DEPI_run-1_from-bold_to-T1w_mode-image_xfm.txt

# first convert to .mat format
wrp_mat=$(dirname ${wrp})/$(basename ${wrp} .txt).mat
ConvertTransformFile 3 ${wrp} ${wrp_mat} --convertToAffineType

# run c3d_affine_tool
boldref=${DIR_DATA_DERIV}/feat/level1/sub-${subID}/ses-${boldSes}/run1.feat/example_func.nii.gz
cp ${boldref} ${out_dir}/example_func.nii.gz
c3d_affine_tool -ref ${out_dir}/highres.nii.gz -src ${out_dir}/example_func.nii.gz -itk ${wrp_mat} -ras2fsl -o ${out_dir}/example_func2highres.mat

# apply to craete example_func2highres.nii.gz
applywarp -i ${out_dir}/example_func.nii.gz -r ${out_dir}/highres.nii.gz --postmat=${out_dir}/example_func2highres.mat -o ${out_dir}/example_func2highres.nii.gz
```

## Update FEAT directory
Now we can update the FEAT directory using `updatefeatreg`

```bash
# define feat-dir and copy 'reg' folder
ft_dir=$(dirname ${boldref})
cp -r ${out_dir} ${ft_dir}

# run updatefeatreg
updatefeatreg ${ft_dir}
```
