# Common errors during fMRIPrep

## FreeSurfer resampling

Sometimes fmriprep throws an error trying to resample data to the surface; the output message looks like this:

```bash
Loading label /data1/projects/MicroFunc/Jurjen/projects/PotentialSubjs/derivatives/freesurfer/fsaverage/label/lh.cortex.label
Standard error:
No such file or directory
mri_vol2surf: could not open label file /data1/projects/MicroFunc/Jurjen/projects/PotentialSubjs/derivatives/freesurfer/fsaverage/label/lh.cortex.label
Invalid argument
mri_vol2surf: could not load label file /data1/projects/MicroFunc/Jurjen/projects/PotentialSubjs/derivatives/freesurfer/fsaverage/label/lh.cortex.label
Invalid argument
Return code: 255
```

the solution that I've seen work is to remove the symlink to `fsaverage` in `SUBJECTS_DIR`. When re-running fmriprep, it reinitializes the symlink and sample the data properly. In my case, that would be:

```bash
rm /data1/projects/MicroFunc/Jurjen/projects/PotentialSubjs/derivatives/freesurfer/fsaverage
```
**IMPORTANT**: to remove symlinks, do NOT use `rm -r`, which will attempt to delete the actual folder. Use `rm` without `-r`-flag, and no `forward slash` at the end of the path

## Fieldmaps
### "_Missing `TotalReadoutTime`_"
Total readout time is required for TOPUP and can be calculated as:
```python
# Check PAR-file for water-fat shift
WFS = 38.445 
#magnetic field strength * water fat difference in ppm * gyromagnetic hydrogen ratio
WFS_hz = 7 * 3.35 * 42.576

# total readout time
TRT = WFS/WFS_hz

# EffectiveEchoSpacing
EES = TRT / EPI_FACTOR+1 # see PAR-file for epi factor
```

Then add this information to the files in `fmap/*_epi.nii.gz` and `func/*_bold.nii.gz`:
```python
bold_file = "ses-2/func/sub-001_ses-2_task-pRF_run-1_bold.nii.gz"

# decide phase encoding direction for EPI and BOLD files (must be opposite):
pedir = {'i': 'Left-Right', 'i-': 'Right-Left', 'j-': 'Anterior-Posterior', 'j': 'Posterior-Anterior'}

epi_pe = "j-"
epi_params = {"IntendedFor" : bold_file,
              "TotalReadoutTime" : TRT,
              "EffectiveEchoSpacing" : EES,
              "PhaseEncodingDirection" : epi_pe,
              "WaterFatShift" : WFS
             }

bold_pe = "j"
bold_params = {"TotalReadoutTime" : TRT,
               "EffectiveEchoSpacing" : EES,
               "PhaseEncodingDirection" : bold_pe,
               "WaterFatShift" : WFS,
               "MagneticFieldStrength" : 7
              }
```

### I have fieldmaps (topups or GRE) but susceptibility distortion correction was not executed
Most likely you forgot the `IntendedFor` field in the `fmap/_epi.json`-file (see above). If you have one fieldmap for multiple functional runs, you can enter a list:
```python
epi_params = {"IntendedFor" : ["path/to/func1.nii.gz",
                               "path/to/func1.nii.gz",
                               "path/to/func1.nii.gz",
                               "path/to/func1.nii.gz"]
             }
```

### "_Missing `SliceTiming`_": add slicetiming to the `fmap/_epi.json`-file:
```python
# check PAR-file
n_slices = 57
TR = 1.5
mb_factor = 3 

slc_timing = list(np.tile(np.linspace(0, TR, int(n_slices/mb_factor), endpoint=False), mb_factor))
```

Add this list to `"SliceTiming"` in json file

## Miscellaneous (the really annoying ones)
### "_pandas.errors.EmptyDataError: No columns to parse from file_"
This will generally mean your brain-mask is bad. See https://github.com/nipreps/fmriprep/issues/2761#issuecomment-1100142486. This can happen especially with Partial FOV inputs. I haven't found a good solution for this yet.

The mask is created during the `init_bold_reference_wf`-[workflow](https://github.com/nipreps/fmriprep/blob/master/fmriprep/workflows/bold/base.py#L486), right at the beginning:
```python
initial_boldref_wf = init_bold_reference_wf(
    name="initial_boldref_wf",
    omp_nthreads=omp_nthreads,
    bold_file=bold_file,
    sbref_files=[],
    multiecho=False,
)
```
