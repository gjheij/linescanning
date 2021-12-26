##################################################################################################################
#
#   This script uses freeview to make screenshots of a freesurfer segmentation.
#   These screenshots are then combined to a movie, for easy diagnosis of
#   FreeSurfer segmentation problems. I have chosen for sagittal orientation,
#   because this allows one to see whether the sagittal sinus is misclassified as V1 gray matter.
#
##################################################################################################################

import subprocess as sb
import os
import glob

freeview_command = 'freeview -cmd {cmd} '
cmd_txt = """ -v {anatomy}:grayscale=10,100 -f {lh_wm}:color=red:edgecolor=red -f {rh_wm}:color=red:edgecolor=red -f {lh_pial}:color=white:edgecolor=white -f {rh_pial}:color=white:edgecolor=white
 -viewport sagittal
 """

# it's also possible to load an EPI dataset, but you'll need a registration file also. The following string has to be appended to the cmd_txt
# -v {EPI}:colormap=jet:heatscale=20,200,500:reg={reg_filename}:opacity=0.65

# and we can also add labels, if we're interested in surface ROIs
# -l {lhV1label}:color=yellow -l {rhV1label}:color=yellow


# To step through the sagittal slices this is added for every slice.
slice_addition = ' -slice {xpos} 127 127 \n -ss {opfn} \n  '

experiment = 'pRF_norm'  # 'DSC_3018028.02_957' 'DSC_3018028.04_752'
freesurfer_subject_dir = os.environ['SUBJECTS_DIR']

subject_lists = range(1,8)
slices = range(90, 240)  # the slices in the anatomy to show. don't want to show a bunch of nothingness outside of the brain.


for sji in subject_lists:  # list of subject indices
    # this subject should be in the freesurfer subjects directory FS_folder
    subject = 'sub-' + str(sji).zfill(3)
    FS_folder = os.path.join(freesurfer_subject_dir, subject)

    # here, another example of an experiment, in which there were EPIs and ROIs to also show
    # not necessary now.
    #     EPI_glob = os.path.join(
    #         funcfolder, f'*.nii')
    #     EPI = glob.glob(EPI_glob)[0]
    #     print('EPI i: ' + EPI)
    #     reg_filename = os.path.join(
    #         funcfolder, egister.dat')
    # labels are in the FS directory
    #     # lhV1label = os.path.join(FS_folder, 'label', 'lh.V1.label')
    #     # rhV1label = os.path.join(FS_folder, 'label', 'rh.V1.label')

    if os.path.exists(FS_folder):

        target_directory = os.path.join(FS_folder, 'movie')

        if not os.path.exists(target_directory):
            os.makedirs(target_directory, exist_ok=True)

            cmd_file = os.path.join(target_directory, 'cmd.txt')

            sj_cmd = cmd_txt.format(
                anatomy=os.path.join(FS_folder, 'mri', 'T1.mgz'),
                lh_wm=os.path.join(FS_folder, 'surf', 'lh.white'),
                lh_pial=os.path.join(FS_folder, 'surf', 'lh.pial'),
                rh_wm=os.path.join(FS_folder, 'surf', 'rh.white'),
                rh_pial=os.path.join(FS_folder, 'surf', 'rh.pial'),
                subject=subject,
                # EPI=EPI,
                # reg_filename=reg_filename
                # lhV1label=lhV1label,
                # rhV1label=rhV1label
            )

            for sag_slice in slices:

                sj_cmd += slice_addition.format(
                    xpos=sag_slice,
                    opfn=os.path.join(target_directory, str(
                        sag_slice).zfill(3) + '.png')
                )

            sj_cmd += ' -quit \n '

            with open(cmd_file, 'w') as f:
                f.write(sj_cmd)

            sb.call(freeview_command.format(cmd=cmd_file), shell=True)

# calling this in a separate for loop for efficiency (this can be done headlessly, the freesurfer stuff cannot)
for sji in subject_lists:
    subject = 'sub-' + str(sji).zfill(3)
    target_directory = os.path.join(freesurfer_subject_dir, subject, 'movie')
    out_movie = os.path.join(target_directory, f'{subject}.mp4')

    if not os.path.exists(out_movie) and os.path.exists(target_directory):
        convert_command = f'ffmpeg -framerate 5 -pattern_type glob -i "{target_directory}/*.png" -b:v 2M -c:v mpeg4 {out_movie}'
        sb.call(convert_command, shell=True)
