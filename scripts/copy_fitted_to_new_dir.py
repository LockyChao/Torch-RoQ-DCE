#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Collect DCE ROI‐fit KEP segment files into one folder and rename them."
    )
    parser.add_argument(
        "--src-filename", "-s",
        required=True,
        help="Name of the file to search for in each source directory (e.g. DCE_T1_ROI_delay_fitkep_seg_body.mat)"
    )
    parser.add_argument(
        "--dest-dir", "-d",
        required=True,
        help="Directory where all matched files will be copied and renamed into"
    )
    args = parser.parse_args()

    # 1) List of all the directories to scan
    filepaths=[

        
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221115/meas_MID00058_FID75709_low_dose_Multitasking_1st_moco_20250122T203719",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221116/1st_moco_0205",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221123/meas_MID00054_FID76437_low_dose_Multitasking_1st_moco_20250123T124728",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221201_004/meas_MID00094_FID77061_low_dose_Multitasking_1st_moco_20250123T185325",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221201_006/1st_moco_0205",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221206/meas_MID00127_FID77465_low_dose_Multitasking_1st_moco_20250127T020213",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221221/1st_moco_0205",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221222/meas_MID00057_FID78610_low_dose_Multitasking_1st_moco_20250128T021523",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230124/1st_moco_0205",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230125/1st_moco_0205",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230126/meas_MID00109_FID80688_low_dose_Multitasking_1st_moco_20250126T214846",

    # regular
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221115/meas_MID00058_FID75709_low_dose_Multitasking_1st_reg_20250115T175925",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221116/meas_MID00150_FID75961_low_dose_Multitasking_1st_reg_20250117T133141",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221123/meas_MID00054_FID76437_low_dose_Multitasking_1st_reg_20250117T144856",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221201_004/meas_MID00094_FID77061_low_dose_Multitasking_1st_reg_20250117T211634",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221201_006/meas_MID00039_FID77006_low_dose_Multitasking_1st_reg_20250118T094830",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221206/meas_MID00127_FID77465_low_dose_Multitasking_1st_reg_20250118T163623",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221221/meas_MID00056_FID78482_low_dose_Multitasking_1st_reg_20250119T003456",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221222/meas_MID00057_FID78610_low_dose_Multitasking_1st_reg_20250128T121348",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230124/meas_MID00059_FID80488_low_dose_Multitasking_1st_reg_20250122T151835",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230125/meas_MID00054_FID80577_low_dose_Multitasking_1st_reg_20250126T132402",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230126/meas_MID00109_FID80688_low_dose_Multitasking_1st_reg_20250126T194558",



        # 2nd full
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221115/meas_MID00087_FID75738_low_dose_Multitasking_2nd_20250122T213007_ng2",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221115/meas_MID00112_FID75763_full_dose_Multitasking_20250123T004620_ng2",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221116/meas_MID00156_FID75969_low_dose_Multitasking_2nd_20250123T083605",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221116/meas_MID00165_FID75978_full_dose_Multitasking_20250123T110720",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221123/meas_MID00079_FID76462_low_dose_Multitasking_2nd_20250123T164608",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221123/meas_MID00087_FID76470_full_dose_Multitasking_20250123T200556",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221201_004/meas_MID00109_FID77076_low_dose_Multitasking_2nd_20250123T234024",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221201_004/meas_MID00116_FID77083_full_dose_Multitasking_20250124T113317",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221201_006/meas_MID00043_FID77010_low_dose_Multitasking_2nd_20250124T172328",
    # "/mnt/LiDXXLab/Files/Chaowei/Low-dose Study/20221201_006/meas_MID00046_FID77013_full_dose_Multitasking_20250126T015259",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221206/meas_MID00134_FID77472_low_dose_Multitasking_2nd_20250126T163609",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221206/meas_MID00139_FID77477_low_dose_Multitasking_full_dose_20250126T184913",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221221/meas_MID00083_FID78509_low_dose_Multitasking_2nd_20250126T223952",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221222/meas_MID00081_FID78634_low_dose_Multitasking_2nd_moco_20250129T182402",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20221222/meas_MID00082_FID78635_full_dose_Multitasking_moco_20250130T022353",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230124/meas_MID00100_FID80529_full_dose_Multitasking_moco_20250130T164558",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230125/meas_MID00059_FID80582_low_dose_Multitasking_2nd_moco_20250130T211949",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230125/meas_MID00062_FID80585_full_dose_Multitasking_moco_20250131T012025",

    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230126/meas_MID00113_FID80692_low_dose_Multitasking_2nd_moco_20250131T091830",
    "/mnt/LiDXXLab/Files/Chaowei/Low-dose-Study/20230126/meas_MID00118_FID80697_full_dose_Multitasking_moco_20250131T133207",
    ];

    # 2) Your two maps, translated to Python dicts
    subject_map = {
        '20221115':'HC01','20221116':'HC02','20221123':'HC03',
        '20221201_004':'HC04','20221201_006':'HC05','20221206':'HC06',
        '20221221':'HC07','20221222':'HC08','20230124':'HC09',
        '20230125':'HC10','20230126':'HC11'
    }

    scan_map = {
        '1st_moco':'LD1',
        '1st_reg':'LD1',
        '2nd':'LD2',
        'full':'SD'
    }

    # 3) Ensure the destination exists
    dest_root = Path(args.dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    # 4) Loop & copy
    for d in filepaths:
        src_folder = Path(d)
        src_file   = src_folder / args.src_filename

        if not src_file.is_file():
            print(f"[MISSING] {args.src_filename} not found in {d}")
            continue

        # extract the date-block from the parent folder name
        date_blk = src_folder.parent.name
        subj_code = subject_map.get(date_blk)
        if subj_code is None:
            print(f"[ERROR] Unknown subject date '{date_blk}' in path {d}")
            continue

        # figure out which scan-key is in the path
        scan_code = None
        for key, code in scan_map.items():
            if key in d:
                scan_code = code
                break
        if scan_code is None:
            print(f"[ERROR] No scan type (1st_moco/1st_reg/2nd/full) found in {d}")
            continue

        # build the new filename
        new_name = f"{subj_code}_{scan_code}_seg_body.mat"
        dest_path = dest_root / new_name

        # actually copy (preserves timestamps, permissions, etc.)
        shutil.copy2(src_file, dest_path)
        print(f"[COPIED] {src_file} → {dest_path}")

if __name__ == "__main__":
    main()
