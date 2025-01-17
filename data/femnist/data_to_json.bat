@echo off
REM Check and create directories if they don't exist

if not exist "..\..\dataset\FEMINIST" (
    mkdir ..\dataset\FEMINIST
)

if not exist "..\..\dataset\FEMINIST\raw" (
    echo ------------------------------
    echo Downloading data
    mkdir ..\..\dataset\FEMINIST\raw
    powershell -ExecutionPolicy Bypass -File get_data.ps1  REM Assuming get_data.sh is converted to get_data.ps1
    echo Finished downloading data
)

if not exist "..\..\dataset\FEMINIST\intermediate" (
    mkdir ..\..\dataset\FEMINIST\intermediate
)

if not exist "..\..\dataset\FEMINIST\intermediate\class_file_dirs.pkl" (
    echo ------------------------------
    echo Extracting file directories of images
    python get_file_dirs.py
    echo Finished extracting file directories of images
)

if not exist "..\..\dataset\FEMINIST\intermediate\class_file_hashes.pkl" (
    echo ------------------------------
    echo Calculating image hashes
    python get_hashes.py
    echo Finished calculating image hashes
)

if not exist "..\..\dataset\FEMINIST\intermediate\write_with_class.pkl" (
    echo ------------------------------
    echo Assigning class labels to write images
    python match_hashes.py
    echo Finished assigning class labels to write images
)

if not exist "..\..\dataset\FEMINIST\intermediate\images_by_writer.pkl" (
    echo ------------------------------
    echo Grouping images by writer
    python group_by_writer.py
    echo Finished grouping images by writer
)

if not exist "..\..\dataset\FEMINIST\all_data" (
    mkdir ..\..\dataset\FEMINIST\all_data
)

REM Check if the all_data directory is empty
for /f %%A in ('dir "..\..\dataset\FEMINIST\all_data" /b /a') do set DIR_NOT_EMPTY=true

if not defined DIR_NOT_EMPTY (
    echo ------------------------------
    echo Converting data to .json format
    F:\virtual_envs\anaconda_envs\torch_py38\python.exe data_to_json.py
    echo Finished converting data to .json format
)

REM Clean up
set DIR_NOT_EMPTY=
