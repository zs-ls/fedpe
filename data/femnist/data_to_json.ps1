# Check and create directories if they don't exist

if (-Not (Test-Path "..\dataset\FEMINIST")) {
    New-Item -ItemType Directory -Force -Path "..\dataset\FEMINIST"
}

if (-Not (Test-Path "..\dataset\FEMINIST\raw")) {
    Write-Host "------------------------------"
    Write-Host "Downloading data"
    New-Item -ItemType Directory -Force -Path "..\dataset\FEMINIST\raw"
    .\get_data.ps1  # Assuming get_data.sh is converted to get_data.ps1
    Write-Host "Finished downloading data"
}

if (-Not (Test-Path "..\dataset\FEMINIST\intermediate")) {
    New-Item -ItemType Directory -Force -Path "..\dataset\FEMINIST\intermediate"
}

if (-Not (Test-Path "..\dataset\FEMINIST\intermediate\class_file_dirs.pkl")) {
    Write-Host "------------------------------"
    Write-Host "Extracting file directories of images"
    python get_file_dirs.py
    Write-Host "Finished extracting file directories of images"
}

if (-Not (Test-Path "..\dataset\FEMINIST\intermediate\class_file_hashes.pkl")) {
    Write-Host "------------------------------"
    Write-Host "Calculating image hashes"
    python get_hashes.py
    Write-Host "Finished calculating image hashes"
}

if (-Not (Test-Path "..\dataset\FEMINIST\intermediate\write_with_class.pkl")) {
    Write-Host "------------------------------"
    Write-Host "Assigning class labels to write images"
    python match_hashes.py
    Write-Host "Finished assigning class labels to write images"
}

if (-Not (Test-Path "..\dataset\FEMINIST\intermediate\images_by_writer.pkl")) {
    Write-Host "------------------------------"
    Write-Host "Grouping images by writer"
    python group_by_writer.py
    Write-Host "Finished grouping images by writer"
}

if (-Not (Test-Path "..\dataset\FEMINIST\all_data")) {
    New-Item -ItemType Directory -Force -Path "..\data\all_data"
}

if (-Not (Test-Path "..\dataset\FEMINIST\all_data" -and (Get-ChildItem -Path "..\dataset\FEMINIST\all_data" | Measure-Object -Property Name -Count).Count)) {
    Write-Host "------------------------------"
    Write-Host "Converting data to .json format"
    python data_to_json.py
    Write-Host "Finished converting data to .json format"
}
