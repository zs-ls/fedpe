# 假设脚本运行在 preprocess 文件夹中

# 切换到 raw_data 目录
Set-Location -Path "..\..\dataset\FEMINIST\raw"

# 下载数据文件
Write-Host "Downloading by_class.zip"
Invoke-WebRequest -Uri "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip" -OutFile "by_class.zip"

Write-Host "Downloading by_write.zip"
Invoke-WebRequest -Uri "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip" -OutFile "by_write.zip"

# 解压 zip 文件
Write-Host "Unzipping by_class.zip"
Expand-Archive -Path "by_class.zip" -DestinationPath "."

Write-Host "Unzipping by_write.zip"
Expand-Archive -Path "by_write.zip" -DestinationPath "."

# 删除 zip 文件
Write-Host "Removing by_class.zip"
Remove-Item -Path "by_class.zip"

Write-Host "Removing by_write.zip"
Remove-Item -Path "by_write.zip"

# # 切换回 preprocess 目录
Set-Location -Path "..\..\..\data\femnist"
