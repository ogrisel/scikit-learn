# URL of the Windows SDK for .NET 3.5 suitable for Python 2 builds
$SDK_ISO_URL = "http://download.microsoft.com/download/2/E/9/2E911956-F90F-4BFB-8231-E292A7B6F287/GRMSDKX_EN_DVD.iso"

# On some platforms you have to change the drive letter.
# On Azure the ISO will be mounted on drive F:.
$MOUNTED_ISO_DRIVE = "D:"

$DOWNLOAD_FOLDER = $HOME + "\Downloads"
$ISO_PATH = $DOWNLOAD_FOLDER + "\GRMSDKX_EN_DVD.iso"

wget -OutFile $ISO_PATH $SDK_ISO_URL
Mount-DiskImage $ISO_PATH
Install-WindowsFeature Net-Framework-Core
$MOUNTED_ISO_DRIVE\setup.exe /quiet
