# SignBridge - Windows PowerShell Downloader for Zenodo Record 4010759 (INCLUDE)
# Source: https://zenodo.org/records/4010759
#
# Usage:
#   .\download_data.ps1 -Category Greetings
#   .\download_data.ps1 -Category Days_and_Time
#   .\download_data.ps1 -List

param (
    [string]$Category = "Greetings",
    [switch]$List
)

$BaseUrl = "https://zenodo.org/record/4010759/files"
$DestDir = "$PSScriptRoot\..\datasets\include\zips"

if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
}

$Categories = [ordered]@{
    "Greetings"               = 2
    "Days_and_Time"           = 3
    "Electronics"             = 2
    "Animals"                 = 2
    "Clothes"                 = 2
    "Colours"                 = 2
    "Pronouns"                = 2
    "Seasons"                 = 1
    "Jobs"                    = 2
    "Home"                    = 4
    "Places"                  = 4
    "Means_of_Transportation" = 2
    "Society"                 = 3
    "People"                  = 5
    "Adjectives"              = 8
}

if ($List) {
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host " Available Categories in Zenodo Record 4010759 (INCLUDE)" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
    foreach ($cat in $Categories.Keys) {
        Write-Host " - $cat ($($Categories[$cat]) zip files)" -ForegroundColor Yellow
    }
    Write-Host "`nUsage: .\download_data.ps1 -Category Greetings" -ForegroundColor Green
    exit 0
}

if (-not $Categories.Contains($Category)) {
    Write-Host "[!] Unknown category '$Category'. Available categories:" -ForegroundColor Red
    $Categories.Keys | ForEach-Object { Write-Host "   - $_" }
    exit 1
}

$count = $Categories[$Category]
Write-Host "`n[*] Starting download for '$Category' ($count files)..." -ForegroundColor Cyan

for ($i = 1; $i -le $count; $i++) {
    $fileName = "${Category}_${i}of${count}.zip"
    $url = "$BaseUrl/$fileName"
    $output = "$DestDir\$fileName"

    if (Test-Path $output) {
        Write-Host "[+] Already exists: $fileName" -ForegroundColor Green
        continue
    }

    Write-Host "[*] Downloading: $fileName ..." -ForegroundColor Yellow
    curl.exe -L --progress-bar "$url" -o "$output"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Saved: $output" -ForegroundColor Green
    } else {
        Write-Host "[!] Download error for $fileName" -ForegroundColor Red
    }
}

Write-Host "`n[OK] Category '$Category' downloaded to $DestDir" -ForegroundColor Green
