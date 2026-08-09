param(
    [switch]$InstallDeps,
    [switch]$OneFile,
    [int]$SampleCount = 5
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$AppName = "SanashPassengerCounter"
$BuildName = if ($OneFile) { "$AppName-OneFile" } else { $AppName }
$DistRoot = Join-Path $ProjectRoot "dist"
$AppDist = Join-Path $DistRoot $AppName
$OneFileExe = Join-Path $DistRoot "$BuildName.exe"
$ZipPath = Join-Path $DistRoot "$AppName-Windows.zip"
$ModelPath = Join-Path $ProjectRoot "sanash_p2pnet_artifacts.zip"
$IconPath = Join-Path $ProjectRoot "sanash_demo\assets\sanash_icon.ico"
$SampleSource = Join-Path $ProjectRoot "p2pnet_almaty_dataset_block_stratified\images\val"
$OneFileSamples = Join-Path $ProjectRoot "build\onefile_resources\demo_samples"

function Assert-PathInsideProject {
    param([string]$PathToCheck)
    $root = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
    $target = [System.IO.Path]::GetFullPath($PathToCheck)
    if ($target -ne $root -and -not $target.StartsWith("$root\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the project: $target"
    }
}

function Compress-WithRetry {
    param(
        [string]$SourceDirectory,
        [string]$DestinationPath,
        [int]$Attempts = 5
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            if (Test-Path -LiteralPath $DestinationPath) {
                Remove-Item -LiteralPath $DestinationPath -Force
            }
            $sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
            $tar = Get-Command tar -ErrorAction SilentlyContinue
            if ($sevenZip) {
                Push-Location $SourceDirectory
                try {
                    & $sevenZip.Source a -tzip $DestinationPath ".\*" -mx=1 -bd -y
                    if ($LASTEXITCODE -notin @(0, 1)) {
                        throw "7z failed with exit code $LASTEXITCODE"
                    }
                }
                finally {
                    Pop-Location
                }
            }
            elseif ($tar) {
                & $tar.Source -a -cf $DestinationPath -C $SourceDirectory .
                if ($LASTEXITCODE -ne 0) {
                    throw "tar failed with exit code $LASTEXITCODE"
                }
            }
            else {
                Compress-Archive -Path (Join-Path $SourceDirectory "*") -DestinationPath $DestinationPath -Force -ErrorAction Stop
            }
            return
        }
        catch {
            if ($attempt -eq $Attempts) {
                throw
            }
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}

if ($InstallDeps) {
    python -m pip install --upgrade pip
    pip install -r requirements-demo.txt
}

python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed. Run: .\package_windows.ps1 -InstallDeps"
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model archive not found: $ModelPath"
}

if (-not $OneFile -and (Test-Path -LiteralPath $AppDist)) {
    Assert-PathInsideProject $AppDist
    Remove-Item -LiteralPath $AppDist -Recurse -Force
}
if (-not $OneFile -and (Test-Path -LiteralPath $ZipPath)) {
    Remove-Item -LiteralPath $ZipPath -Force
}
if ($OneFile -and (Test-Path -LiteralPath $OneFileExe)) {
    Assert-PathInsideProject $OneFileExe
    Remove-Item -LiteralPath $OneFileExe -Force
}

if ($OneFile) {
    New-Item -ItemType Directory -Force -Path $OneFileSamples | Out-Null
    Get-ChildItem -LiteralPath $OneFileSamples -File -ErrorAction SilentlyContinue | Remove-Item -Force
    if (Test-Path -LiteralPath $SampleSource) {
        Get-ChildItem -LiteralPath $SampleSource -File |
            Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' } |
            Sort-Object Name |
            Select-Object -First $SampleCount |
            Copy-Item -Destination $OneFileSamples -Force
    }
}

$PyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    $(if ($OneFile) { "--onefile" } else { "--onedir" }),
    "--windowed",
    "--name", $BuildName,
    "--hidden-import", "PIL._tkinter_finder",
    "--hidden-import", "onnxruntime",
    "--collect-binaries", "torch",
    "--collect-data", "torch",
    "--collect-all", "PIL",
    "--collect-binaries", "onnxruntime",
    "--collect-data", "onnxruntime",
    "--exclude-module", "datasets",
    "--exclude-module", "IPython",
    "--exclude-module", "jedi",
    "--exclude-module", "matplotlib",
    "--exclude-module", "pandas",
    "--exclude-module", "pytest",
    "--exclude-module", "scipy",
    "--exclude-module", "sklearn",
    "--exclude-module", "tensorflow",
    "--exclude-module", "torchaudio",
    "--exclude-module", "torchvision",
    "--exclude-module", "transformers"
)

if (Test-Path -LiteralPath $IconPath) {
    $PyInstallerArgs += @("--icon", $IconPath)
}

if ($OneFile) {
    $PyInstallerArgs = @(
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name", $BuildName,
        "--hidden-import", "PIL._tkinter_finder",
        "--hidden-import", "onnxruntime",
        "--collect-binaries", "torch",
        "--collect-data", "torch",
        "--collect-all", "PIL",
        "--collect-binaries", "onnxruntime",
        "--collect-data", "onnxruntime",
        "--exclude-module", "datasets",
        "--exclude-module", "IPython",
        "--exclude-module", "jedi",
        "--exclude-module", "matplotlib",
        "--exclude-module", "pandas",
        "--exclude-module", "pytest",
        "--exclude-module", "scipy",
        "--exclude-module", "sklearn",
        "--exclude-module", "tensorflow",
        "--exclude-module", "torchaudio",
        "--exclude-module", "torchvision",
        "--exclude-module", "transformers",
        "--add-data", "sanash_p2pnet_artifacts.zip;.",
        "--add-data", "README_SANASH_DEMO.md;.",
        "--add-data", "build\onefile_resources\demo_samples;demo_samples"
    )
    if (Test-Path -LiteralPath $IconPath) {
        $PyInstallerArgs += @("--icon", $IconPath)
    }
}

$PyInstallerArgs += "run_demo.py"

python -m PyInstaller @PyInstallerArgs

if ($OneFile) {
    Write-Host ""
    Write-Host "Packaged one-file demo:"
    Write-Host "  $OneFileExe"
    Write-Host ""
    Write-Host "Send this EXE to a Windows user. First launch can be slow while it extracts bundled libraries."
    exit 0
}

Copy-Item -LiteralPath $ModelPath -Destination (Join-Path $AppDist "sanash_p2pnet_artifacts.zip") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "README_SANASH_DEMO.md") -Destination (Join-Path $AppDist "README_SANASH_DEMO.md") -Force

$SamplesOut = Join-Path $AppDist "demo_samples"
New-Item -ItemType Directory -Force -Path $SamplesOut | Out-Null
if (Test-Path -LiteralPath $SampleSource) {
    Get-ChildItem -LiteralPath $SampleSource -File |
        Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' } |
        Sort-Object Name |
        Select-Object -First $SampleCount |
        Copy-Item -Destination $SamplesOut -Force
}

Compress-WithRetry -SourceDirectory $AppDist -DestinationPath $ZipPath

Write-Host ""
Write-Host "Packaged demo:"
Write-Host "  $ZipPath"
Write-Host ""
Write-Host "Send this ZIP to a Windows user. They should unzip it and run $AppName.exe."
