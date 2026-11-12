$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$stagePath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot 'outputs\report_eda_verified'))
$reportPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot 'report'))

# Only replace the explicitly authorized report folder inside this workspace.
if ($reportPath -ne (Join-Path $projectRoot 'report') -or
    -not $stagePath.StartsWith($projectRoot + [System.IO.Path]::DirectorySeparatorChar) -or
    $stagePath -eq $reportPath -or
    -not (Test-Path -LiteralPath (Join-Path $projectRoot 'README.md'))) {
    throw 'Report publication paths failed workspace safety checks.'
}
$validation = Get-Content -LiteralPath (Join-Path $stagePath 'checks\validation.json') -Raw | ConvertFrom-Json
if (-not $validation.pass_all) { throw 'EDA validation has not passed.' }
if ((Get-FileHash -LiteralPath (Join-Path $projectRoot 'scripts\complete_report_eda.py') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $validation.script_sha256) {
    throw 'Validation belongs to another analysis script revision; rerun the analysis.'
}
$audit = Import-Csv -LiteralPath (Join-Path $stagePath 'brief_audit.csv')
if ($audit.Count -ne 45) { throw 'Expected all 45 section-12 requirements.' }
$manifest = Import-Csv -LiteralPath (Join-Path $stagePath 'manifest.csv')
foreach ($entry in $manifest) {
    $sourceFile = [System.IO.Path]::GetFullPath((Join-Path $stagePath $entry.file))
    if (-not $sourceFile.StartsWith($stagePath + [System.IO.Path]::DirectorySeparatorChar)) { throw 'Invalid manifest path.' }
    if ((Get-FileHash -LiteralPath $sourceFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw ('Staged output hash mismatch: ' + $entry.file)
    }
}
if (Test-Path -LiteralPath $reportPath) { Remove-Item -LiteralPath $reportPath -Recurse -Force }
New-Item -ItemType Directory -Path $reportPath -Force | Out-Null
# Stream file contents explicitly, including the manifest, then verify below.
foreach ($sourceItem in Get-ChildItem -LiteralPath $stagePath -Recurse -File) {
    $relative = [System.IO.Path]::GetRelativePath($stagePath, $sourceItem.FullName)
    $destination = [System.IO.Path]::GetFullPath((Join-Path $reportPath $relative))
    if (-not $destination.StartsWith($reportPath + [System.IO.Path]::DirectorySeparatorChar)) { throw 'Invalid publication destination.' }
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($destination)) | Out-Null
    $inputStream = [System.IO.File]::OpenRead($sourceItem.FullName)
    $outputStream = [System.IO.File]::Open($destination, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try { $inputStream.CopyTo($outputStream) }
    finally { $inputStream.Dispose(); $outputStream.Dispose() }
}
foreach ($entry in $manifest) {
    $publishedFile = Join-Path $reportPath $entry.file
    if ((Get-FileHash -LiteralPath $publishedFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw ('Published output hash mismatch: ' + $entry.file)
    }
}
Write-Output ('Published verified code-generated EDA outputs: ' + $reportPath)
Write-Output ('Verified files in manifest: ' + $manifest.Count)
