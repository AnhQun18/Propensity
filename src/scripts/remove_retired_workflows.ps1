$ErrorActionPreference = 'Stop'
$workspacePath = [System.IO.Path]::GetFullPath((Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$planFile = Join-Path $workspacePath 'docs\cleanup_20261116\deletion_plan.json'
$plan = Get-Content -LiteralPath $planFile -Raw -Encoding utf8 | ConvertFrom-Json
if ($plan.workspace -ne $workspacePath -or $plan.status -ne 'planned') { throw 'Plan/workspace mismatch or plan already applied.' }
$separator = [System.IO.Path]::DirectorySeparatorChar
$protectedPaths = @('.git', 'data', 'sample_data', 'docs\report', 'outputs\dataset_eda', 'outputs\mbd_review_cache', 'outputs\propensity_processing_v3') | ForEach-Object { [System.IO.Path]::GetFullPath((Join-Path $workspacePath $_)) }
# Check all absolute paths before the first deletion; never evaluate shell-built path strings.
foreach ($entry in $plan.items) {
    $targetPath = [System.IO.Path]::GetFullPath((Join-Path $workspacePath $entry.path))
    if ($targetPath -ne $entry.absolute_path -or -not $targetPath.StartsWith($workspacePath + $separator, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe target: $targetPath" }
    foreach ($protectedPath in $protectedPaths) {
        if ($targetPath -eq $protectedPath -or $targetPath.StartsWith($protectedPath + $separator, [System.StringComparison]::OrdinalIgnoreCase) -or $protectedPath.StartsWith($targetPath + $separator, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Protected target: $targetPath" }
    }
    if (Test-Path -LiteralPath $targetPath) {
        $targetItem = Get-Item -LiteralPath $targetPath -Force
        if ($targetItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $targetPath" }
    }
}
$results = [System.Collections.Generic.List[object]]::new()
foreach ($entry in $plan.items) {
    try {
        if (Test-Path -LiteralPath $entry.absolute_path) { Remove-Item -LiteralPath $entry.absolute_path -Recurse -Force }
        if (Test-Path -LiteralPath $entry.absolute_path) { throw 'Target still exists' }
        $results.Add([pscustomobject]@{ path=$entry.path; status='deleted' })
    } catch {
        $results.Add([pscustomobject]@{ path=$entry.path; status='failed'; error=$_.Exception.Message })
    }
    $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $workspacePath 'docs\cleanup_20261116\deletion_results.json') -Encoding utf8
}
$failed = @($results | Where-Object status -eq 'failed')
Write-Output "Deleted targets: $($results.Count - $failed.Count) / $($results.Count)"
if ($failed.Count -gt 0) { $failed | Format-Table -AutoSize; exit 1 }
