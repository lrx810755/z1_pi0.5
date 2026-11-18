[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$IsaacSimRoot,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DatasetRoot = (Join-Path $HOME "smolvla_data"),
    [string]$Scene = "",
    [int]$RosDomainId = 0,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$resolvedRepo = (Resolve-Path $RepoRoot).Path
$resolvedIsaac = (Resolve-Path $IsaacSimRoot).Path
$assetRoot = Join-Path $resolvedRepo "isaac_assets\z1"
$python = Join-Path $resolvedIsaac "python.bat"
$script = Join-Path $assetRoot "scripts\run_z1_ros2_scene.py"

if (-not (Test-Path $python -PathType Leaf)) {
    throw "Isaac Sim Python launcher is missing: $python"
}
if (-not (Test-Path $script -PathType Leaf)) {
    throw "Z1 Isaac launcher is missing: $script"
}
if ([string]::IsNullOrWhiteSpace($Scene)) {
    $Scene = Join-Path $assetRoot "generated_dual_camera\smolvla_dual_wrist_camera.usd"
}

$env:Z1_REPO_ROOT = $resolvedRepo
$env:Z1_ASSET_ROOT = $assetRoot
$env:Z1_DATASET_ROOT = $DatasetRoot
$env:ISAAC_SIM_ROOT = $resolvedIsaac
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
$env:ROS_DOMAIN_ID = $RosDomainId.ToString()

$isaacArgs = @($script, "--scene", $Scene)
if ($Headless) {
    $isaacArgs += "--headless"
}

& $python @isaacArgs
exit $LASTEXITCODE
