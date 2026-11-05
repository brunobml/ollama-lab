# Lab 09 Step 3: a read-only snapshot of the user-level places an installer can change, to diff before/after.
# Usage: pwsh scripts\system_snapshot.ps1 -Out results\raw\snapshot-before.json
#        pwsh scripts\system_snapshot.ps1 -Out results\raw\snapshot-after.json -Compare results\raw\snapshot-before.json
# Reads only. It records names and paths, not file contents.
param([Parameter(Mandatory)] [string] $Out, [string] $Compare)

function Names($path) { if (Test-Path $path) { @(Get-ChildItem -Force $path | ForEach-Object Name | Sort-Object) } else { @() } }
function RegValues($key) {
    if (-not (Test-Path $key)) { return @() }
    $p = Get-ItemProperty $key
    @($p.PSObject.Properties | Where-Object Name -notlike 'PS*' | ForEach-Object { "$($_.Name) = $($_.Value)" } | Sort-Object)
}

$snap = [ordered]@{
    taken             = (Get-Date).ToString('s')
    user_path         = @([Environment]::GetEnvironmentVariable('Path', 'User') -split ';' | Where-Object { $_ })
    machine_path      = @([Environment]::GetEnvironmentVariable('Path', 'Machine') -split ';' | Where-Object { $_ })
    run_hkcu          = RegValues 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
    run_hklm          = RegValues 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run'
    startup_folder    = Names "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup"
    start_menu        = Names "$env:APPDATA\Microsoft\Windows\Start Menu\Programs"
    uninstall_hkcu    = @(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall' -ErrorAction SilentlyContinue |
                          ForEach-Object { "$($_.PSChildName) | $((Get-ItemProperty $_.PSPath).DisplayName) | $((Get-ItemProperty $_.PSPath).InstallLocation)" } | Sort-Object)
    url_protocols     = @(Get-ChildItem 'HKCU:\Software\Classes' -ErrorAction SilentlyContinue |
                          Where-Object { (Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue).'URL Protocol' -ne $null } | ForEach-Object PSChildName | Sort-Object)
    scheduled_tasks   = @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object TaskPath -notlike '\Microsoft\*' | ForEach-Object { "$($_.TaskPath)$($_.TaskName)" } | Sort-Object)
    services          = @(Get-Service | ForEach-Object Name | Sort-Object)
    firewall_rules    = @(Get-NetFirewallRule -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -match 'LM|lms|llm|Studio' } | ForEach-Object { "$($_.DisplayName) | $($_.Direction) | $($_.Action) | $($_.Enabled)" } | Sort-Object)
    localappdata      = Names $env:LOCALAPPDATA
    localappdata_prog = Names "$env:LOCALAPPDATA\Programs"
    appdata           = Names $env:APPDATA
    home_dotdirs      = @(Names $HOME | Where-Object { $_ -like '.*' })
    program_files     = Names $env:ProgramFiles
    listening_ports   = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | ForEach-Object { "$($_.LocalAddress):$($_.LocalPort) pid=$($_.OwningProcess)" } | Sort-Object -Unique)
}
$snap | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $Out
"snapshot written: $Out"

if ($Compare) {
    $old = Get-Content $Compare -Raw | ConvertFrom-Json
    foreach ($k in $snap.Keys) {
        if ($k -eq 'taken') { continue }
        $a = @($old.$k); $b = @($snap[$k])
        $added = @($b | Where-Object { $_ -notin $a }); $removed = @($a | Where-Object { $_ -notin $b })
        if ($added -or $removed) {
            "## $k"
            $added | ForEach-Object { "  + $_" }
            $removed | ForEach-Object { "  - $_" }
        }
    }
}
