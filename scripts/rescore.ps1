param(
    [Parameter(Mandatory)][string[]]$Files,                 # results/raw/quality-*.md written by quality.ps1
    [string]$Script = (Join-Path $PSScriptRoot 'quality.ps1')
)

# Re-applies the current Checks/Wrong patterns of quality.ps1 to saved answers, offline
# (no model needed). Use it after changing the patterns: OldScore/OldWrong are what the
# file recorded at run time, NewScore/NewWrong what the current patterns give.

# Take the $questions = @(...) assignment out of quality.ps1 so the patterns live in one place
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$null)
$assign = $ast.Find({ $args[0] -is [System.Management.Automation.Language.AssignmentStatementAst] -and $args[0].Left.Extent.Text -eq '$questions' }, $true)
$questions = Invoke-Expression $assign.Right.Extent.Text
$byId = @{}; foreach ($q in $questions) { $byId[$q.Id] = $q }

foreach ($f in $Files) {
    $text  = Get-Content $f -Raw
    $model = ([regex]::Match($text, '(?m)^# (\S+)')).Groups[1].Value
    # Section header written by quality.ps1: "## <id> (<hits>/<checks>, wrong <n>, <done_reason>)"
    $sections = [regex]::Matches($text, '(?ms)^## (\S+) \(([^,]+), wrong (\d+), (\w+)\)\r?\n\r?\n(.*?)(?=^## |^# |\z)')
    foreach ($s in $sections) {
        $q = $byId[$s.Groups[1].Value]
        if (-not $q) { continue }   # question no longer in quality.ps1
        $answer = ($s.Groups[5].Value -replace '(?s)<details><summary>thinking</summary>.*?</details>\s*', '').Trim()
        $hits  = @($q.Checks | Where-Object { $answer -match $_ }).Count
        $wrong = @($q.Wrong  | Where-Object { $answer -match $_ }).Count
        [pscustomobject]@{
            File     = Split-Path $f -Leaf
            Model    = $model
            Question = $q.Id
            OldScore = $s.Groups[2].Value
            OldWrong = [int]$s.Groups[3].Value
            NewScore = "$hits/$($q.Checks.Count)"
            NewWrong = $wrong
            Done     = $s.Groups[4].Value
        }
    }
}
