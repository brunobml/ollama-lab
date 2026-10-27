param(
    [string]$Model = 'qwen3.5:4b',
    [string]$QuestionId = 'helm-state',   # a question Id from quality.ps1
    [int]$Runs = 3,                       # runs per sampled condition (greedy ones run once: deterministic)
    [int]$NumPredict = 4096,
    [int]$NumCtx = 8192,
    [string[]]$Only,                      # run only these condition names, e.g. -Only greedy
    [string]$OutDir = 'results/raw'       # gitignored
)

# Step 4a. Lab 03 Step 1: with thinking on and temperature 0, helm-state looped until num_predict.
# Does sampling break the loop? Is it the randomness or the presence penalty?
# Effective sampler settings are read back from server.log, not assumed.

$api = 'http://127.0.0.1:11434'
$log = Join-Path $env:LOCALAPPDATA 'Ollama\server.log'

# Reuse the question text and Checks/Wrong patterns from quality.ps1 (same trick as rescore.ps1)
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'quality.ps1'), [ref]$null, [ref]$null)
$assign = $ast.Find({ $args[0] -is [System.Management.Automation.Language.AssignmentStatementAst] -and $args[0].Left.Extent.Text -eq '$questions' }, $true)
$q = (Invoke-Expression $assign.Right.Extent.Text) | Where-Object { $_.Id -eq $QuestionId }
if (-not $q) { throw "Unknown question id '$QuestionId'" }

# Options not listed fall back to the Modelfile (`ollama show $Model --parameters`)
$conditions = @(
    # presence_penalty must be set explicitly: the Modelfile default (1.5) applies otherwise.
    # (The 2026-10-27 13:10 run lacked it here, so 'greedy' and 'greedy+presence' were identical.)
    @{ Name = 'greedy';           Runs = 1;     Opts = @{ temperature = 0; seed = 42; presence_penalty = 0 } }
    @{ Name = 'greedy+presence';  Runs = 1;     Opts = @{ temperature = 0; seed = 42; presence_penalty = 1.5 } }
    @{ Name = 'defaults';         Runs = $Runs; Opts = @{} }                          # temp 1, top_k 20, top_p 0.95, presence 1.5
    @{ Name = 'defaults-presence';Runs = $Runs; Opts = @{ presence_penalty = 0 } }    # same, without the penalty
)

if ($Only) { $conditions = @($conditions | Where-Object { $_.Name -in $Only }) }

function Get-LastSamplerParams {
    # llama.cpp logs the effective sampler settings for each request ("sampler params: ...")
    $tail = (Get-Content $log -Tail 200) -join ' '
    $m = [regex]::Matches($tail, 'presence_penalty = ([\d.]+).*?top_k = (\S+?), top_p = (\S+?), min_p = (\S+?),.*?temp = (\S+)')
    if ($m.Count) { $g = $m[$m.Count - 1].Groups; "temp $($g[5].Value) top_k $($g[2].Value) top_p $($g[3].Value) min_p $($g[4].Value) presence $($g[1].Value)" }
    else { '(not in log)' }
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
$report = Join-Path $OutDir "sampling-loop-$(Get-Date -Format 'yyyyMMdd-HHmmss').md"

$total = ($conditions | ForEach-Object { $_.Runs } | Measure-Object -Sum).Sum
$n = 0
$rows = foreach ($c in $conditions) {
    for ($i = 1; $i -le $c.Runs; $i++) {
        # Updates between requests only: a single request can take ≈35 s when it loops
        Write-Progress -Activity "sampling-loop: $Model, $QuestionId" -Status "Request $($n + 1) of ${total}: $($c.Name), run $i of $($c.Runs)" -PercentComplete ([int](100 * $n / $total))
        $n++
        $opts = @{ num_predict = $NumPredict; num_ctx = $NumCtx } + $c.Opts
        $body = @{ model = $Model; prompt = $q.Q; stream = $false; think = $true; options = $opts } | ConvertTo-Json -Depth 4
        $r = Invoke-RestMethod "$api/api/generate" -Method Post -Body $body -ContentType 'application/json'
        $sampler = Get-LastSamplerParams

        # Loop signal: how often the most repeated (non-trivial) thinking line occurs
        $maxRepeat = (@("$($r.thinking)" -split '\r?\n' | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -ge 15 }) |
            Group-Object | Measure-Object -Property Count -Maximum).Maximum
        $hits  = @($q.Checks | Where-Object { $r.response -match $_ }).Count
        $wrong = @($q.Wrong  | Where-Object { $r.response -match $_ }).Count

        Add-Content $report "## $($c.Name) run $i ($($r.done_reason), $($r.eval_count) tokens, max line repeat $maxRepeat)`n`nSampler: $sampler`n`n<details><summary>thinking</summary>`n`n$($r.thinking)`n`n</details>`n`n$($r.response)`n"
        [pscustomobject]@{
            Condition = $c.Name
            Run       = $i
            Done      = $r.done_reason                 # 'length' = never reached an answer
            GenTok    = $r.eval_count
            Seconds   = [math]::Round($r.total_duration / 1e9, 1)
            MaxRepeat = [int]$maxRepeat                # ≥10 = looping
            Score     = "$hits/$($q.Checks.Count)"
            Wrong     = $wrong
            Sampler   = $sampler
        }
    }
}
Write-Progress -Activity "sampling-loop: $Model, $QuestionId" -Completed
$rows | Format-Table -AutoSize
"Answers saved to $report"
