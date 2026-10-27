param(
    [string]$Model = 'qwen3.5:4b',
    [string]$OutDir = 'results/raw'   # gitignored
)

# Step 5a. What does the model see when one prompt is bigger than num_ctx?
# The prompt: a fact at the TOP, numbered log lines (L0001...), a question at the END.
#   Part 1: prompt overflow. Real token count (measured at a large num_ctx) vs prompt_eval_count.
#           Hypothesis from Lab 02: kept = 4 + (num_ctx - 4) / 2 (first 4 tokens + last half).
#   Part 2: undocumented `truncate: false`. Error, or silently ignored?
#   Part 3: generation overflow. The prompt fits, but prompt + answer doesn't.
# New server.log lines mentioning truncation/shift are shown after each request.

$api = 'http://127.0.0.1:11434'
$log = Join-Path $env:LOCALAPPDATA 'Ollama\server.log'
$countCtx = 16384   # big enough to hold every prompt here untruncated (4b at 16K ≈ 4.3 GB)

function New-Prompt([int]$lines, [string]$question) {
    $body = (1..$lines | ForEach-Object {
        'L{0:0000} 2026-10-27T12:{1:00}:{2:00}Z pod/api-{3} ns=prod level=INFO msg="request served" latency_ms={4}' -f $_, ([math]::Floor($_ / 60) % 60), ($_ % 60), ($_ % 7), (100 + ($_ * 37) % 900)
    }) -join "`n"
    "The cluster name is blue-falcon. Remember it.`n`nLog:`n$body`n`n$question"
}
$askFirst = 'What is the cluster name, and what is the first log line number (Lxxxx) you can see above? Reply exactly as: cluster=<name> first=<Lxxxx>'

function Invoke-Gen([string]$prompt, [hashtable]$opts, [hashtable]$extra = @{}) {
    $before = (Get-Content $log).Count
    $req = @{ model = $Model; prompt = $prompt; stream = $false; think = $false; options = $opts } + $extra
    try {
        $r = Invoke-RestMethod "$api/api/generate" -Method Post -Body ($req | ConvertTo-Json -Depth 4) -ContentType 'application/json'
        $err = $null
    } catch { $r = $null; $err = "$($_.ErrorDetails.Message)" }
    $logHits = Get-Content $log | Select-Object -Skip $before | Where-Object { $_ -match 'truncat|shift|exceed|context limit' } |
        ForEach-Object { ($_ -replace '^time=\S+ ', '').Substring(0, [math]::Min(140, $_.Length)) }
    [pscustomobject]@{ R = $r; Error = $err; Log = @($logHits) }
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
$report = Join-Path $OutDir "context-overflow-$(Get-Date -Format 'yyyyMMdd-HHmmss').md"

# --- Part 1: prompt overflow ---
$cases = @(
    @{ Lines = 50;  Ctx = 4096 }   # fits
    @{ Lines = 120; Ctx = 4096 }   # ≈ Lab 02 size
    @{ Lines = 250; Ctx = 4096 }   # more than 2x the context
    @{ Lines = 120; Ctx = 8192 }   # same prompt, bigger context: fits?
)
$steps = $cases.Count * 2 + 3
$n = 0
# Count real prompt sizes first, all at one num_ctx (avoids a reload per case)
foreach ($c in $cases) {
    Write-Progress -Activity "context-overflow: $Model" -Status "Step $($n + 1) of ${steps}: counting tokens, $($c.Lines) lines" -PercentComplete ([int](100 * $n / $steps)); $n++
    $c.Prompt = New-Prompt $c.Lines $askFirst
    $c.SentTok = (Invoke-Gen $c.Prompt @{ num_ctx = $countCtx; num_predict = 1; temperature = 0 }).R.prompt_eval_count
}
Add-Content $report "# Part 1: prompt overflow ($Model)`n"
$part1 = foreach ($c in $cases) {
    Write-Progress -Activity "context-overflow: $Model" -Status "Step $($n + 1) of ${steps}: $($c.Lines) lines at num_ctx $($c.Ctx)" -PercentComplete ([int](100 * $n / $steps)); $n++
    $g = Invoke-Gen $c.Prompt @{ num_ctx = $c.Ctx; num_predict = 30; temperature = 0; seed = 42 }
    $hyp = if ($c.SentTok -gt $c.Ctx) { 4 + [math]::Floor(($c.Ctx - 4) / 2) } else { $c.SentTok }
    Add-Content $report "## $($c.Lines) lines, num_ctx $($c.Ctx)`n`nSent $($c.SentTok), kept $($g.R.prompt_eval_count)`n`nAnswer: $($g.R.response)`n`nLog: $($g.Log -join ' | ')`n"
    [pscustomobject]@{
        Lines    = $c.Lines
        NumCtx   = $c.Ctx
        SentTok  = $c.SentTok                     # real prompt size
        KeptTok  = $g.R.prompt_eval_count         # what the model processed
        HypTok   = $hyp                           # 4 + (num_ctx - 4) / 2 if it overflows
        Answer   = "$($g.R.response)".Trim() -replace '\s+', ' '
        LogLines = $g.Log.Count
        LogText  = $g.Log
    }
}
$part1 | Select-Object -Property * -ExcludeProperty LogText | Format-Table -AutoSize
"server.log lines about truncation (distinct):"
$part1.LogText | Sort-Object -Unique | ForEach-Object { "  $_" }

# --- Part 2: truncate=false (not in the API docs) ---
Write-Progress -Activity "context-overflow: $Model" -Status "Step $($n + 1) of ${steps}: truncate=false" -PercentComplete ([int](100 * $n / $steps)); $n++
$g = Invoke-Gen $cases[2].Prompt @{ num_ctx = 4096; num_predict = 30; temperature = 0; seed = 42 } @{ truncate = $false }
"`nPart 2, truncate=false on $($cases[2].SentTok) tokens at num_ctx 4096:"
if ($g.Error) { "  ERROR: $($g.Error)" } else { "  no error; kept $($g.R.prompt_eval_count) tokens; answer: $("$($g.R.response)".Trim())" }
Add-Content $report "# Part 2: truncate=false`n`nError: $($g.Error)`n`nKept: $($g.R.prompt_eval_count)`n`nAnswer: $($g.R.response)`n"

# --- Part 3: generation overflow ---
# ≈75 lines ≈ 3,000 tokens of prompt at num_ctx 4096 leaves ≈1,000 for the answer; ask for far more.
$genPrompt = New-Prompt 75 'Copy every log line above, verbatim, one per line. Do not summarize.'
Add-Content $report "# Part 3: generation overflow`n"
$part3 = foreach ($shift in @($null, $false)) {
    $label = if ($null -eq $shift) { 'default' } else { 'shift=false' }
    Write-Progress -Activity "context-overflow: $Model" -Status "Step $($n + 1) of ${steps}: generation overflow, $label" -PercentComplete ([int](100 * $n / $steps)); $n++
    $extra = if ($null -eq $shift) { @{} } else { @{ shift = $false } }   # `shift` is also not in the docs
    $g = Invoke-Gen $genPrompt @{ num_ctx = 4096; num_predict = 3000; temperature = 0; seed = 42 } $extra
    Add-Content $report "## $label`n`nError: $($g.Error)`n`nLog: $($g.Log -join ' | ')`n`n$($g.R.response)`n"
    [pscustomobject]@{
        Mode      = $label
        PromptTok = $g.R.prompt_eval_count
        GenTok    = $g.R.eval_count
        Total     = $g.R.prompt_eval_count + $g.R.eval_count   # > 4096 = generation continued past the context
        Done      = $g.R.done_reason
        Error     = $g.Error
        LogLines  = $g.Log.Count
    }
}
Write-Progress -Activity "context-overflow: $Model" -Completed
"`nPart 3, generation overflow (num_ctx 4096):"
$part3 | Format-Table -AutoSize
"Prompts' answers and server.log lines saved to $report"
