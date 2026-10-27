param(
    [string]$Model = 'qwen3.5:4b',
    [int]$Runs = 20,
    [double]$Temperature = 0,
    [int]$Seed = 42,
    [switch]$NoSeed,                   # omit seed: Ollama picks a random one per request
    [int]$NumCtx = 4096,
    [int]$NumPredict = 200,
    [string]$Prompt = 'In one paragraph, explain the most common reasons a Kubernetes Pod ends up in CrashLoopBackOff.',
    [string]$OutDir = 'results/raw'    # gitignored
)

# Sends the same request $Runs times, sequentially, and counts how many distinct outputs come back.
# Thinking is off so every token is answer text.

$options = @{ temperature = $Temperature; num_ctx = $NumCtx; num_predict = $NumPredict }
if (-not $NoSeed) { $options.seed = $Seed }
$body = @{ model = $Model; prompt = $Prompt; stream = $false; think = $false; options = $options } | ConvertTo-Json -Depth 4

$sha  = [System.Security.Cryptography.SHA256]::Create()
$seen = [ordered]@{}   # hash -> first text
$rows = foreach ($i in 1..$Runs) {
    Write-Progress -Activity "repro: $Model, temperature $Temperature" -Status "Run $i of $Runs ($($seen.Count) distinct so far)" -PercentComplete ([int](100 * ($i - 1) / $Runs))
    $r = Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType 'application/json'
    $hash = -join ($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($r.response))[0..3] | ForEach-Object { $_.ToString('x2') })
    if (-not $seen.Contains($hash)) { $seen[$hash] = $r.response }
    [pscustomobject]@{
        Run       = $i
        Hash      = $hash
        Variant   = [array]::IndexOf(@($seen.Keys), $hash) + 1
        GenTok    = $r.eval_count
        PromptTok = $r.prompt_eval_count
        Cached    = $r.prompt_eval_cached_count   # prompt tokens reused from the prompt cache
        LoadS     = [math]::Round($r.load_duration / 1e9, 2)
        Done      = $r.done_reason
    }
}
Write-Progress -Activity "repro: $Model, temperature $Temperature" -Completed
$rows | Format-Table -AutoSize

$seedText = if ($NoSeed) { 'none' } else { $Seed }
"{0}: {1} runs, temperature {2}, seed {3}, num_ctx {4} -> {5} distinct output(s)" -f $Model, $Runs, $Temperature, $seedText, $NumCtx, $seen.Count
$rows | Group-Object Variant | Sort-Object Count -Descending | ForEach-Object { "  variant {0}: {1} run(s)" -f $_.Name, $_.Count }

New-Item -ItemType Directory -Force $OutDir | Out-Null
$report = Join-Path $OutDir ("repro-{0}.md" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Add-Content $report "# $Model, temperature $Temperature, seed $seedText, num_ctx $NumCtx, $Runs runs`n`nPrompt: $Prompt`n"
$v = 0
foreach ($h in $seen.Keys) { $v++; Add-Content $report "## Variant $v ($h)`n`n$($seen[$h])`n" }
"Distinct outputs saved to $report"
