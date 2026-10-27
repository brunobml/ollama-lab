param(
    [string]$Model = 'qwen3.5:4b',
    [string]$Prompt = 'Explain in about 150 words what a Kubernetes readinessProbe is for.',
    [int]$NumPredict = 400,
    [switch]$Quiet,                    # don't print the tokens live
    [string]$OutDir = 'results/raw'    # gitignored
)

# Lab 04 Step 1. Streaming vs non-streaming on /api/chat, measured from the CLIENT side.
# Invoke-RestMethod only returns when the whole body has arrived, so this uses .NET HttpClient
# with ResponseHeadersRead and reads the NDJSON stream line by line (one JSON object per line).
#   Client TTFT  = time until the first chunk with text arrives (what a user waits for)
#   Client total = time until the last chunk (done: true)
# Server-side durations (load_duration, prompt_eval_duration, eval_duration) come in the final chunk.

$api = 'http://127.0.0.1:11434'
Add-Type -AssemblyName System.Net.Http
$http = [System.Net.Http.HttpClient]::new()
$http.Timeout = [TimeSpan]::FromMinutes(5)

function Invoke-Measured([bool]$stream, [bool]$think, [string]$label) {
    $body = @{ model = $Model; messages = @(@{ role = 'user'; content = $Prompt }); stream = $stream; think = $think
               options = @{ temperature = 0; seed = 42; num_predict = $NumPredict } } | ConvertTo-Json -Depth 5
    $req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$api/api/chat")
    $req.Content = [System.Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, 'application/json')

    $sw = [Diagnostics.Stopwatch]::StartNew()
    # ResponseHeadersRead: return as soon as headers arrive, don't buffer the body
    $resp = $http.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
    $reader = [IO.StreamReader]::new($resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult())

    $firstAny = $null; $firstAnswer = $null; $chunks = 0; $text = ''; $thinking = ''; $last = $null; $raw = @()
    if (-not $Quiet) { Write-Host "`n--- $label ---" -ForegroundColor Cyan }
    while ($null -ne ($line = $reader.ReadLine())) {
        if (-not $line) { continue }
        $chunks++
        if ($raw.Count -lt 3) { $raw += $line }          # keep the first few raw lines to show the wire format
        $obj = $line | ConvertFrom-Json
        $c = "$($obj.message.content)"; $t = "$($obj.message.thinking)"
        if (($c -or $t) -and $null -eq $firstAny) { $firstAny = $sw.Elapsed.TotalSeconds }
        if ($c -and $null -eq $firstAnswer) { $firstAnswer = $sw.Elapsed.TotalSeconds }
        $text += $c; $thinking += $t
        if (-not $Quiet) {
            if ($t) { Write-Host $t -NoNewline -ForegroundColor DarkGray }
            if ($c) { Write-Host $c -NoNewline }
        }
        if ($obj.done) { $last = $obj; $raw += $line }
    }
    $total = $sw.Elapsed.TotalSeconds
    $reader.Dispose(); $resp.Dispose()
    if (-not $Quiet) { Write-Host '' }

    [pscustomobject]@{
        Mode        = $label
        HTTP        = [int]$resp.StatusCode
        TTFT        = [math]::Round($firstAny, 2)       # first text (thinking or answer)
        FirstAnswer = [math]::Round($firstAnswer, 2)    # first answer text
        Total       = [math]::Round($total, 2)
        LoadS       = [math]::Round($last.load_duration / 1e9, 2)
        PromptS     = [math]::Round($last.prompt_eval_duration / 1e9, 3)
        Chunks      = $chunks
        EvalCount   = $last.eval_count
        TokS        = [math]::Round($last.eval_count / ($last.eval_duration / 1e9), 1)
        Done        = $last.done_reason
        Text        = $text
        Raw         = $raw
    }
}

# Unload first so the first run shows a cold start (keep_alive 0 = unload now; documented on /api/generate)
Invoke-RestMethod "$api/api/generate" -Method Post -ContentType 'application/json' -Body (@{ model = $Model; keep_alive = 0 } | ConvertTo-Json) | Out-Null

$runs = @(
    @{ Stream = $true;  Think = $false; Label = 'stream, cold' }
    @{ Stream = $true;  Think = $false; Label = 'stream, warm' }
    @{ Stream = $false; Think = $false; Label = 'no stream, warm' }
    @{ Stream = $true;  Think = $true;  Label = 'stream, think, warm' }
)
$i = 0
$rows = foreach ($r in $runs) {
    Write-Progress -Activity "stream: $Model" -Status "Run $($i + 1) of $($runs.Count): $($r.Label)" -PercentComplete ([int](100 * $i / $runs.Count)); $i++
    Invoke-Measured $r.Stream $r.Think $r.Label
}
Write-Progress -Activity "stream: $Model" -Completed
$http.Dispose()

"`n"
$rows | Select-Object Mode, HTTP, TTFT, FirstAnswer, Total, LoadS, PromptS, Chunks, EvalCount, TokS, Done | Format-Table -AutoSize
"Same answer text, streamed vs not (warm, thinking off): $($rows[1].Text -eq $rows[2].Text)"
"`nWire format, 'stream, warm' (first 2 lines and the final line):"
$rows[1].Raw[0..1] + $rows[1].Raw[-1] | ForEach-Object { '  ' + $_.Substring(0, [math]::Min(230, $_.Length)) }
"`nWire format, 'no stream, warm' (the whole body is one JSON object):"
'  ' + $rows[2].Raw[0].Substring(0, [math]::Min(230, $rows[2].Raw[0].Length)) + ' ...'

New-Item -ItemType Directory -Force $OutDir | Out-Null
$report = Join-Path $OutDir "stream-$(Get-Date -Format 'yyyyMMdd-HHmmss').md"
Add-Content $report "# stream ($Model)`n`n$($rows | Select-Object Mode, TTFT, FirstAnswer, Total, LoadS, Chunks, EvalCount, TokS, Done | Format-Table -AutoSize | Out-String)"
foreach ($r in $rows) { Add-Content $report "## $($r.Mode)`n`n$($r.Text)`n`nRaw:`n``````text`n$($r.Raw -join "`n")`n```````n" }
"Saved to $report"
