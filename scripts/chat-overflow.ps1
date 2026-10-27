param(
    [string]$Model = 'qwen3.5:4b',
    [int]$Batches = 12,               # filler turns; 12 x 20 log lines ≈ 2x a 4,096 context
    [int]$NumCtx = 4096,
    [string]$OutDir = 'results/raw'   # gitignored
)

# Step 5b. A long CHAT: the conversation outgrows num_ctx one message at a time.
# Is it cut the same way as one long prompt (5a), or does /api/chat drop whole old messages?
# And does the system message survive?
#   system:  a fact (rotation code ORCA-7)
#   turn 1:  another fact (ticket INC-4821)
#   turns 2..N+1: log batches, each answered "Received batch k."
#   last:    ask for both facts and the lowest batch number still visible

$api = 'http://127.0.0.1:11434'
$log = Join-Path $env:LOCALAPPDATA 'Ollama\server.log'
$countCtx = 16384

$messages = @(
    @{ role = 'system';    content = 'The on-call rotation code is ORCA-7. Remember it.' }
    @{ role = 'user';      content = 'Note for later: the incident ticket is INC-4821.' }
    @{ role = 'assistant'; content = 'Noted: INC-4821.' }
)
for ($b = 1; $b -le $Batches; $b++) {
    $lines = (1..20 | ForEach-Object {
        'batch={0:00} 2026-10-27T13:{0:00}:{1:00}Z pod/web-{2} ns=shop level=WARN msg="slow upstream" latency_ms={3}' -f $b, $_, ($_ % 5), (500 + ($b * 31 + $_ * 17) % 900)
    }) -join "`n"
    $messages += @{ role = 'user'; content = "Log batch ${b}:`n$lines" }
    $messages += @{ role = 'assistant'; content = "Received batch $b." }
}
$messages += @{ role = 'user'; content = 'What is the on-call rotation code, what is the incident ticket, and what is the lowest log batch number you can see? Reply exactly as: code=<code> ticket=<ticket> lowest=<n>' }

function Invoke-Chat([int]$ctx, [int]$predict) {
    $body = @{ model = $Model; messages = $messages; stream = $false; think = $false
               options = @{ num_ctx = $ctx; num_predict = $predict; temperature = 0; seed = 42 } } | ConvertTo-Json -Depth 5
    Invoke-RestMethod "$api/api/chat" -Method Post -Body $body -ContentType 'application/json'
}

Write-Progress -Activity "chat-overflow: $Model" -Status "Step 1 of 2: counting tokens at num_ctx $countCtx" -PercentComplete 0
$full = Invoke-Chat $countCtx 40
Write-Progress -Activity "chat-overflow: $Model" -Status "Step 2 of 2: same chat at num_ctx $NumCtx" -PercentComplete 50
$before = (Get-Content $log).Count
$small = Invoke-Chat $NumCtx 40
$logHits = Get-Content $log | Select-Object -Skip $before | Where-Object { $_ -match 'truncat|shift|exceed|context limit' } |
    ForEach-Object { ($_ -replace '^time=\S+ ', '').Substring(0, [math]::Min(140, $_.Length)) }
Write-Progress -Activity "chat-overflow: $Model" -Completed

$rows = foreach ($x in @(@{ Ctx = $countCtx; R = $full }, @{ Ctx = $NumCtx; R = $small })) {
    [pscustomobject]@{
        NumCtx    = $x.Ctx
        Messages  = $messages.Count
        PromptTok = $x.R.prompt_eval_count
        Answer    = "$($x.R.message.content)".Trim() -replace '\s+', ' '
    }
}
$rows | Format-Table -AutoSize
"server.log lines about truncation (num_ctx $NumCtx run):"
$logHits | Sort-Object -Unique | ForEach-Object { "  $_" }

New-Item -ItemType Directory -Force $OutDir | Out-Null
$report = Join-Path $OutDir "chat-overflow-$(Get-Date -Format 'yyyyMMdd-HHmmss').md"
Add-Content $report "# chat-overflow ($Model, $($messages.Count) messages)`n`n$($rows | Format-Table -AutoSize | Out-String)`n`nLog:`n$($logHits -join "`n")`n"
"Saved to $report"
