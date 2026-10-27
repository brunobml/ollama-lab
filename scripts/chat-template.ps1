param(
    [string]$Model = 'qwen3.5:4b',
    [string]$OutDir = 'results/raw'   # gitignored
)

# Step 3a. Part 1: /api/chat turns a messages array into ONE prompt string (the chat template).
# qwen3.5 has no visible template (`TEMPLATE {{ .Prompt }}` + `RENDERER qwen3.5`), so we rebuild
# the prompt by hand in ChatML and send it with /api/generate raw=true (no template applied).
# Same prompt tokens + same answer = we reproduced exactly what the model sees.
# Part 2: the API is stateless. "Memory" is the client re-sending the history every turn.

$api  = 'http://127.0.0.1:11434'
$opts = @{ temperature = 0; seed = 42; num_predict = 120 }
$im = '<|im_start|>'; $ie = '<|im_end|>'

function Invoke-Chat($messages, [bool]$think = $false) {
    $body = @{ model = $Model; messages = $messages; stream = $false; think = $think; options = $opts } | ConvertTo-Json -Depth 5
    Invoke-RestMethod "$api/api/chat" -Method Post -Body $body -ContentType 'application/json'
}
function Invoke-Raw([string]$prompt) {
    $body = @{ model = $Model; prompt = $prompt; raw = $true; stream = $false; options = $opts } | ConvertTo-Json -Depth 5
    Invoke-RestMethod "$api/api/generate" -Method Post -Body $body -ContentType 'application/json'
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
$report = Join-Path $OutDir "chat-template-$(Get-Date -Format 'yyyyMMdd-HHmmss').md"
$sys  = 'Answer in one word.'
$user = 'Name one Kubernetes controller.'
$noThink = "<think>`n`n</think>`n`n"   # what think=false prefills (verified 2026-10-27: same tokens and output)

# --- Part 1: chat vs hand-built ChatML ---
$cases = @(
    @{ Name = 'user only';      Msgs = @(@{ role = 'user'; content = $user })
       Raw  = "${im}user`n$user$ie`n${im}assistant`n$noThink" }
    @{ Name = 'system + user';  Msgs = @(@{ role = 'system'; content = $sys }, @{ role = 'user'; content = $user })
       Raw  = "${im}system`n$sys$ie`n${im}user`n$user$ie`n${im}assistant`n$noThink" }
    @{ Name = 'think on';       Msgs = @(@{ role = 'user'; content = $user }); Think = $true   # hypothesis: prefill is only "<think>\n"
       Raw  = "${im}user`n$user$ie`n${im}assistant`n<think>`n" }
)
Add-Content $report "# Part 1: /api/chat vs raw ChatML ($Model)`n"
$turns = @(   # Part 2 conversation
    'Our production cluster is called blue-falcon and runs 12 nodes. Just acknowledge.'
    'How many nodes does it run?'
    'What is the cluster called?'
)
$steps = $cases.Count + $turns.Count
$n = 0
$part1 = foreach ($c in $cases) {
    Write-Progress -Activity "chat-template: $Model" -Status "Step $($n + 1) of ${steps}: part 1, $($c.Name)" -PercentComplete ([int](100 * $n / $steps))
    $n++
    $chat = Invoke-Chat $c.Msgs ([bool]$c.Think)
    $raw  = Invoke-Raw $c.Raw
    Add-Content $report "## $($c.Name)`n`n### Raw prompt`n`n``````text`n$($c.Raw)`n```````n`n### chat thinking`n`n$($chat.message.thinking)`n`n### chat answer`n`n$($chat.message.content)`n`n### raw response`n`n$($raw.response)`n"
    [pscustomobject]@{
        Case       = $c.Name
        ChatTok    = $chat.prompt_eval_count
        RawTok     = $raw.prompt_eval_count
        SameAnswer = $chat.message.content.Trim() -eq $raw.response.Trim()   # think on: False, raw keeps the thinking text inline
        # think on: does chat's separate `thinking` field hold the same text raw returns inline?
        SameThink  = if ($c.Think) { $raw.response.Trim().StartsWith("$($chat.message.thinking)".Trim()) } else { '' }
        ChatAnswer = ($chat.message.content -replace '\s+', ' ').Substring(0, [math]::Min(50, $chat.message.content.Length))
    }
}
$part1 | Format-Table -AutoSize

# --- Part 2: statelessness ---
# The same conversation, with and without the client re-sending history ($turns, defined above).
Add-Content $report "# Part 2: statelessness`n"
$history = @()
$part2 = for ($i = 0; $i -lt $turns.Count; $i++) {
    Write-Progress -Activity "chat-template: $Model" -Status "Step $($n + 1) of ${steps}: part 2, turn $($i + 1)" -PercentComplete ([int](100 * $n / $steps))
    $n++
    $history += @{ role = 'user'; content = $turns[$i] }
    $with    = Invoke-Chat $history
    $without = if ($i -gt 0) { Invoke-Chat @(@{ role = 'user'; content = $turns[$i] }) }
    $history += @{ role = 'assistant'; content = $with.message.content }
    Add-Content $report "## Turn $($i + 1): $($turns[$i])`n`n**With history:** $($with.message.content)`n`n**Without history:** $($without.message.content)`n"
    [pscustomobject]@{
        Turn          = $i + 1
        Messages      = $history.Count - 1               # messages sent this turn
        PromptTok     = $with.prompt_eval_count          # tokens in the prompt this turn
        Cached        = $with.prompt_eval_cached_count   # of those, reused from the prompt cache
        WithHistory   = ($with.message.content -replace '\s+', ' ').Substring(0, [math]::Min(45, $with.message.content.Length))
        NoHistory     = if ($without) { ($without.message.content -replace '\s+', ' ').Substring(0, [math]::Min(45, $without.message.content.Length)) } else { '' }
    }
}
Write-Progress -Activity "chat-template: $Model" -Completed
$part2 | Format-Table -AutoSize
"Prompts and answers saved to $report"
