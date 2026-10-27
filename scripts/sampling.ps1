param(
    [string]$Model = 'qwen3.5:4b',
    [int]$Runs = 10,                  # samples per prompt and condition
    [string]$OutDir = 'results/raw'   # gitignored
)

# Step 4b. Change one sampling setting at a time and measure two things:
#   Distinct: how many different answers out of $Runs (variety)
#   Valid:    how many pass a correctness check (quality)
# Baseline "raw" = temperature 1 with every filter off, i.e. sampling straight from the model's
# probabilities. Each other condition changes ONE setting from that baseline.
# Thinking off and no seed, so only the sampler creates the variation.

$api = 'http://127.0.0.1:11434'
$prompts = @(
    @{ Id = 'name'   # creative: variety is welcome, but the result must still be a valid namespace name
       Q  = 'Suggest a name for a Kubernetes namespace for the payments team''s staging environment. Reply with the name only.'
       Valid = '^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$' }   # RFC 1123 label: lowercase, digits, '-', max 63
    @{ Id = 'port'   # factual: one right answer, variety means errors
       Q  = 'What is the default secure port of the Kubernetes API server? Reply with the number only.'
       Valid = '^6443$' }
)
# Filters off: top_k 0, top_p 1, min_p 0 (llama.cpp treats these as disabled); no penalties
$raw = @{ temperature = 1; top_k = 0; top_p = 1; min_p = 0; presence_penalty = 0 }
$conditions = @(
    @{ Name = 'greedy (temp 0)';  Set = @{ temperature = 0 } }
    @{ Name = 'temp 0.3';         Set = @{ temperature = 0.3 } }
    @{ Name = 'raw (temp 1)';     Set = @{} }
    @{ Name = 'temp 1.5';         Set = @{ temperature = 1.5 } }
    @{ Name = 'temp 1.5, top_k 5';Set = @{ temperature = 1.5; top_k = 5 } }
    @{ Name = 'temp 1.5, top_p 0.5'; Set = @{ temperature = 1.5; top_p = 0.5 } }
    @{ Name = 'temp 1.5, min_p 0.1'; Set = @{ temperature = 1.5; min_p = 0.1 } }
    @{ Name = 'model defaults';   Set = $null }   # no options: Modelfile values (temp 1, top_k 20, top_p 0.95, presence 1.5)
)

New-Item -ItemType Directory -Force $OutDir | Out-Null
$report = Join-Path $OutDir "sampling-$(Get-Date -Format 'yyyyMMdd-HHmmss').md"

$rows = foreach ($c in $conditions) {
    $opts = if ($null -eq $c.Set) { @{} } else { $o = $raw.Clone(); foreach ($k in $c.Set.Keys) { $o[$k] = $c.Set[$k] }; $o }
    foreach ($p in $prompts) {
        $answers = for ($i = 1; $i -le $Runs; $i++) {
            $body = @{ model = $Model; prompt = $p.Q; stream = $false; think = $false; options = $opts + @{ num_predict = 40 } } | ConvertTo-Json -Depth 4
            (Invoke-RestMethod "$api/api/generate" -Method Post -Body $body -ContentType 'application/json').response.Trim()
        }
        $list = ($answers | ForEach-Object { '* `' + $_ + '`' }) -join "`n"
        Add-Content $report "## $($c.Name) / $($p.Id)`n`n$list`n"
        $distinct = @($answers | Sort-Object -Unique -CaseSensitive)   # 'Payments-Staging' ≠ 'payments-staging'
        [pscustomobject]@{
            Condition = $c.Name
            Prompt    = $p.Id
            Distinct  = "$($distinct.Count)/$Runs"
            Valid     = "$(@($answers | Where-Object { $_ -cmatch $p.Valid }).Count)/$Runs"
            Examples  = (($distinct | Select-Object -First 3) -join ' | ') -replace '\s+', ' '
        }
    }
}
$rows | Format-Table -AutoSize
"Answers saved to $report"
