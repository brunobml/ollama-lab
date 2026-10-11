param(
    [Parameter(Mandatory)][string]$Model,
    [int]$Runs = 2,
    [int]$NumPredict = 256,
    [int[]]$NumCtx = @(4096),
    [int]$LogLines = 0,
    [string]$Prompt = 'Explain step by step how the Kubernetes scheduler places a pod on a node.'
)

# Synthetic pod logs: about 25 tokens per line, so 120 lines is about 3,000 tokens
$logs = if ($LogLines) {
    (1..$LogLines | ForEach-Object {
        "2026-10-10T12:{0:00}:{1:00}Z pod/api-{2} ns=prod level=INFO msg=`"request served`" latency_ms={3}" -f ($_ % 60), ($_ % 60), ($_ % 7), (Get-Random -Maximum 900)
    }) -join "`n"
} else { '' }

$results = foreach ($ctx in $NumCtx) {
    1..$Runs | ForEach-Object {
        $body = @{
            model   = $Model
            prompt  = "[run $_ $(Get-Random)]`n$logs`n$Prompt"
            stream  = $false
            think   = $false
            options = @{ num_predict = $NumPredict; num_ctx = $ctx; temperature = 0; seed = 42 }
        } | ConvertTo-Json -Depth 4

        $r  = Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType 'application/json'
        $ps = (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models | Where-Object name -like "$Model*"

        [pscustomobject]@{
            Model      = $Model
            Ctx        = $ctx
            Run        = $_
            LoadS      = [math]::Round($r.load_duration / 1e9, 2)
            PromptTok  = $r.prompt_eval_count
            PromptTokS = [math]::Round($r.prompt_eval_count / ($r.prompt_eval_duration / 1e9))
            GenTok     = $r.eval_count
            GenTokS    = [math]::Round($r.eval_count / ($r.eval_duration / 1e9), 1)
            VramMiB    = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
            GpuPct     = if ($ps.size) { [int](100 * $ps.size_vram / $ps.size) } else { $null }
        }
    }
}
$results | Format-Table -AutoSize
