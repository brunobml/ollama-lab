param([int]$IntervalSeconds = 2)

while ($true) {
    # Driver view: everything on the GPU (Windows, browser, Ollama...)
    $used, $total, $util = (nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu `
        --format=csv,noheader,nounits) -split ',\s*'

    # Ollama view: what Ollama has loaded right now
    $loaded = (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models
    $models = if ($loaded) {
        ($loaded | ForEach-Object {
            $gpuPct = if ($_.size) { [int](100 * $_.size_vram / $_.size) } else { 0 }
            '{0} | {1:N2} GB total, {2}% GPU | ctx {3} | unloads {4:HH:mm:ss}' -f `
                $_.name, ($_.size / 1GB), $gpuPct, $_.context_length, ([datetime]$_.expires_at)
        }) -join '; '
    } else { 'none loaded' }

    '{0:HH:mm:ss}  VRAM {1,5}/{2} MiB (free {3,5})  GPU {4,3}%  |  {5}' -f `
        (Get-Date), $used, $total, ($total - $used), $util, $models
    Start-Sleep -Seconds $IntervalSeconds
}