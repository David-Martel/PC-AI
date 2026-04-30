BeforeAll {
  $scriptPath = 'C:\scripts\startup\ollama-service.ps1'
  if (-not (Test-Path $scriptPath)) {
    throw "Missing script: $scriptPath"
  }
}

Describe 'Ollama Service Startup Script' {
  It 'runs in DryRun mode without throwing' {
    $scriptPath = 'C:\scripts\startup\ollama-service.ps1'
    { & $scriptPath -Ensure -DryRun } | Should -Not -Throw
  }

  It 'accepts CUDA device overrides in DryRun mode' {
    $scriptPath = 'C:\scripts\startup\ollama-service.ps1'
    { & $scriptPath -Ensure -DryRun -CudaDevices '1' -PreferCuda } | Should -Not -Throw
  }

  It 'handles missing preferred CUDA GPU names in DryRun mode' {
    $scriptPath = 'C:\scripts\startup\ollama-service.ps1'
    { & $scriptPath -Ensure -DryRun -PreferCudaGpuName 'Nonexistent GPU' -PreferCuda } | Should -Not -Throw
  }
}
