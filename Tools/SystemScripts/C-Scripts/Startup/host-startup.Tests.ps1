BeforeAll {
  $scriptPath = 'C:\scripts\startup\host-startup.ps1'
  if (-not (Test-Path $scriptPath)) {
    throw "Missing script: $scriptPath"
  }
}

Describe 'Host Startup Script' {
  It 'runs in DryRun mode without throwing' {
    $scriptPath = 'C:\scripts\startup\host-startup.ps1'
    { & $scriptPath -EnsureOllama -StartDocker -StartWSL -DryRun } | Should -Not -Throw
  }
}
