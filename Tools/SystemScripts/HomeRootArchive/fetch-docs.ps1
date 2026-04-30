$docs = @(
  @{ Url = "https://docs.ollama.com/faq"; Out = "C:\Users\david\.ollama\docs\faq.md"; Title = "Ollama FAQ" },
  @{ Url = "https://docs.ollama.com/gpu"; Out = "C:\Users\david\.ollama\docs\gpu.md"; Title = "Ollama GPU" },
  @{ Url = "https://docs.ollama.com/windows"; Out = "C:\Users\david\.ollama\docs\windows.md"; Title = "Ollama Windows" },
  @{ Url = "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"; Out = "C:\Users\david\.nvidia\docs\container-toolkit-install.md"; Title = "NVIDIA Container Toolkit Install Guide" },
  @{ Url = "https://docs.docker.com/desktop/features/gpu/"; Out = "C:\Users\david\.docker\docs\desktop-gpu.md"; Title = "Docker Desktop GPU Support" },
  @{ Url = "https://docs.nvidia.com/cuda/wsl-user-guide/index.html"; Out = "C:\Users\david\.nvidia\docs\wsl-user-guide.md"; Title = "NVIDIA CUDA on WSL User Guide" }
)

foreach ($doc in $docs) {
  $outDir = Split-Path -Path $doc.Out -Parent
  if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
  }

  $tmpHtml = Join-Path $env:TEMP ("doc-" + [guid]::NewGuid().ToString() + ".html")
  $tmpMd = Join-Path $env:TEMP ("doc-" + [guid]::NewGuid().ToString() + ".md")

  try {
    Invoke-WebRequest -Uri $doc.Url -OutFile $tmpHtml -UseBasicParsing -MaximumRedirection 5
  } catch {
    Write-Warning "Failed to download $($doc.Url): $_"
    Remove-Item $tmpHtml -Force -ErrorAction SilentlyContinue
    continue
  }

  if (-not (Test-Path $tmpHtml)) {
    Write-Warning "Missing HTML file for $($doc.Url)"
    continue
  }

  & pandoc -f html -t gfm -o $tmpMd $tmpHtml
  if (-not (Test-Path $tmpMd)) {
    Write-Warning "Pandoc failed for $($doc.Url)"
    Remove-Item $tmpHtml -Force -ErrorAction SilentlyContinue
    continue
  }

  $header = @(
    "---",
    "title: $($doc.Title)",
    "source: $($doc.Url)",
    "fetched: $(Get-Date -Format 'yyyy-MM-dd')",
    "---",
    ""
  ) -join "`n"

  $content = Get-Content -Path $tmpMd -Raw
  Set-Content -Path $doc.Out -Value ($header + $content)

  Remove-Item $tmpHtml, $tmpMd -Force -ErrorAction SilentlyContinue
}

Write-Host "Downloaded and converted docs to markdown."
