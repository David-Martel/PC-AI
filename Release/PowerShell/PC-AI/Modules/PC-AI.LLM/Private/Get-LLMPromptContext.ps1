#Requires -Version 5.1

<#
.SYNOPSIS
    Generates structured metadata and grounding context for LLM prompts.
#>
function Get-LLMPromptContext {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$AnalysisType = 'Quick'
    )

    $projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

    # 1. Collect Metadata
    $metadata = @{
        os_version    = (Get-CimInstance Win32_OperatingSystem).Caption
        ps_version    = $PSVersionTable.PSVersion.ToString()
        pcai_version  = '2.0.0'
        native_engine = if (Get-Command Initialize-PcaiNative -ErrorAction SilentlyContinue) { 'Available' } else { 'Unavailable' }
        current_time  = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
        analysis_type = $AnalysisType
    }

    # 2. Read Grounding Files
    $diagnosePromptPath = Join-Path -Path $projectRoot -ChildPath 'DIAGNOSE.md'
    $diagnoseLogicPath = Join-Path -Path $projectRoot -ChildPath 'DIAGNOSE_LOGIC.md'
    $diagnoseTemplatePath = Join-Path -Path $projectRoot -ChildPath 'Config\DIAGNOSE_TEMPLATE.json'

    $prompts = @{
        System   = if (Test-Path $diagnosePromptPath) { Get-Content $diagnosePromptPath -Raw -Encoding utf8 } else { '' }
        Logic    = if (Test-Path $diagnoseLogicPath) { Get-Content $diagnoseLogicPath -Raw -Encoding utf8 } else { '' }
        Template = if (Test-Path $diagnoseTemplatePath) { Get-Content $diagnoseTemplatePath -Raw } else { '{}' }
    }

    return [PSCustomObject]@{
        Metadata     = $metadata
        MetadataJson = $metadata | ConvertTo-Json -Compress
        Prompts      = $prompts
    }
}
