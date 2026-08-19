# GitHub Repository Creation Script for Priority Projects
# This script creates private GitHub repositories for the critical Priority 1 projects
# identified in the DevOps audit

# Set error action preference
$ErrorActionPreference = "Stop"

# Define Priority 1 projects with their locations and descriptions
$Priority1Projects = @(
    @{
        Name = "ai-agent-framework"
        Path = "C:\Users\david\ai-agent-framework"
        Description = "Comprehensive AI agent ecosystem monorepo with Gemini, Claude, and automation frameworks. Production-ready multi-agent system with MCP integration, authentication, and orchestration tools."
        HasGit = $true
    },
    @{
        Name = "UDC"
        Path = "T:\projects\mcp_servers\UDC"
        Description = "Universal Desktop Commander - Advanced cross-platform command execution and automation MCP server with comprehensive Windows, WSL, and Unix support."
        HasGit = $true
    },
    @{
        Name = "google-adk-mcp-server"
        Path = "T:\projects\mcp_servers\google-adk-mcp-server"
        Description = "Google Application Development Kit MCP Server - Business integration with Google Workspace, Gemini AI, and comprehensive authentication framework."
        HasGit = $true
    },
    @{
        Name = "mcp-powershell-exec"
        Path = "T:\projects\mcp_servers\mcp-powershell-exec"
        Description = "PowerShell Execution MCP Server - Secure Windows automation server with FastMCP integration, command history, and comprehensive logging."
        HasGit = $true
    },
    @{
        Name = "rust-commander"
        Path = "T:\projects\rust-commander"
        Description = "High-performance Rust-based command execution framework with MCP integration, comprehensive testing, and cross-platform support."
        HasGit = $true
    }
)

# High Priority projects for Week 2
$HighPriorityProjects = @(
    @{
        Name = "google-cloud-comprehensive-mcp"
        Path = "T:\projects\mcp_servers\google-cloud-comprehensive-mcp"
        Description = "Comprehensive Google Cloud Platform MCP server with Vertex AI, Cloud Run, and GCP service integration."
    },
    @{
        Name = "dbx-mcp-server"
        Path = "T:\projects\mcp_servers\dbx-mcp-server"
        Description = "Database integration MCP server with multi-database support and query optimization."
    },
    @{
        Name = "adobe-controller-mcp"
        Path = "T:\projects\mcp_servers\adobe-controller"
        Description = "Adobe Creative Suite automation and control MCP server."
    },
    @{
        Name = "memory-mcp-server"
        Path = "T:\projects\mcp_servers\memory"
        Description = "Persistent memory and knowledge graph MCP server with advanced retrieval capabilities."
    }
)

function Test-GitHubCLI {
    try {
        $null = Get-Command gh -ErrorAction Stop
        $authStatus = gh auth status 2>&1
        if ($authStatus -match "Logged in") {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

function Test-GitInstallation {
    try {
        $null = Get-Command git -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Create-GitHubRepository {
    param(
        [string]$Name,
        [string]$Description,
        [string]$Path
    )

    Write-Host "Creating repository: $Name" -ForegroundColor Green

    if (Test-GitHubCLI) {
        try {
            # Require an explicit signed approval before creating a repo
            & "$HOME\bin\Approve-GitHubRepoCreation.ps1" -Owner 'David-Martel' -Name $Name -Visibility private -Description $Description -Reason 'Priority project inventory bootstrap'
            & "$HOME\bin\New-GitHubRepoGuarded.ps1" -Owner 'David-Martel' -Name $Name -Visibility private -Description $Description
            Write-Host "✅ Successfully created repository: $Name" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Warning "❌ Failed to create repository $Name using GitHub CLI: $_"
            return $false
        }
    }
    else {
        Write-Warning "⚠️  GitHub CLI not available or not authenticated for $Name"
        Write-Host "Manual creation required at: https://github.com/new" -ForegroundColor Yellow
        Write-Host "Repository Name: $Name" -ForegroundColor Cyan
        Write-Host "Description: $Description" -ForegroundColor Cyan
        Write-Host "Visibility: Private" -ForegroundColor Cyan
        Write-Host ""
        return $false
    }
}

function Initialize-GitRepository {
    param(
        [string]$Name,
        [string]$Path,
        [bool]$HasGit
    )

    Write-Host "Initializing git for: $Name at $Path" -ForegroundColor Blue

    if (-not (Test-Path $Path)) {
        Write-Warning "❌ Path does not exist: $Path"
        return $false
    }

    try {
        Push-Location $Path

        if (-not $HasGit) {
            git init
            Write-Host "✅ Initialized git repository" -ForegroundColor Green
        }

        # Check if remote origin exists
        $remotes = git remote 2>$null
        if ($remotes -notcontains "origin") {
            git remote add origin "https://github.com/David-Martel/$Name.git"
            Write-Host "✅ Added remote origin" -ForegroundColor Green
        }
        else {
            Write-Host "ℹ️  Remote origin already exists" -ForegroundColor Yellow
        }

        # Check if there are uncommitted changes
        $status = git status --porcelain 2>$null
        if ($status) {
            git add .
            git commit -m "Initial commit: $Name project setup

This commit represents the initial state of the $Name project,
including all current functionality and configuration.

Priority: Critical (Week 1)
Type: Repository initialization
Scope: Complete project codebase"
            Write-Host "✅ Committed changes" -ForegroundColor Green
        }

        # Push to GitHub (if remote exists and is accessible)
        try {
            git push -u origin main 2>$null
            Write-Host "✅ Pushed to GitHub successfully" -ForegroundColor Green
        }
        catch {
            try {
                git push -u origin master 2>$null
                Write-Host "✅ Pushed to GitHub successfully (master branch)" -ForegroundColor Green
            }
            catch {
                Write-Warning "⚠️  Could not push to GitHub. Manual push required after repository creation."
                Write-Host "Manual push command: git push -u origin main" -ForegroundColor Cyan
            }
        }

        return $true
    }
    catch {
        Write-Warning "❌ Failed to initialize git for ${Name}: $_"
        return $false
    }
    finally {
        Pop-Location
    }
}

function Create-ProjectInventoryReport {
    $reportPath = "C:\Users\david\GitHub-Repository-Creation-Report.md"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    $report = @"
# GitHub Repository Creation Report

**Generated:** $timestamp
**Status:** Repository creation and git initialization for Priority 1 projects

## Executive Summary

This report documents the creation of GitHub repositories for the Priority 1 projects identified in the DevOps audit. These projects represent the most critical intellectual property that was not under version control.

## Priority 1 Projects (Critical - Week 1)

"@

    foreach ($project in $Priority1Projects) {
        $report += @"

### $($project.Name)
- **Location:** $($project.Path)
- **Description:** $($project.Description)
- **Git Status:** $($project.HasGit -eq $true ? "Existing repository" : "New repository")
- **GitHub URL:** https://github.com/David-Martel/$($project.Name)

"@
    }

    $report += @"

## High Priority Projects (Week 2)

"@

    foreach ($project in $HighPriorityProjects) {
        $report += @"

### $($project.Name)
- **Location:** $($project.Path)
- **Description:** $($project.Description)
- **GitHub URL:** https://github.com/David-Martel/$($project.Name)

"@
    }

    $report += @"

## Git Initialization Commands

For manual execution if automated process fails:

``````powershell
# For each project directory:
cd "PROJECT_PATH"
git init
git add .
git commit -m "Initial commit: PROJECT_NAME"
git remote add origin https://github.com/David-Martel/PROJECT_NAME.git
git push -u origin main
``````

## Security Configuration

All repositories should be configured with:
- Private visibility
- Branch protection on main branch
- Security scanning enabled
- Dependency vulnerability alerts enabled
- Code scanning enabled

## Next Steps

1. Complete repository creation for all Priority 1 projects
2. Configure branch protection and security settings
3. Begin High Priority projects (Week 2)
4. Set up automated backups and CI/CD pipelines
5. Conduct security audit of all repositories

## Verification Commands

``````powershell
# Check repository status
gh repo list David-Martel --limit 50

# Verify local git configuration
foreach (\$project in \$Priority1Projects) {
    cd \$project.Path
    git remote -v
    git status
}
``````

---
*Report generated by GitHub Repository Creation Script*
*Addressing critical 95% sync gap identified in DevOps audit*
"@

    $report | Out-File -FilePath $reportPath -Encoding UTF8
    Write-Host "📋 Report generated: $reportPath" -ForegroundColor Green
    return $reportPath
}

# Main execution
Write-Host "🚀 Starting Priority 1 GitHub Repository Creation" -ForegroundColor Magenta
Write-Host "=" * 60

# Check prerequisites
$hasGitHub = Test-GitHubCLI
$hasGit = Test-GitInstallation

Write-Host "Prerequisites Check:" -ForegroundColor Yellow
Write-Host "  GitHub CLI: $($hasGitHub ? '✅' : '❌')" -ForegroundColor ($hasGitHub ? 'Green' : 'Red')
Write-Host "  Git: $($hasGit ? '✅' : '❌')" -ForegroundColor ($hasGit ? 'Green' : 'Red')
Write-Host ""

if (-not $hasGit) {
    Write-Error "Git is required but not found. Please install Git and try again."
    exit 1
}

$createdRepos = @()
$failedRepos = @()

# Process Priority 1 projects
Write-Host "Processing Priority 1 Projects (Critical - Week 1):" -ForegroundColor Magenta
Write-Host "-" * 50

foreach ($project in $Priority1Projects) {
    Write-Host "`n🎯 Processing: $($project.Name)" -ForegroundColor Cyan

    # Create GitHub repository
    $repoCreated = Create-GitHubRepository -Name $project.Name -Description $project.Description -Path $project.Path

    # Initialize and push git repository
    if (Test-Path $project.Path) {
        $gitInitialized = Initialize-GitRepository -Name $project.Name -Path $project.Path -HasGit $project.HasGit

        if ($repoCreated -and $gitInitialized) {
            $createdRepos += $project.Name
            Write-Host "✅ COMPLETED: $($project.Name)" -ForegroundColor Green
        }
        elseif ($gitInitialized) {
            Write-Host "⚠️  PARTIAL: $($project.Name) - Git initialized, manual GitHub creation needed" -ForegroundColor Yellow
        }
        else {
            $failedRepos += $project.Name
            Write-Host "❌ FAILED: $($project.Name)" -ForegroundColor Red
        }
    }
    else {
        $failedRepos += $project.Name
        Write-Host "❌ FAILED: $($project.Name) - Path not found" -ForegroundColor Red
    }
}

# Generate report
Write-Host "`n📋 Generating comprehensive report..." -ForegroundColor Blue
$reportPath = Create-ProjectInventoryReport

# Summary
Write-Host "`n" + "=" * 60
Write-Host "🎯 EXECUTION SUMMARY" -ForegroundColor Magenta
Write-Host "=" * 60

Write-Host "✅ Successfully processed: $($createdRepos.Count) repositories" -ForegroundColor Green
if ($createdRepos.Count -gt 0) {
    $createdRepos | ForEach-Object { Write-Host "   - $_" -ForegroundColor Green }
}

if ($failedRepos.Count -gt 0) {
    Write-Host "`n❌ Failed/Partial: $($failedRepos.Count) repositories" -ForegroundColor Red
    $failedRepos | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
}

Write-Host "`n📋 Detailed report: $reportPath" -ForegroundColor Cyan

if (-not $hasGitHub) {
    Write-Host "`n⚠️  AUTHENTICATION REQUIRED:" -ForegroundColor Yellow
    Write-Host "   Run 'gh auth login' to authenticate with GitHub CLI" -ForegroundColor Yellow
    Write-Host "   Then re-run this script for automated repository creation" -ForegroundColor Yellow
}

Write-Host "`n🏆 Priority 1 repository creation process completed!" -ForegroundColor Green
Write-Host "📅 Next: Begin High Priority projects (Week 2)" -ForegroundColor Blue
