# Manual GitHub Repository Setup Commands
# Use this script when automated creation fails or for manual verification

Write-Host "🔧 Manual GitHub Repository Setup Commands" -ForegroundColor Magenta
Write-Host "=" * 60

$Priority1Projects = @(
    @{Name = "ai-agent-framework"; Path = "C:\Users\david\ai-agent-framework"},
    @{Name = "UDC"; Path = "T:\projects\mcp_servers\UDC"},
    @{Name = "google-adk-mcp-server"; Path = "T:\projects\mcp_servers\google-adk-mcp-server"},
    @{Name = "mcp-powershell-exec"; Path = "T:\projects\mcp_servers\mcp-powershell-exec"},
    @{Name = "rust-commander"; Path = "T:\projects\rust-commander"}
)

Write-Host "`n📋 MANUAL REPOSITORY CREATION INSTRUCTIONS" -ForegroundColor Yellow
Write-Host "-" * 50

Write-Host "`n1. Use the guarded local workflow for each repository" -ForegroundColor Cyan
Write-Host "   This now requires a signed local approval before creation" -ForegroundColor Red

foreach ($project in $Priority1Projects) {
    Write-Host "`n🔹 Repository: $($project.Name)" -ForegroundColor Green
    Write-Host "   URL: https://github.com/David-Martel/$($project.Name)" -ForegroundColor Gray
    Write-Host "   Visibility: Private" -ForegroundColor Red
}

Write-Host "`n`n2. Git initialization commands for each project:" -ForegroundColor Cyan
Write-Host "-" * 40

foreach ($project in $Priority1Projects) {
    Write-Host "`n# $($project.Name)" -ForegroundColor Green
    Write-Host "cd `"$($project.Path)`""
    Write-Host "git remote -v  # Check existing remotes"
    Write-Host "pwsh -NoProfile -File $HOME\\bin\\Approve-GitHubRepoCreation.ps1 -Owner David-Martel -Name $($project.Name) -Visibility private -Description `"<describe project>`" -Reason `"initial protected bootstrap`""
    Write-Host "pwsh -NoProfile -File $HOME\\bin\\New-GitHubRepoGuarded.ps1 -Owner David-Martel -Name $($project.Name) -Visibility private -Description `"<describe project>`""
    Write-Host "git remote add origin https://github.com/David-Martel/$($project.Name).git"
    Write-Host "git add ."
    Write-Host "git commit -m `"Initial commit: $($project.Name) project setup`""
    Write-Host "git push -u origin main"
    Write-Host ""
}

Write-Host "`n📊 VERIFICATION COMMANDS" -ForegroundColor Yellow
Write-Host "-" * 30

Write-Host @"

# Check all repositories
gh repo list David-Martel --limit 20

# Verify git status for all projects
"@

foreach ($project in $Priority1Projects) {
    Write-Host "cd `"$($project.Path)`" && git status && git remote -v"
}

Write-Host "`n`n🔐 SECURITY CONFIGURATION (After creation)" -ForegroundColor Red
Write-Host "-" * 40

Write-Host @"

For each repository, configure:
1. Branch protection on main branch
2. Enable security scanning
3. Enable dependency vulnerability alerts
4. Set up code scanning
5. Configure automated security updates

Use GitHub web interface or:
gh repo edit REPO_NAME --enable-issues --enable-wiki=false --visibility private

"@

Write-Host "`n🎯 PRIORITY EXECUTION ORDER" -ForegroundColor Magenta
Write-Host "-" * 30

$priority = 1
foreach ($project in $Priority1Projects) {
    Write-Host "$priority. $($project.Name) - $($project.Path)" -ForegroundColor Cyan
    $priority++
}

Write-Host "`n✅ After completion, verify all repositories exist at:" -ForegroundColor Green
Write-Host "   https://github.com/David-Martel?tab=repositories" -ForegroundColor Gray
