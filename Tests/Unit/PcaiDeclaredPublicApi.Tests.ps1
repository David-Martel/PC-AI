#Requires -Version 7.0
param([string]$ModulesRoot)

BeforeAll {
    $script:DeclaredApiModules = if($ModulesRoot){[IO.Path]::GetFullPath($ModulesRoot)}else{[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../Modules'))}
    $script:DeclaredApiPwsh = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    $script:DeclaredApiChild = Join-Path $TestDrive 'declared-api-consumer.ps1'
    [IO.File]::WriteAllText($DeclaredApiChild, @'
param([string]$Manifest,[string]$Report)
$ErrorActionPreference='Stop'
$metadata=Import-PowerShellDataFile -LiteralPath $Manifest
$name=[IO.Path]::GetFileNameWithoutExtension($Manifest)
$expectedScript=[IO.Path]::GetFullPath((Join-Path (Split-Path $Manifest) $metadata.RootModule))
$failure=$null;$rows=@();$missing=@();$count=0;$selectedPath=$null;$actualNames=@()
try {
    Import-Module $Manifest -Force -ErrorAction Stop
    $loaded=@(Microsoft.PowerShell.Core\Get-Module -Name $name -All)
    $count=$loaded.Count
    if($count-ne1){throw 'The selected component is not unique'}
    $selected=$loaded[0];$selectedPath=$selected.Path
    if([IO.Path]::GetFullPath($selected.Path)-ne$expectedScript){throw 'Selected component path differs from manifest RootModule'}
    $actualNames=@($selected.ExportedFunctions.Keys)
    $rows=@(foreach($declared in @($metadata.FunctionsToExport)) {
        $function=$selected.ExportedFunctions[$declared]
        if($null-eq$function){$missing+=,$declared}
        [ordered]@{
            Name=$declared;Present=$null-ne$function
            Type=if($null-ne$function){$function.GetType().FullName}else{$null}
            Module=if($null-ne$function){$function.ModuleName}else{$null}
            SourceFile=if($null-ne$function){$function.ScriptBlock.File}else{$null}
        }
    })
} catch {$failure=$_.Exception.ToString()}
[ordered]@{Failure=$failure;Component=$name;Manifest=$Manifest;ExpectedScript=$expectedScript;SelectedPath=$selectedPath;ModuleCount=$count;Declared=@($metadata.FunctionsToExport);Exported=$actualNames;Missing=$missing;Rows=$rows}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $Report -Encoding utf8NoBOM
if($failure){exit 1}
'@)

    function Invoke-DeclaredApiConsumer {
        param([string]$Component)
        $root=Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void][IO.Directory]::CreateDirectory($root)
        $report=Join-Path $root 'exports.json'
        $manifest=Join-Path $script:DeclaredApiModules "$Component/$Component.psd1"
        $start=[Diagnostics.ProcessStartInfo]::new($script:DeclaredApiPwsh)
        $start.UseShellExecute=$false;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
        $start.Environment['PSModulePath']=Join-Path $PSHOME 'Modules'
        $start.Environment['PCAI_DISABLE_NATIVE']='1'
        $start.Environment['PCAI_CACHE_PROVIDER']='memory'
        foreach($arg in @('-NoLogo','-NoProfile','-File',$script:DeclaredApiChild,'-Manifest',$manifest,'-Report',$report)){$start.ArgumentList.Add($arg)}
        $child=[Diagnostics.Process]::Start($start)
        try {
            $stdout=$child.StandardOutput.ReadToEndAsync();$stderr=$child.StandardError.ReadToEndAsync()
            if(-not$child.WaitForExit(20000)){$child.Kill($true);$child.WaitForExit();throw 'Declared API metadata consumer exceeded twenty seconds.'}
            [pscustomobject]@{ExitCode=$child.ExitCode;StdOut=$stdout.GetAwaiter().GetResult();StdErr=$stderr.GetAwaiter().GetResult();Data=if([IO.File]::Exists($report)){[IO.File]::ReadAllText($report)|ConvertFrom-Json}else{$null};ComponentRoot=Split-Path $manifest}
        } finally {if(-not$child.HasExited){$child.Kill($true);$child.WaitForExit()};$child.Dispose()}
    }
}

Describe 'Declared component public API is actually exported' -Tag 'Unit','Portable','Modules' {
    # Protects: every manifest-declared function is exported by its real component.
    # Detects: stale Export-ModuleMember lists, missing definitions and foreign copies.
    # Needs: actual manifests/script modules and fresh profile-free metadata consumers.
    # Breadcrumb: Performance/Virtualization FunctionsToExport and Export-ModuleMember.
    It 'exports every declared <Component> function from the selected source' -ForEach @(
        @{Component='PC-AI.Performance'},@{Component='PC-AI.Virtualization'}
    ) {
        $r=Invoke-DeclaredApiConsumer $Component
        $r.ExitCode|Should -Be 0 -Because $r.StdErr
        $r.Data.Failure|Should -BeNullOrEmpty
        $r.Data.ModuleCount|Should -Be 1
        @($r.Data.Missing)|Should -HaveCount 0
        @($r.Data.Exported)|Should -HaveCount @($r.Data.Declared).Count
        foreach($row in $r.Data.Rows){
            $row.Present|Should -BeTrue
            $row.Type|Should -BeExactly 'System.Management.Automation.FunctionInfo'
            $row.Module|Should -BeExactly $Component
            $row.SourceFile|Should -Not -BeNullOrEmpty
            $comparison=if($IsWindows){[StringComparison]::OrdinalIgnoreCase}else{[StringComparison]::Ordinal}
            [IO.Path]::GetFullPath($row.SourceFile).StartsWith($r.ComponentRoot+[IO.Path]::DirectorySeparatorChar,$comparison)|Should -BeTrue
        }
    }
    # Protects: four existing declared entrypoints cannot be removed to hide drift.
    # Detects: export omissions and singular/plural mismatches behind declarations.
    # Needs: real component imports; reads FunctionInfo only, never calls metrics/OS actions.
    # Breadcrumb: the three Performance Public files and plural Defender public file.
    It 'retains canonical <Name> as a source-bound public entrypoint' -ForEach @(
        @{Component='PC-AI.Performance';Name='Get-PcaiMemoryPressure'}
        @{Component='PC-AI.Performance';Name='Get-PcaiProcessCategories'}
        @{Component='PC-AI.Performance';Name='Get-PcaiOptimizationPlan'}
        @{Component='PC-AI.Virtualization';Name='Set-WSLDefenderExclusions'}
    ) {
        $r=Invoke-DeclaredApiConsumer $Component
        $r.ExitCode|Should -Be 0 -Because $r.StdErr
        $r.Data.Declared|Should -Contain $Name
        $row=@($r.Data.Rows|Where-Object Name -EQ $Name)
        $row|Should -HaveCount 1
        $row[0].Present|Should -BeTrue
        $row[0].Module|Should -BeExactly $Component
        [IO.Path]::GetFullPath($row[0].SourceFile)|Should -Be ([IO.Path]::GetFullPath((Join-Path $r.ComponentRoot "Public/$Name.ps1")))
    }
}
