#Requires -Version 5.1
param([string]$SourcePath,[string]$SourceSHA256,[string]$FixtureRoot)
# Complete maintained fallback function; the native command lookup alone is suppressed.
# Private NoProfile baseline/candidate runs separately qualify absence of native dispatch.
BeforeAll {
    # Legacy discovery reports unsupported controls as SKIP, never PASS.
    if ($PSVersionTable.PSVersion -lt [version]'7.2') { return }
    Set-StrictMode -Version Latest
    $ErrorActionPreference='Stop'
    if(-not$SourcePath){$SourcePath=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../Modules/PC-AI.Acceleration/Public/Get-ProcessLassoSnapshot.ps1'))}
    if(-not$SourceSHA256){$SourceSHA256=(Get-FileHash -LiteralPath $SourcePath).Hash}
    if(-not$FixtureRoot){$FixtureRoot=Join-Path $TestDrive 'lasso-inputs'}
    # Protects: CI deterministically covers the real fallback even with other modules loaded.
    # Detects: configuration/list regressions in the actual complete maintained function.
    # Needs: native lookup suppressed only; no parser, input or snapshot result mocks.
    # Breadcrumb: separate unchanged fourteen-case NoProfile runs reproduce predecessor failures.
    Mock Get-Command { } -ParameterFilter { $Name -eq 'Invoke-PcaiNativeProcessLassoSnapshot' }
    if($SourceSHA256-cnotmatch'^[A-F0-9]{64}$'-or(Get-FileHash -LiteralPath $SourcePath).Hash-cne$SourceSHA256){throw 'Reviewed complete parser input differs.'}
    if(Get-Command Invoke-PcaiNativeProcessLassoSnapshot -ErrorAction SilentlyContinue){throw 'Fresh NoProfile host must have no native snapshot command.'}
    $tokens=$null;$errors=$null
    $script:SourceAst=[Management.Automation.Language.Parser]::ParseFile($SourcePath,[ref]$tokens,[ref]$errors)
    $top=@($script:SourceAst.EndBlock.Statements)
    if($errors.Count-or$top.Count-ne1-or$top[0]-isnot[Management.Automation.Language.FunctionDefinitionAst]-or$top[0].Name-cne'Get-ProcessLassoSnapshot'){throw 'Complete actual source must contain one declarations-only function.'}
    . $SourcePath
    if((Get-Command Get-ProcessLassoSnapshot -CommandType Function).ScriptBlock.Ast.Extent.Text-cne$top[0].Extent.Text){throw 'Actual complete function binding differs.'}
    if([IO.Directory]::Exists($FixtureRoot)-or[IO.File]::Exists($FixtureRoot)){throw 'Fixture namespace must be absent.'}
    [void][IO.Directory]::CreateDirectory($FixtureRoot)
    $script:Outcomes=[Collections.Generic.List[object]]::new()
    $script:CompleteIni=@'
; literal comments are ignored
# second comment
[PowerManagement]
StartWithPowerPlan=Balanced
[GamingMode]
GamingModeEnabled=false
TargetPowerPlan=literal=embedded
[Logging]
LogCPUSets=true
LogCPUSets=false
LogEfficiencyMode=true
[OutOfControlProcessRestraint]
OocExclusions=explorer.exe,工具.exe
[MemoryManagement]
SmartTrimExclusions=工具.exe
[ProcessDefaults]
DefaultPriorities=工具.exe,High,explorer.exe,Normal
[ProcessAllowances]
EfficiencyMode=工具.exe,0,explorer.exe,0
'@
    function Write-FixtureBytes([string]$Path,[string]$Text,[string]$Encoding){
        $encoder=switch($Encoding){Utf8{[Text.UTF8Encoding]::new($false,$true)}Utf8Bom{[Text.UTF8Encoding]::new($true,$true)}Utf16LE{[Text.UnicodeEncoding]::new($false,$true,$true)}Utf16BE{[Text.UnicodeEncoding]::new($true,$true,$true)}default{throw 'Explicit fixture encoding required.'}}
        $bytes=[byte[]]@($encoder.GetPreamble()+$encoder.GetBytes($Text))
        if($bytes.Length-gt8192){throw 'Tiny fixture cap exceeded.'}
        $stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try{$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
    }
    function Get-FixtureInventory([string]$Root){@([IO.Directory]::EnumerateFiles($Root)|Sort-Object|ForEach-Object{[pscustomobject]@{Path=$_;Bytes=([IO.FileInfo]::new($_)).Length;SHA256=(Get-FileHash -LiteralPath $_).Hash}})}
    function Invoke-ActualSnapshot([string]$Name,[string]$Ini=$script:CompleteIni,[string]$Encoding='Utf16LE',[string]$Log='',[switch]$Missing){
        Set-StrictMode -Version Latest
        $root=Join-Path $FixtureRoot $Name
        if([IO.Directory]::Exists($root)){throw 'Owned fixture collision.'}
        [void][IO.Directory]::CreateDirectory($root)
        $config=Join-Path $root 'prolasso.ini';$logPath=Join-Path $root 'processlasso.log'
        if(-not$Missing){Write-FixtureBytes $config $Ini $Encoding;Write-FixtureBytes $logPath $Log Utf8}
        $before=Get-FixtureInventory $root;$output=@();$cause=$null;$readOnly=$false
        $originalError=$null;$after=@();$diagnosticErrors=[Collections.Generic.List[Management.Automation.ErrorRecord]]::new()
        try{
            if(Get-Command Invoke-PcaiNativeProcessLassoSnapshot -ErrorAction SilentlyContinue){throw 'Ambient native command appeared.'}
            $output=@(Get-ProcessLassoSnapshot -ConfigPath $config -LogPath $logPath -LookbackMinutes 60 -ErrorAction Stop)
            $output.Count|Should -Be 1
            $output[0].config_path|Should -BeExactly $config
            $output[0].log_path|Should -BeExactly $logPath
        }catch{$originalError=$_;$cause=$_.Exception.ToString()}
        finally{
            # Preserve the original parser/assertion ErrorRecord before any audit IO.
            try{
                $after=Get-FixtureInventory $root
                $beforeText=ConvertTo-Json -InputObject @($before) -Depth 5 -Compress
                $afterText=ConvertTo-Json -InputObject @($after) -Depth 5 -Compress
                $readOnly=$beforeText-ceq$afterText
                if(-not$readOnly){throw 'Actual parser changed tiny fixture bytes or inventory.'}
            }catch{$diagnosticErrors.Add($_)}
            try{$script:Outcomes.Add([ordered]@{Name=$Name;Before=$before;After=$after;ReadOnly=$readOnly;ParserFailure=$cause;DiagnosticSecondary=@($diagnosticErrors|ForEach-Object{$_.Exception.ToString()})})}
            catch{$diagnosticErrors.Add($_)}
        }
        if($diagnosticErrors.Count){
            # Native owner captures bounded stderr; full records remain in this scope.
            foreach($errorRecord in $diagnosticErrors){
                try{$text=$errorRecord.Exception.ToString();[Console]::Error.WriteLine('FIXTURE_AUDIT_SECONDARY: '+$text.Substring(0,[Math]::Min(2048,$text.Length)))}catch{}
            }
        }
        if($null-ne$originalError){throw $originalError}
        if($diagnosticErrors.Count){throw $diagnosticErrors[0]}
        return $output[0]
    }
    function Assert-CompleteIni([object]$Snapshot){
        $Snapshot.sections.PowerManagement.StartWithPowerPlan|Should -BeExactly 'Balanced'
        $Snapshot.sections.GamingMode.TargetPowerPlan|Should -BeExactly 'literal=embedded'
        $Snapshot.sections.Logging.LogCPUSets|Should -BeExactly 'false'
        @($Snapshot.summary.ooc_exclusions).Count|Should -Be 2
        $Snapshot.summary.ooc_exclusions[0]|Should -BeExactly 'explorer.exe'
        $Snapshot.summary.ooc_exclusions[1]|Should -BeExactly '工具.exe'
        @($Snapshot.summary.smart_trim_exclusions).Count|Should -Be 1
        $Snapshot.summary.smart_trim_exclusions[0]|Should -BeExactly '工具.exe'
        $Snapshot.summary.default_priorities.'工具.exe'|Should -BeExactly 'High'
        $Snapshot.summary.default_priorities.'explorer.exe'|Should -BeExactly 'Normal'
        @($Snapshot.summary.efficiency_mode_off).Count|Should -Be 2
        $Snapshot.summary.efficiency_mode_off[0]|Should -BeExactly '工具.exe'
        $Snapshot.summary.efficiency_mode_off[1]|Should -BeExactly 'explorer.exe'
    }
}

Describe 'Actual complete Process Lasso fallback parser against private literal files' -Tag 'Unit','Acceleration','Portable' -Skip:($PSVersionTable.PSVersion -lt [version]'7.2') {
    # Protects: fixture executes the complete maintained function in isolation.
    # Detects: mirror parser, native fallback stub or wrong dot-source binding.
    # Needs: full-source pin and no Invoke-PcaiNativeProcessLassoSnapshot command.
    # Breadcrumb: public source function AST extent is compared before invocation.
    It 'binds the exact complete canonical function and excludes native dispatch' {
        (Get-Command Get-ProcessLassoSnapshot).ScriptBlock.Ast.Extent.Text|Should -BeExactly $script:SourceAst.EndBlock.Statements[0].Extent.Text
        @(Get-Command Invoke-PcaiNativeProcessLassoSnapshot -ErrorAction SilentlyContinue).Count|Should -Be 0
    }
    # Protects: valid legacy UTF16 LE input remains functional.
    # Detects: loss of existing config fields, nonASCII, duplicate or embedded equals.
    # Needs: independently literal expected fields, complete priority/efficiency pairs.
    # Breadcrumb: original parser's positive encoding control retained before repair.
    It 'retains valid UTF16 LE sections lists and literal values' {Assert-CompleteIni (Invoke-ActualSnapshot 'utf16-le' -Encoding Utf16LE)}
    # Protects: BOM-marked big-endian UTF16 represents the same config.
    # Detects: fixed-endian decoder ignoring BOM identity.
    # Needs: actual big-endian physical bytes and literal expected fields.
    # Breadcrumb: config decode is tested via full public function.
    It 'retains UTF16 BE BOM sections lists and literal values' {Assert-CompleteIni (Invoke-ActualSnapshot 'utf16-be' -Encoding Utf16BE)}
    # Protects: repo-produced BOM-less UTF8 INI is consumable.
    # Detects: silent Unicode decoding of UTF8 bytes without a thrown read error.
    # Needs: exact UTF8 bytes including nonASCII and complete known pair lists.
    # Breadcrumb: Get-ProcessLassoText always selected Unicode in predecessor.
    It 'retains UTF8 without BOM sections lists and literal values' {Assert-CompleteIni (Invoke-ActualSnapshot 'utf8' -Encoding Utf8)}
    # Protects: BOM-marked UTF8 remains equivalent to other admitted encodings.
    # Detects: BOM text leaking into initial section/comment parsing.
    # Needs: explicit UTF8 BOM bytes and same literal expected sections.
    # Breadcrumb: BOM-aware decode proposal must preserve config values.
    It 'retains UTF8 BOM sections lists and literal values' {Assert-CompleteIni (Invoke-ActualSnapshot 'utf8-bom' -Encoding Utf8Bom)}
    # Protects: empty inputs remain a valid empty read-only snapshot.
    # Detects: null comma-list Count exception under StrictMode Latest.
    # Needs: physically zero-byte INI/log and unchanged zero assertions.
    # Breadcrumb: Get-CommaList empty output becomes scalar/null at call sites.
    It 'returns one empty snapshot for zero-byte config and log under StrictMode' {
        $s=Invoke-ActualSnapshot 'empty' -Ini '' -Encoding Utf8
        @($s.summary.ooc_exclusions).Count|Should -Be 0;@($s.summary.smart_trim_exclusions).Count|Should -Be 0
        @($s.summary.efficiency_mode_off).Count|Should -Be 0;@($s.summary.default_priorities.PSObject.Properties).Count|Should -Be 0
        $s.log_summary.total_events|Should -Be 0
    }
    # Protects: nonexistent explicit fixture paths do not create config/log.
    # Detects: missing-input cardinality exceptions or implicit installed fallback.
    # Needs: same actual parser, absent files, zero expected values.
    # Breadcrumb: public Test-Path guards retain existing missing-file contract.
    It 'returns one empty snapshot for missing explicit paths under StrictMode' {
        $s=Invoke-ActualSnapshot 'missing' -Missing
        @($s.summary.efficiency_mode_off).Count|Should -Be 0;@($s.summary.default_priorities.PSObject.Properties).Count|Should -Be 0
        $s.log_summary.total_events|Should -Be 0
    }
    # Protects: zero comma-list entries are arrays, not null scalar failures.
    # Detects: null Count/member access despite valid config sections.
    # Needs: literal empty priority/efficiency/exclusion values encoded UTF16.
    # Breadcrumb: two call sites must collect Get-CommaList output explicitly.
    It 'preserves zero lists in otherwise valid UTF16 config' {
        $ini="[ProcessDefaults]`nDefaultPriorities=`n[ProcessAllowances]`nEfficiencyMode=`n[OutOfControlProcessRestraint]`nOocExclusions=`n[MemoryManagement]`nSmartTrimExclusions="
        $s=Invoke-ActualSnapshot 'zero-lists' -Ini $ini
        @($s.summary.default_priorities.PSObject.Properties).Count|Should -Be 0;@($s.summary.efficiency_mode_off).Count|Should -Be 0
        @($s.summary.ooc_exclusions).Count|Should -Be 0;@($s.summary.smart_trim_exclusions).Count|Should -Be 0
    }
    # Protects: one complete pair and one exclusion retain exact names.
    # Detects: singleton output/indexing changing multi-character executable identity.
    # Needs: literal complete pair and positive count-one controls.
    # Breadcrumb: no priority/efficiency flag semantics change is proposed.
    It 'preserves one priority pair efficiency pair and exclusion' {
        $ini="[ProcessDefaults]`nDefaultPriorities=solo.exe,High`n[ProcessAllowances]`nEfficiencyMode=solo.exe,0`n[OutOfControlProcessRestraint]`nOocExclusions=solo.exe`n[MemoryManagement]`nSmartTrimExclusions=solo.exe"
        $s=Invoke-ActualSnapshot 'one-pair' -Ini $ini
        @($s.summary.default_priorities.PSObject.Properties).Count|Should -Be 1;$s.summary.default_priorities.'solo.exe'|Should -BeExactly 'High'
        @($s.summary.efficiency_mode_off).Count|Should -Be 1;$s.summary.efficiency_mode_off[0]|Should -BeExactly 'solo.exe'
        @($s.summary.ooc_exclusions).Count|Should -Be 1;$s.summary.ooc_exclusions[0]|Should -BeExactly 'solo.exe'
        @($s.summary.smart_trim_exclusions).Count|Should -Be 1;$s.summary.smart_trim_exclusions[0]|Should -BeExactly 'solo.exe'
    }
    # Protects: existing odd-tail policy retains full efficiency name, drops incomplete priority.
    # Detects: scalar string indexing or null Count confusion, not flag semantics.
    # Needs: single raw list entry, exact full executable and zero priority assertions.
    # Breadcrumb: native chunks-two policy and intended public array iteration agree.
    It 'preserves whole singleton efficiency name and empty incomplete priority map' {
        $ini="[ProcessDefaults]`nDefaultPriorities=solo.exe`n[ProcessAllowances]`nEfficiencyMode=solo.exe"
        $s=Invoke-ActualSnapshot 'one-entry' -Ini $ini
        @($s.summary.default_priorities.PSObject.Properties).Count|Should -Be 0
        @($s.summary.efficiency_mode_off).Count|Should -Be 1;$s.summary.efficiency_mode_off[0]|Should -BeExactly 'solo.exe'
    }
    # Protects: multiple list pairs preserve counts, identities and ordering.
    # Detects: repair collapsing lists or skipping complete pairs.
    # Needs: three literal complete pairs and exclusions with known expected values.
    # Breadcrumb: collection repair must retain existing pair iteration policy.
    It 'preserves three complete priority efficiency and exclusion entries' {
        $ini="[ProcessDefaults]`nDefaultPriorities=a.exe,High,b.exe,Normal,c.exe,Low`n[ProcessAllowances]`nEfficiencyMode=a.exe,0,b.exe,0,c.exe,0`n[OutOfControlProcessRestraint]`nOocExclusions=a.exe,b.exe,c.exe`n[MemoryManagement]`nSmartTrimExclusions=a.exe,b.exe,c.exe"
        $s=Invoke-ActualSnapshot 'many' -Ini $ini
        @($s.summary.default_priorities.PSObject.Properties).Count|Should -Be 3
        $s.summary.default_priorities.'a.exe'|Should -BeExactly 'High';$s.summary.default_priorities.'b.exe'|Should -BeExactly 'Normal';$s.summary.default_priorities.'c.exe'|Should -BeExactly 'Low'
        (@($s.summary.efficiency_mode_off)-join ',')|Should -BeExactly 'a.exe,b.exe,c.exe'
        (@($s.summary.ooc_exclusions)-join ',')|Should -BeExactly 'a.exe,b.exe,c.exe'
        (@($s.summary.smart_trim_exclusions)-join ',')|Should -BeExactly 'a.exe,b.exe,c.exe'
    }
    # Protects: valid recent nine-field row counts while old/invalid/short rows do not.
    # Detects: log regression during config/cardinality-only repair.
    # Needs: large local-time margins, literal quoted fields and exact maps.
    # Breadcrumb: local cutoff and CSV delimiter behavior intentionally unchanged.
    It 'counts recent quoted log row and excludes old invalid-date and short rows' {
        $recent=(Get-Date).AddMinutes(-5).ToString('yyyy-MM-dd HH:mm:ss',[Globalization.CultureInfo]::InvariantCulture)
        $old=(Get-Date).AddMinutes(-180).ToString('yyyy-MM-dd HH:mm:ss',[Globalization.CultureInfo]::InvariantCulture)
        $log='"1","'+$recent+'","user","42","x","NOTEPAD.EXE","0","Efficiency Mode OFF","detail"'+"`n"+'"2","'+$old+'","user","42","x","old.exe","0","SmartTrim","detail"'+"`n"+'"3","invalid-date","user","42","x","bad.exe","0","CPU Set","detail"'+"`n"+'"short","row"'
        $s=Invoke-ActualSnapshot 'log-filter' -Log $log
        $s.log_summary.total_events|Should -Be 1;$s.log_summary.efficiency_mode_events|Should -Be 1
        $s.log_summary.cpu_set_events|Should -Be 0;$s.log_summary.smart_trim_events|Should -Be 0;$s.log_summary.power_profile_events|Should -Be 0
        @($s.log_summary.actions.PSObject.Properties).Count|Should -Be 1;$s.log_summary.actions.'Efficiency Mode OFF'|Should -Be 1
        @($s.log_summary.processes.PSObject.Properties).Count|Should -Be 1;$s.log_summary.processes.'notepad.exe'|Should -Be 1
    }
    # Protects: independent event categories and duplicate process accumulation remain correct.
    # Detects: loss of category counters or per-process/action aggregation.
    # Needs: four complete literal rows, expected total/category/map values.
    # Breadcrumb: no timezone, future-time, CSV dialect or Boolean contract changes.
    It 'counts four literal actions and exact process aggregation' {
        $recent=(Get-Date).AddMinutes(-5).ToString('yyyy-MM-dd HH:mm:ss',[Globalization.CultureInfo]::InvariantCulture)
        $log=@(('"1","'+$recent+'","user","42","x","notepad.exe","0","Efficiency Mode OFF","detail"'),('"2","'+$recent+'","user","42","x","notepad.exe","0","CPU Set changed","detail"'),('"3","'+$recent+'","user","42","x","tool.exe","0","SmartTrim applied","detail"'),('"4","'+$recent+'","user","42","x","tool.exe","0","Power profile changed","detail"'))-join "`n"
        $s=Invoke-ActualSnapshot 'log-counters' -Log $log
        $s.log_summary.total_events|Should -Be 4;$s.log_summary.efficiency_mode_events|Should -Be 1;$s.log_summary.cpu_set_events|Should -Be 1;$s.log_summary.smart_trim_events|Should -Be 1;$s.log_summary.power_profile_events|Should -Be 1
        @($s.log_summary.actions.PSObject.Properties).Count|Should -Be 4;@($s.log_summary.processes.PSObject.Properties).Count|Should -Be 2
        $s.log_summary.processes.'notepad.exe'|Should -Be 2;$s.log_summary.processes.'tool.exe'|Should -Be 2
    }
    # Protects: parser is read-only even when parsing throws in predecessor.
    # Detects: rewritten config/log or unexpected fixture child leaves.
    # Needs: each invocation's actual before/after inventory and hashes captured in finally.
    # Breadcrumb: no global config/service/process action occurs in this suite.
    It 'preserves exact input bytes and inventories for all actual parser invocations' {
        $script:Outcomes.Count|Should -Be 12
        @($script:Outcomes|Where-Object{-not$_.ReadOnly}).Count|Should -Be 0
        @([IO.Directory]::EnumerateDirectories($FixtureRoot)).Count|Should -Be 12
    }
}
AfterAll {
    if ($PSVersionTable.PSVersion -lt [version]'7.2') { return }
    if((Get-FileHash -LiteralPath $SourcePath).Hash-cne$SourceSHA256){throw 'Actual complete source changed.'}
    if($script:Outcomes){
        $receipt=Join-Path $FixtureRoot 'input-readback.json'
        $bytes=[Text.UTF8Encoding]::new($false,$true).GetBytes(($script:Outcomes.ToArray()|ConvertTo-Json -Depth 8))
        $stream=[IO.File]::Open($receipt,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try{$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
    }
}
