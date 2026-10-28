#Requires -Version 7.0
param([string]$CommonModulePath)
Describe 'Common diagnostic streams preserve PowerShell contracts' -Tag 'Unit', 'Common' {
 BeforeAll {
  if (-not $CommonModulePath) { $CommonModulePath=Join-Path $PSScriptRoot '../../Modules/PC-AI.Common/PC-AI.Common.psm1' }
  $script:commonStreamsModule=Import-Module $CommonModulePath -PassThru
  function Invoke-OwnedStreamChild([string]$Body) {
   $path=$CommonModulePath.Replace("'","''")
   $command="`$ErrorActionPreference='Stop'; Import-Module '$path' -Force; $Body"
   $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
   $result=@(& ([Diagnostics.Process]::GetCurrentProcess().MainModule.FileName) -NoLogo -NoProfile -EncodedCommand $encoded)
   if ($LASTEXITCODE -ne 0) { throw 'Owned stream child failed.' }
   ($result -join [Environment]::NewLine) | ConvertFrom-Json -ErrorAction Stop
  }
 }
 It 'emits an ErrorRecord rather than successful host text' {
  $records=@(Write-Error 'Owned diagnostic refusal' -ErrorAction Continue 2>&1 6>$null)
  $records.Count | Should -Be 1
  $records[0] | Should -BeOfType ([System.Management.Automation.ErrorRecord])
  $records[0].Exception.Message | Should -BeExactly 'Owned diagnostic refusal'
 }
 It 'honors explicit ErrorAction Stop' {
  { Write-Error 'Owned stop refusal' -ErrorAction Stop } | Should -Throw '*Owned stop refusal*'
 }
 It 'retains exception and category metadata' {
  $exception=[InvalidOperationException]::new('Owned exception identity')
  $records=@(Write-Error -Exception $exception -Message 'Owned contextual message' -Category InvalidOperation -ErrorId 'Owned.Refusal' -TargetObject 'owned-target' -ErrorAction Continue 2>&1 6>$null)
  $records.Count | Should -Be 1
  [object]::ReferenceEquals($records[0].Exception,$exception) | Should -BeTrue
  $records[0].CategoryInfo.Category | Should -Be ([System.Management.Automation.ErrorCategory]::InvalidOperation)
  $records[0].FullyQualifiedErrorId | Should -Match '^Owned.Refusal'
  $records[0].TargetObject | Should -BeExactly 'owned-target'
 }
 It 'retains an existing ErrorRecord' {
  $record=[System.Management.Automation.ErrorRecord]::new([InvalidOperationException]::new('Owned retained record'),'Owned.Retained',[System.Management.Automation.ErrorCategory]::PermissionDenied,'owned-path')
  $records=@(Write-Error -ErrorRecord $record -ErrorAction Continue 2>&1 6>$null)
  $records.Count | Should -Be 1
  [object]::ReferenceEquals($records[0].Exception,$record.Exception) | Should -BeTrue
  $records[0].CategoryInfo.Category | Should -Be $record.CategoryInfo.Category
  $records[0].TargetObject | Should -BeExactly 'owned-path'
 }
 It 'honors SilentlyContinue without publishing success output' {
  $captured=@()
  $output=@(Write-Error 'Owned silent refusal' -ErrorAction SilentlyContinue -ErrorVariable captured 6>$null)
  $output.Count | Should -Be 0
  @($captured).Count | Should -BeGreaterThan 0
  $captured[-1].Exception.Message | Should -BeExactly 'Owned silent refusal'
 }
 It 'honors Ignore without populating the requested error variable' {
  $captured=@()
  $output=@(Write-Error 'Owned ignored refusal' -ErrorAction Ignore -ErrorVariable captured 6>$null)
  $output.Count | Should -Be 0
  @($captured).Count | Should -Be 0
 }
 It 'preserves pipeline message cardinality' {
  $records=@('owned-first','owned-second' | Write-Error -ErrorAction Continue 2>&1 6>$null)
  $records.Count | Should -Be 2
  $records[0].Exception.Message | Should -BeExactly 'owned-first'
  $records[1].Exception.Message | Should -BeExactly 'owned-second'
 }
 It 'emits WarningRecord on the warning stream' {
  $records=@(Write-Warning 'Owned warning' -WarningAction Continue 3>&1 6>$null)
  $records.Count | Should -Be 1
  $records[0] | Should -BeOfType ([System.Management.Automation.WarningRecord])
  $records[0].Message | Should -BeExactly 'Owned warning'
 }
 It 'honors WarningAction Stop' {
  { Write-Warning 'Owned warning stop' -WarningAction Stop } | Should -Throw '*Owned warning stop*'
 }
 It 'preserves suppressed warning records for WarningVariable' {
  $captured=@()
  $output=@(Write-Warning 'Owned silent warning' -WarningAction SilentlyContinue -WarningVariable captured 6>$null)
  $output.Count | Should -Be 0
  @($captured).Count | Should -Be 1
  $captured[0].Message | Should -BeExactly 'Owned silent warning'
 }
 It 'honors the caller error preference without an explicit action' {
  $ErrorActionPreference='Stop'
  { Write-Error 'Owned inherited error preference' } | Should -Throw '*Owned inherited error preference*'
 }
 It 'honors the caller warning preference without an explicit action' {
  $r=Invoke-OwnedStreamChild '$WarningPreference="Stop"; $stopped=$false; try { Write-Warning "Owned inherited warning preference" 3>$null } catch { $stopped=$_.Exception.Message -like "*Owned inherited warning preference*" }; @{Stopped=$stopped}|ConvertTo-Json -Compress'
  $r.Stopped | Should -BeTrue
 }
 It 'retains the native error message alias' {
  $records=@(Write-Error -Msg 'Owned error alias' -ErrorAction Continue 2>&1)
  $records.Count | Should -Be 1
  $records[0].Exception.Message | Should -BeExactly 'Owned error alias'
 }
 It 'retains the native warning message alias' {
  $records=@(Write-Warning -Msg 'Owned warning alias' -WarningAction Continue 3>&1)
  $records.Count | Should -Be 1
  $records[0].Message | Should -BeExactly 'Owned warning alias'
 }
 It 'appends each error record exactly once' {
  $captured=@()
  Write-Error 'owned-first' -ErrorAction SilentlyContinue -ErrorVariable +captured
  Write-Error 'owned-second' -ErrorAction SilentlyContinue -ErrorVariable +captured
  @($captured).Count | Should -Be 2
  $captured[0].Exception.Message | Should -BeExactly 'owned-first'
  $captured[1].Exception.Message | Should -BeExactly 'owned-second'
 }
 It 'appends each warning record exactly once' {
  $captured=@()
  Write-Warning 'owned-first' -WarningAction SilentlyContinue -WarningVariable +captured
  Write-Warning 'owned-second' -WarningAction SilentlyContinue -WarningVariable +captured
  @($captured).Count | Should -Be 2
  $captured[0].Message | Should -BeExactly 'owned-first'
  $captured[1].Message | Should -BeExactly 'owned-second'
 }
 It 'retains recommended action and category detail overrides' {
  $records=@(Write-Error 'Owned detail' -RecommendedAction 'Review owned receipt' -Activity 'OwnedActivity' -Reason 'OwnedReason' -TargetName 'OwnedName' -TargetType 'OwnedType' -ErrorAction Continue 2>&1)
  $records.Count | Should -Be 1
  $records[0].ErrorDetails.RecommendedAction | Should -BeExactly 'Review owned receipt'
  $records[0].CategoryInfo.Activity | Should -BeExactly 'OwnedActivity'
  $records[0].CategoryInfo.Reason | Should -BeExactly 'OwnedReason'
  $records[0].CategoryInfo.TargetName | Should -BeExactly 'OwnedName'
  $records[0].CategoryInfo.TargetType | Should -BeExactly 'OwnedType'
 }
 It 'appends exactly once in a real consumer process error variable' {
  $r=Invoke-OwnedStreamChild '$caught=@(); Write-Error first -ErrorAction SilentlyContinue -ErrorVariable +caught; Write-Error second -ErrorAction SilentlyContinue -ErrorVariable +caught; @{Count=@($caught).Count;First=$caught[0].Exception.Message;Second=$caught[1].Exception.Message}|ConvertTo-Json -Compress'
  $r.Count | Should -Be 2
  $r.First | Should -BeExactly 'first'
  $r.Second | Should -BeExactly 'second'
 }
 It 'appends exactly once in a real consumer process warning variable' {
  $r=Invoke-OwnedStreamChild '$caught=@(); Write-Warning first -WarningAction SilentlyContinue -WarningVariable +caught; Write-Warning second -WarningAction SilentlyContinue -WarningVariable +caught; @{Count=@($caught).Count;First=$caught[0].Message;Second=$caught[1].Message}|ConvertTo-Json -Compress'
  $r.Count | Should -Be 2
  $r.First | Should -BeExactly 'first'
  $r.Second | Should -BeExactly 'second'
 }
}
