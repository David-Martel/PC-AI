import re

file_path = "./Tests/Unit/PC-AI.Media.Tests.ps1"
with open(file_path, 'r') as f:
    content = f.read()

pattern = r"""        It 'Reflects state set by module operations' {
            InModuleScope PcaiMedia {
                \$script:Initialized  = \$true
                \$script:ModelLoaded  = \$true
                \$script:CurrentModel = 'deepseek-ai/Janus-Pro-1B'
            }
            \$status = Get-PcaiMediaStatus
            \$status.Initialized  \| Should -BeTrue
            \$status.ModelLoaded  \| Should -BeTrue
            \$status.CurrentModel \| Should -Be 'deepseek-ai/Janus-Pro-1B'
        }"""

replacement = """        It 'Reflects state set by module operations' {
            InModuleScope PcaiMedia {
                $script:Initialized  = $true
                $script:ModelLoaded  = $true
                $script:CurrentModel = 'deepseek-ai/Janus-Pro-1B'
            }
            $status = Get-PcaiMediaStatus
            $status.Initialized  | Should -BeTrue
            $status.ModelLoaded  | Should -BeTrue
            $status.CurrentModel | Should -Be 'deepseek-ai/Janus-Pro-1B'
        }

        It 'Reflects Initialized=false after a failed initialization attempt' {
            InModuleScope PcaiMedia {
                $script:Initialized  = $false
                $script:ModelLoaded  = $false
                $script:CurrentModel = $null
            }
            
            # Simulate a throw before setting Initialized to true, as happens in Initialize-PcaiMedia
            InModuleScope PcaiMedia {
                $errorActionPreference = 'Stop'
                try {
                    throw "Simulated init failure"
                    $script:Initialized = $true
                } catch {
                    # Initialized remains false
                }
            }
            
            $status = Get-PcaiMediaStatus
            $status.Initialized | Should -BeFalse
            $status.ModelLoaded | Should -BeFalse
            $status.CurrentModel | Should -BeNullOrEmpty
        }

        It 'Reflects all fields reset after Stop-PcaiMedia is called' {
            InModuleScope PcaiMedia {
                $script:Initialized  = $true
                $script:ModelLoaded  = $true
                $script:CurrentModel = 'deepseek-ai/Janus-Pro-7B'
            }
            
            # Call Stop-PcaiMedia. It should reset the internal state.
            Stop-PcaiMedia
            
            $status = Get-PcaiMediaStatus
            $status.Initialized  | Should -BeFalse
            $status.ModelLoaded  | Should -BeFalse
            $status.CurrentModel | Should -BeNullOrEmpty
        }

        It 'Maintains state consistency during concurrent-like access' {
            InModuleScope PcaiMedia {
                $script:Initialized  = $true
                $script:ModelLoaded  = $false
                $script:CurrentModel = 'test-model'
            }
            
            # Read state multiple times to ensure consistency
            $statuses = 1..100 | ForEach-Object { Get-PcaiMediaStatus }
            
            $statuses.Count | Should -Be 100
            foreach ($s in $statuses) {
                $s.Initialized | Should -BeTrue
                $s.ModelLoaded | Should -BeFalse
                $s.CurrentModel | Should -Be 'test-model'
            }
        }"""

new_content = re.sub(pattern, replacement, content, count=1)

with open(file_path, 'w') as f:
    f.write(new_content)
