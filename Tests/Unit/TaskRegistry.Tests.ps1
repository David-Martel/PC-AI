#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

# Why this file exists: docs/scheduled-tasks is the owner's answer to "what is this task for and how will
# I know it failed?" without guessing. These tests keep the committed registry honest: every row has an
# intent and a failure signal, nothing is left UNKNOWN silently, the JSON and the Markdown describe the same
# tasks, and the curated notes cannot contain a duplicate key that would shadow another task.

BeforeAll {
    $script:Dir = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'docs\scheduled-tasks'
    $script:Notes = @(Get-Content -LiteralPath (Join-Path $script:Dir 'task-registry-notes.json') -Raw | ConvertFrom-Json)
    $script:Registry = Get-Content -LiteralPath (Join-Path $script:Dir 'task-registry.json') -Raw | ConvertFrom-Json
    $script:Markdown = Get-Content -LiteralPath (Join-Path $script:Dir 'TASK_REGISTRY.md') -Raw
}

Describe 'task-registry-notes.json' {
    It 'has one entry per task (no duplicate path + name)' {
        $keys = $script:Notes | ForEach-Object { "$($_.path)$($_.name)" }
        ($keys | Group-Object | Where-Object Count -gt 1) | Should -BeNullOrEmpty
    }

    It 'gives every entry a purpose, an owner and a known confidence' {
        foreach ($n in $script:Notes) {
            $n.purpose | Should -Not -BeNullOrEmpty -Because "$($n.path)$($n.name) needs a purpose"
            $n.owner | Should -Not -BeNullOrEmpty
            'read', 'vendor', 'unknown' | Should -Contain $n.confidence
        }
    }

    It 'records what was read for every task that is not a vendor task' {
        foreach ($n in $script:Notes | Where-Object confidence -eq 'read') {
            @($n.checked).Count | Should -BeGreaterThan 0 -Because "$($n.name) claims to have been read"
        }
    }
}

Describe 'task-registry.json' {
    It 'is internally consistent' {
        $script:Registry.counts.total | Should -Be @($script:Registry.tasks).Count
        ($script:Registry.counts.own + $script:Registry.counts.vendor) | Should -Be $script:Registry.counts.total
    }

    It 'leaves no purpose UNKNOWN' {
        @($script:Registry.tasks | Where-Object { $_.purpose -match '^UNKNOWN' }) | Should -BeNullOrEmpty
    }

    It 'gives every one of our tasks a failure signal and a dtm-p1gen9 note' {
        foreach ($t in $script:Registry.tasks | Where-Object category -eq 'own') {
            $t.failureSignal | Should -Not -BeNullOrEmpty -Because "$($t.task) must say where a failure shows up"
            $t.p1gen9 | Should -Match '^(yes|no|unknown)' -Because "$($t.task) needs a dtm-p1gen9 answer"
        }
    }

    It 'points every launcher task at its status file and log' {
        foreach ($t in $script:Registry.tasks | Where-Object launcher -eq 'dtm-task-run') {
            $t.launcherId | Should -Not -BeNullOrEmpty
            $t.statusFile | Should -Match '\\dtm-tasks\\status\\.+\.json$'
            $t.logFile | Should -Match '\\dtm-tasks\\logs\\.+\.log$'
        }
    }

    It 'explains why each notify-bus task notifies' {
        foreach ($t in $script:Registry.tasks | Where-Object notifyBus) {
            $t.notifyBusReason | Should -Not -BeNullOrEmpty -Because "$($t.task) notifies the bus; the owner asked for the reason to be documented"
        }
    }

    It 'documents every task that is still behind conhost --headless as not migrated, with a reason' {
        foreach ($t in $script:Registry.tasks | Where-Object launcher -eq 'conhost --headless') {
            $t.launcherNote | Should -Match 'NOT migrated' -Because "$($t.task) hides its exit code"
        }
    }

    It 'carries no secret-looking assignment' {
        $raw = Get-Content -LiteralPath (Join-Path $script:Dir 'task-registry.json') -Raw
        $raw | Should -Not -Match '(?i)(token|passw(or)?d|secret|api[-_]?key|bearer)\s*[:=]\s*[^\s",*]{4,}'
    }
}

Describe 'TASK_REGISTRY.md' {
    It 'mentions every task in the JSON registry' {
        foreach ($t in $script:Registry.tasks) {
            $script:Markdown.Contains($t.task) | Should -BeTrue -Because "$($t.task) is missing from the Markdown"
        }
    }

    It 'links to the launcher document and the machine-readable registry' {
        $script:Markdown | Should -Match 'LAUNCHER\.md'
        $script:Markdown | Should -Match 'task-registry\.json'
    }
}
