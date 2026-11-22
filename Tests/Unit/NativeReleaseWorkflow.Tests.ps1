#Requires -Version 7.0

BeforeAll {
    $script:WorkflowText = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../../.github/workflows/release-cuda.yml') -Raw
    $script:SourceCommit = '0123456789abcdef0123456789abcdef01234567'

    function Invoke-ReleaseSourceCheck {
        param(
            [string]$Event,
            [string]$Ref,
            [string]$RefType,
            [string]$RefName,
            [string]$CheckoutRef = $Ref,
            [string]$CheckoutCommit = $script:SourceCommit
        )

        $pattern = '(?ms)^      - name: Verify Source Revision\r?\n.*?^        run: \|\r?\n(?<body>.*?)(?=^      - |\z)'
        $body = [regex]::Match($script:WorkflowText, $pattern).Groups['body'].Value
        if (-not $body) { throw 'Missing native release source verification step.' }
        $check = [scriptblock]::Create(($body -replace '(?m)^          ', ''))
        $outputPath = Join-Path $TestDrive 'github-output'
        Remove-Item -LiteralPath $outputPath -ErrorAction SilentlyContinue
        $values = @{
            GITHUB_EVENT_NAME = $Event
            GITHUB_REF = $Ref
            GITHUB_REF_TYPE = $RefType
            GITHUB_REF_NAME = $RefName
            GITHUB_SHA = $script:SourceCommit
            GITHUB_OUTPUT = $outputPath
            PCAI_CHECKOUT_REF = $CheckoutRef
            PCAI_CHECKOUT_COMMIT = $CheckoutCommit
        }
        $prior = @{}
        try {
            foreach ($name in $values.Keys) {
                $prior[$name] = [Environment]::GetEnvironmentVariable($name)
                [Environment]::SetEnvironmentVariable($name, $values[$name])
            }
            & $check
            Get-Content -LiteralPath $outputPath -ErrorAction Stop
        }
        finally {
            foreach ($name in $prior.Keys) {
                [Environment]::SetEnvironmentVariable($name, $prior[$name])
            }
        }
    }
}

Describe 'Native release workflow source and permissions' -Tag 'Unit', 'Portable' {
    # Protects: Both stable and prerelease artifacts identify the exact pushed tag and checked-out commit.
    # Detects: Source admission drops the ref, treats the tag name as a different version, or loses the exact SHA.
    # Needs: PowerShell 7, Pester 5, workflow text, and TestDrive. No GitHub or network access.
    # Breadcrumb: .github/workflows/release-cuda.yml Verify Source Revision.
    It 'accepts a checked-out tag and source commit for <Tag>' -ForEach @(
        @{ Tag = 'v1.2.3' },
        @{ Tag = 'v1.2.3-rc.2' }
    ) {
        $ref = "refs/tags/$Tag"
        $output = Invoke-ReleaseSourceCheck -Event push -Ref $ref -RefType tag -RefName $Tag
        $output | Should -Contain "source_ref=$ref"
        $output | Should -Contain "source_commit=$script:SourceCommit"
        $output | Should -Contain "release_tag=$Tag"
        $output | Should -Contain "release_version=$($Tag.Substring(1))"
    }

    # Protects: Only an admitted release tag whose ref and checkout commit match can declare publication inputs.
    # Detects: Commit mismatches, ref mismatches, branch pushes and malformed stable or prerelease versions.
    # Needs: PowerShell 7, Pester 5, workflow text, and TestDrive. No GitHub or network access.
    # Breadcrumb: .github/workflows/release-cuda.yml Verify Source Revision.
    It 'refuses publication inputs when <Case>' -ForEach @(
        @{ Case = 'the checkout commit differs'; Override = @{ CheckoutCommit = '1111111111111111111111111111111111111111' }; ErrorText = '*event source commit*' },
        @{ Case = 'the checkout tag differs'; Override = @{ CheckoutRef = 'refs/tags/v1.2.4' }; ErrorText = '*release tag and checkout*' },
        @{ Case = 'the push is to a branch'; Override = @{ Ref = 'refs/heads/v1.2.3'; RefType = 'branch' }; ErrorText = '*release tag and checkout*' },
        @{ Case = 'the tag contains a leading zero'; Override = @{ Ref = 'refs/tags/v01.2.3'; RefName = 'v01.2.3' }; ErrorText = '*release tag and checkout*' },
        @{ Case = 'a numeric prerelease identifier contains a leading zero'; Override = @{ Ref = 'refs/tags/v1.2.3-rc.01'; RefName = 'v1.2.3-rc.01' }; ErrorText = '*release tag and checkout*' }
    ) {
        $params = @{ Event = 'push'; Ref = 'refs/tags/v1.2.3'; RefType = 'tag'; RefName = 'v1.2.3' }
        foreach ($name in $Override.Keys) { $params[$name] = $Override[$name] }
        { Invoke-ReleaseSourceCheck @params } | Should -Throw -ExpectedMessage $ErrorText
        Test-Path -LiteralPath (Join-Path $TestDrive 'github-output') | Should -BeFalse
    }

    # Protects: Selecting an existing tag during a manual dispatch yields a build-only source SHA.
    # Detects: A manual run supplies any tag or version output to the publication job.
    # Needs: PowerShell 7, Pester 5, workflow text, and TestDrive. No GitHub or network access.
    # Breadcrumb: .github/workflows/release-cuda.yml Verify Source Revision and release job condition.
    It 'admits a manual build without emitting a publication tag' {
        $output = Invoke-ReleaseSourceCheck -Event workflow_dispatch -Ref 'refs/tags/v1.2.3' -RefType tag -RefName 'v1.2.3'
        $output | Should -Contain "source_ref=$script:SourceCommit"
        $output | Should -Contain "source_commit=$script:SourceCommit"
        @($output -match '^release_').Count | Should -Be 0
    }

    # Protects: Only a successful tag build can write a release, with matrix builds using the admitted ref.
    # Detects: Workflow-wide write permissions, a release job callable by dispatch, omitted checkout pinning or missing prerequisites.
    # Needs: PowerShell 7, Pester 5, and workflow text. No GitHub or network access.
    # Breadcrumb: .github/workflows/release-cuda.yml test, build and release jobs.
    It 'limits write access and publication to the checked tag after test and build' {
        $script:WorkflowText | Should -Match '(?m)^permissions: \{\}\s*$'
        $writes = [regex]::Matches($script:WorkflowText, '(?m)^\s*contents: write\s*$')
        $writes.Count | Should -Be 1
        $forbiddenInput = [regex]::Matches($script:WorkflowText, 'github\.event\.inputs\.tag')
        $forbiddenInput.Count | Should -Be 0
        $build = [regex]::Match($script:WorkflowText, '(?ms)^  build:\r?\n(?<job>.*?)(?=^  release:)').Groups['job'].Value
        $release = [regex]::Match($script:WorkflowText, '(?ms)^  release:\r?\n(?<job>.*)$').Groups['job'].Value
        $build | Should -Match '(?m)^      contents: read\s*$'
        $build | Should -Match '(?m)^          ref: \$\{\{ needs\.test\.outputs\.source_ref \}\}\s*$'
        $build | Should -Match '(?m)^        if: steps\.source_checkout\.outputs\.commit != needs\.test\.outputs\.source_commit\s*$'
        $release | Should -Match "(?m)^    if: github\.event_name == 'push' && github\.ref_type == 'tag' && needs\.test\.outputs\.release_tag != ''\s*$"
        $release | Should -Match '(?m)^    needs: \[test, build\]\s*$'
        $release | Should -Match '(?m)^      contents: write\s*$'
        $release | Should -Match '(?m)^          ref: \$\{\{ needs\.test\.outputs\.source_ref \}\}\s*$'
        $release | Should -Match '(?m)^        if: steps\.source_checkout\.outputs\.commit != needs\.test\.outputs\.source_commit\s*$'
        $release | Should -Match '(?m)^          tag_name: \$\{\{ needs\.test\.outputs\.release_tag \}\}\s*$'
    }
}
