#Requires -Version 7.0

BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '../../Modules/PC-AI.Evaluation/PC-AI.Evaluation.psd1') -Force
    function Invoke-StatisticsFixture {
        param([double[]]$A, [double[]]$B, [double]$Alpha=0.05)
        $test = New-ABTest -Name 'statistics-reference' -VariantAName 'first' -VariantBName 'second'
        foreach ($score in $A) { Add-ABTestResult -TestName $test.Name -Variant A -Score $score }
        foreach ($score in $B) { Add-ABTestResult -TestName $test.Name -Variant B -Score $score }
        Get-ABTestAnalysis -TestName $test.Name -Alpha $Alpha
    }
}
AfterAll { Remove-Module PC-AI.Evaluation -Force -ErrorAction SilentlyContinue }

Describe 'Welch inference uses the actual two-sided Student t probability' {
    It 'does not declare two tiny overlapping samples a significant improvement' {
        $result = Invoke-StatisticsFixture -A @(0.0,0.5) -B @(0.5,1.0)
        # df=2: P(|T|>=sqrt(2)) = 1 - 1/sqrt(2), independently analytic.
        [Math]::Abs($result.PValue - (1.0 - 1.0/[Math]::Sqrt(2.0))) | Should -BeLessThan 0.00005
        $result.DegreesOfFreedom | Should -Be 2.0
        $result.StatisticallySignificant | Should -BeFalse
        $result.Winner | Should -BeExactly 'inconclusive'
    }
    It 'returns probability one for identical nonconstant samples' {
        $result = Invoke-StatisticsFixture -A @(0.0,0.5,1.0) -B @(0.0,0.5,1.0)
        $result.PValue | Should -Be 1.0
        $result.StatisticallySignificant | Should -BeFalse
    }
    It 'retains a genuine large separation and swaps only its direction' {
        $forward = Invoke-StatisticsFixture -A @(0.0,0.1,0.2,0.3) -B @(0.7,0.8,0.9,1.0)
        $reverse = Invoke-StatisticsFixture -A @(0.7,0.8,0.9,1.0) -B @(0.0,0.1,0.2,0.3)
        $forward.PValue | Should -BeLessThan 0.001
        $forward.PValue | Should -Be $reverse.PValue
        $forward.TStatistic | Should -Be (-$reverse.TStatistic)
        $forward.Winner | Should -BeExactly 'second'
        $reverse.Winner | Should -BeExactly 'first'
    }
    It 'rejects an invalid significance level <Value>' -ForEach @(
        @{Value=0.0}, @{Value=1.0}, @{Value=[double]::NaN}, @{Value=[double]::PositiveInfinity}
    ) {
        { Invoke-StatisticsFixture -A @(0.0,0.5) -B @(0.5,1.0) -Alpha $Value -ErrorAction Stop } | Should -Throw '*Alpha*'
    }
    It 'returns a reviewable error for zero sample variance' {
        $result = Invoke-StatisticsFixture -A @(0.5,0.5) -B @(0.8,0.8)
        $result.Error | Should -Match 'variance'
        $result.ContainsKey('Winner') | Should -BeFalse
    }
    It 'refuses a non-finite observation <Value>' -ForEach @(
        @{Value=[double]::NaN}, @{Value=[double]::PositiveInfinity}, @{Value=[double]::NegativeInfinity}
    ) {
        $result = Invoke-StatisticsFixture -A @(0.0,$Value) -B @(0.5,1.0)
        $result.Error | Should -Match 'finite'
        $result.ContainsKey('Winner') | Should -BeFalse
    }
    It 'preserves the insufficient-samples boundary' {
        $result = Invoke-StatisticsFixture -A @(0.0) -B @(0.5,1.0)
        $result.Error | Should -Match 'Insufficient'
        $result.VariantASamples | Should -Be 1
    }
    It 'handles one constant group with the df-one Cauchy reference' {
        $result = Invoke-StatisticsFixture -A @(0.0,0.0) -B @(0.0,1.0)
        $result.PValue | Should -Be 0.5
        $result.DegreesOfFreedom | Should -Be 1.0
        $result.StatisticallySignificant | Should -BeFalse
    }
    It 'retains the same inference under a common <Scale> rescaling' -ForEach @(
        @{Scale=1e150}, @{Scale=1e-150}
    ) {
        $result = Invoke-StatisticsFixture -A @(0.0,(0.5*$Scale)) -B @((0.5*$Scale),$Scale)
        $result.PValue | Should -Be 0.2929
        $result.TStatistic | Should -Be 1.4142
        $result.StatisticallySignificant | Should -BeFalse
    }
    It 'returns an error when the output summary cannot represent its difference' {
        $result = Invoke-StatisticsFixture -A @(-1e308,-9e307) -B @(9e307,1e308)
        $result.Error | Should -Match 'numerical domain'
        $result.ContainsKey('Winner') | Should -BeFalse
    }
    It 'refuses a relative improvement that cannot be represented as a finite summary' {
        $result = Invoke-StatisticsFixture -A @(1e-308,1e-308) -B @(0.5,1.0)
        $result.Error | Should -Match 'numerical domain'
        $result.ContainsKey('Winner') | Should -BeFalse
    }
    It 'decides significance from the unrounded probability at alpha <Alpha>' -ForEach @(
        @{Alpha=0.29289; Significant=$false}, @{Alpha=0.292895; Significant=$true}
    ) {
        $result = Invoke-StatisticsFixture -A @(0.0,0.5) -B @(0.5,1.0) -Alpha $Alpha
        $result.PValue | Should -Be 0.2929
        $result.StatisticallySignificant | Should -Be $Significant
    }
    It 'returns one scalar effect label <Label> for a standardized difference <Effect>' -ForEach @(
        @{Effect=0.0;Label='negligible'}
        @{Effect=0.19;Label='negligible'}
        @{Effect=0.21;Label='small'}
        @{Effect=0.49;Label='small'}
        @{Effect=0.51;Label='medium'}
        @{Effect=0.79;Label='medium'}
        @{Effect=0.81;Label='large'}
    ) {
        $shift=$Effect/[Math]::Sqrt(2.0)
        $result=Invoke-StatisticsFixture -A @(0.0,1.0) -B @($shift,(1.0+$shift))
        ($result.EffectSize -is [string]) | Should -BeTrue
        $result.EffectSize | Should -BeExactly $Label
    }
}

Describe 'Student t probabilities agree with independent references' {
    It 'matches <Reference> within numerical precision' -ForEach @(
        @{Reference='Cauchy t=1'; Statistic=1.0; Df=1.0; Expected=0.5}
        @{Reference='Cauchy t=sqrt(3)'; Statistic=[Math]::Sqrt(3.0); Df=1.0; Expected=1.0/3.0}
        @{Reference='df=2 analytic'; Statistic=[Math]::Sqrt(2.0); Df=2.0; Expected=1.0-1.0/[Math]::Sqrt(2.0)}
        # Published SciPy ttest_ind(equal_var=False) examples; no package dependency.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
        @{Reference='SciPy identical-means example'; Statistic=-0.4390847099199348; Df=997.4602304121448; Expected=0.6606952553131064}
        @{Reference='SciPy unequal-variance example'; Statistic=-1.637098448290542; Df=765.1098655246868; Expected=0.10202110497954867}
        @{Reference='SciPy unequal-count example'; Statistic=-1.3146566100751664; Df=110.41349083985212; Expected=0.1913495266513811}
        @{Reference='SciPy different-means example'; Statistic=-1.8686598649188084; Df=109.32167496550137; Expected=0.06434714193919686}
        @{Reference='zero statistic'; Statistic=0.0; Df=1.0; Expected=1.0}
    ) {
        $probability = & (Get-Module PC-AI.Evaluation) {
            param($statistic, $df)
            Get-StudentTProbability -Statistic $statistic -DegreesOfFreedom $df
        } $Statistic $Df
        [Math]::Abs($probability-$Expected) | Should -BeLessThan 2e-11
    }
    It 'refuses invalid private probability inputs <Statistic>/<Df>' -ForEach @(
        @{Statistic=[double]::NaN; Df=1.0}
        @{Statistic=[double]::PositiveInfinity; Df=1.0}
        @{Statistic=1.0; Df=0.0}
        @{Statistic=1.0; Df=[double]::NaN}
    ) {
        { & (Get-Module PC-AI.Evaluation) {
            param($statistic, $df)
            Get-StudentTProbability -Statistic $statistic -DegreesOfFreedom $df
        } $Statistic $Df } | Should -Throw '*finite*'
    }
    It 'preserves a representable Cauchy tail when squaring the statistic would overflow' {
        $probability = & (Get-Module PC-AI.Evaluation) { Get-StudentTProbability -Statistic 1e200 -DegreesOfFreedom 1.0 }
        # Cauchy survival: 2*atan(1/t)/pi, avoiding pi/2 cancellation.
        $expected = 2.0*[Math]::Atan(1e-200)/[Math]::PI
        [Math]::Abs($probability/$expected-1.0) | Should -BeLessThan 1e-12
    }
}
