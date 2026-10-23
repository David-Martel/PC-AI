function Get-EvaluationLogGamma {
    param([double]$Value)
    # Lanczos approximation with g=7, for positive arguments >= 0.5.
    $coefficients = @(676.5203681218851, -1259.1392167224028, 771.3234287776531,
        -176.6150291621406, 12.507343278686905, -0.13857109526572012,
        9.984369578019572e-6, 1.5056327351493116e-7)
    $z = $Value - 1.0
    $sum = 0.99999999999980993
    for ($index=0; $index -lt $coefficients.Count; $index++) {
        $sum += $coefficients[$index] / ($z + $index + 1.0)
    }
    $shift = $z + 7.5
    return 0.9189385332046727 + ($z + 0.5)*[Math]::Log($shift) - $shift + [Math]::Log($sum)
}

function Get-EvaluationBetaFraction {
    param([double]$A, [double]$B, [double]$X)
    # Modified Lentz iteration of the incomplete-beta continued fraction.
    # Bounds control numerical convergence, not a statistical approximation.
    $tiny = 1e-300
    $sum = $A + $B
    $c = 1.0
    $d = 1.0 - $sum*$X/($A + 1.0)
    if ([Math]::Abs($d) -lt $tiny) { $d = $tiny }
    $d = 1.0/$d
    $fraction = $d
    for ($iteration=1; $iteration -le 512; $iteration++) {
        $twice = 2.0*$iteration
        $coefficient = $iteration*($B-$iteration)*$X/(($A+$twice-1.0)*($A+$twice))
        $d = 1.0 + $coefficient*$d
        if ([Math]::Abs($d) -lt $tiny) { $d = $tiny }
        $c = 1.0 + $coefficient/$c
        if ([Math]::Abs($c) -lt $tiny) { $c = $tiny }
        $d = 1.0/$d
        $fraction *= $d*$c
        $coefficient = -($A+$iteration)*($sum+$iteration)*$X/(($A+$twice)*($A+$twice+1.0))
        $d = 1.0 + $coefficient*$d
        if ([Math]::Abs($d) -lt $tiny) { $d = $tiny }
        $c = 1.0 + $coefficient/$c
        if ([Math]::Abs($c) -lt $tiny) { $c = $tiny }
        $d = 1.0/$d
        $change = $d*$c
        $fraction *= $change
        if ([Math]::Abs($change-1.0) -le 3e-14) { return $fraction }
    }
    throw 'Student t probability did not converge.'
}

function Get-StudentTProbability {
    <#
    .SYNOPSIS
        Computes a two-sided Student t tail probability.
    .DESCRIPTION
        Uses I_x(df/2,1/2), x=df/(df+t*t), with the symmetry relation
        and continued fraction from https://dlmf.nist.gov/8.17.
        Refuses invalid inputs or failed convergence instead of inventing confidence.
    #>
    param([double]$Statistic, [double]$DegreesOfFreedom)
    if (-not [double]::IsFinite($Statistic) -or -not [double]::IsFinite($DegreesOfFreedom) -or $DegreesOfFreedom -lt 1.0) {
        throw 'Student t probability requires finite statistic and degrees of freedom >= 1.'
    }
    if ($Statistic -eq 0.0) { return 1.0 }
    $ratio = [Math]::Abs($Statistic)/[Math]::Sqrt($DegreesOfFreedom)
    # Keep log(x) when x itself underflows: with df=1 its square-root tail
    # can still be representable even though x and t*t are not.
    if ($ratio -gt 1.0) {
        $inverse = 1.0/$ratio
        $logX = -2.0*[Math]::Log($ratio) - [Math]::Log(1.0+$inverse*$inverse)
        $x = [Math]::Exp($logX)
    } else {
        $x = 1.0/(1.0+$ratio*$ratio)
        $logX = [Math]::Log($x)
    }
    $a = $DegreesOfFreedom/2.0
    $b = 0.5
    $complement = $x -gt (($a+1.0)/($a+$b+2.0))
    if ($complement) {
        $x = 1.0-$x; $a,$b = $b,$a
        if ($x -eq 0.0) { return 1.0 }
        $logX = [Math]::Log($x)
    }
    $logBeta = (Get-EvaluationLogGamma $a) + (Get-EvaluationLogGamma $b) - (Get-EvaluationLogGamma ($a+$b))
    $front = [Math]::Exp($a*$logX + $b*[Math]::Log(1.0-$x) - $logBeta)
    $probability = $front*(Get-EvaluationBetaFraction -A $a -B $b -X $x)/$a
    if ($complement) { $probability = 1.0-$probability }
    if (-not [double]::IsFinite($probability) -or $probability -lt -1e-12 -or $probability -gt (1.0+1e-12)) {
        throw 'Student t probability is outside its numerical domain.'
    }
    return [Math]::Max(0.0, [Math]::Min(1.0, $probability))
}
