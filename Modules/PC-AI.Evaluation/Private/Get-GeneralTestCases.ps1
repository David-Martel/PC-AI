function Get-GeneralTestCases {
    return @(
        [EvaluationTestCase]@{
            Id = "gen-001"
            Category = "factual"
            Prompt = "What is the capital of France?"
            ExpectedOutput = "Paris"
            Tags = @('factual', 'geography')
        }
        [EvaluationTestCase]@{
            Id = "gen-002"
            Category = "reasoning"
            Prompt = "If a car travels at 60 mph for 2.5 hours, how far does it travel?"
            ExpectedOutput = "150 miles"
            Tags = @('math', 'reasoning')
        }
        [EvaluationTestCase]@{
            Id = "gen-003"
            Category = "coding"
            Prompt = "Write a Python function to check if a number is prime."
            ExpectedOutput = "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))"
            Tags = @('coding', 'python')
        }
    )
}
