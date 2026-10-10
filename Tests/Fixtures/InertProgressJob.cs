using System;
using System.Management.Automation;

namespace PcaiProgressCustodyR1
{
    // Pure typed fixture: never registered with a job repository, runspace or process.
    public sealed class InertJob : Job
    {
        public InertJob(string name, bool running) : base("inert-fixture", name)
        {
            SetJobState(running ? JobState.Running : JobState.Completed);
        }
        public override string StatusMessage { get { return "inert"; } }
        public override bool HasMoreData { get { return false; } }
        public override string Location { get { return "fixture-only"; } }
        public override void StopJob() { throw new InvalidOperationException("Unmocked typed fixture StopJob forbidden"); }
    }
}
