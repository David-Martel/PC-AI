#Requires -Version 7.3
# The complete canonical functions call one actual owned loopback TCP responder.
# No provider, model, mock HTTP response or copied SSE parser is used.
BeforeAll {
    $script:StreamingHelperPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../Modules/PC-AI.LLM/Private/LLM-Helpers.ps1'))
    $tokens = $null
    $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($script:StreamingHelperPath, [ref]$tokens, [ref]$errors)
    if ($errors.Count -ne 0 -or @($ast.EndBlock.Statements).Count -ne 23 -or @($ast.EndBlock.Statements | Where-Object { $_ -isnot [Management.Automation.Language.FunctionDefinitionAst] }).Count -ne 0) { throw 'Expected only the 23 canonical function declarations.' }
    . $script:StreamingHelperPath
    $script:ModuleConfig = @{ DefaultTimeout = 7 }
    $script:StreamingOutcomes = [Collections.Generic.List[object]]::new()
    $script:PendingStreamingFixtures = [Collections.Generic.List[object]]::new()
    $global:PCAI_StreamingOutcomesR1 = $script:StreamingOutcomes
    $global:PCAI_PendingStreamingFixturesR1 = $script:PendingStreamingFixtures
    Add-Type -ErrorAction Stop -WarningAction Stop -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
namespace PcaiStreamingFixtureR1 {
    public sealed class Responder {
        private readonly TcpListener listener;
        private readonly CancellationTokenSource lifetime;
        private readonly Task operation;
        private Task callerCancellation;
        private TcpClient client;
        public string Url { get; }
        public string RequestPath { get; private set; }
        public string RequestBody { get; private set; }
        public bool Accepted { get; private set; }
        public bool HeadersSent { get; private set; }
        public bool TaskObserved { get; private set; }
        public bool SocketClosed { get; private set; }
        public bool ListenerStopped { get; private set; }
        public bool LifetimeDisposed { get; private set; }
        public bool CallerTaskObserved { get; private set; }
        public bool CallerCancelledAfterHeaders { get; private set; }
        public Exception FirstFailure { get; private set; }
        public List<Exception> SecondaryFailures { get; } = new List<Exception>();
        public Responder(string mode, bool chat) {
            lifetime = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            listener = new TcpListener(IPAddress.IPv6Loopback, 0);
            listener.Start(1);
            Url = "http://localhost:" + ((IPEndPoint)listener.LocalEndpoint).Port;
            operation = Run(mode, chat);
        }
        private async Task Run(string mode, bool chat) {
            CancellationToken token = lifetime.Token;
            try {
                client = await listener.AcceptTcpClientAsync(token).ConfigureAwait(false);
                Accepted = true;
                NetworkStream stream = client.GetStream();
                var header = new List<byte>();
                byte[] one = new byte[1];
                while (header.Count < 32768) {
                    int read = await stream.ReadAsync(one.AsMemory(), token).ConfigureAwait(false);
                    if (read == 0) throw new EndOfStreamException("Fixture request ended inside headers");
                    header.Add(one[0]);
                    int n = header.Count;
                    if (n >= 4 && header[n-4] == 13 && header[n-3] == 10 && header[n-2] == 13 && header[n-1] == 10) break;
                }
                string headers = Encoding.ASCII.GetString(header.ToArray());
                if (!headers.EndsWith("\r\n\r\n", StringComparison.Ordinal)) throw new InvalidDataException("Fixture header cap");
                RequestPath = headers.Split(' ')[1];
                int length = -1;
                foreach (string line in headers.Split(new[] {"\r\n"}, StringSplitOptions.None)) {
                    if (line.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase)) length = int.Parse(line.Substring(15).Trim());
                }
                if (length < 0 || length > 32768) throw new InvalidDataException("Fixture body cap");
                byte[] request = new byte[length];
                int received = 0;
                while (received < length) {
                    int read = await stream.ReadAsync(request.AsMemory(received), token).ConfigureAwait(false);
                    if (read == 0) throw new EndOfStreamException("Fixture request ended inside body");
                    received += read;
                }
                RequestBody = new UTF8Encoding(false, true).GetString(request);
                string first = chat ? "data: {\"choices\":[{\"delta\":{\"content\":\"hé\"}}]}" : "data: {\"choices\":[{\"text\":\"hé\"}]}";
                string second = chat ? "data: {\"choices\":[{\"delta\":{\"content\":\"λ\"}}]}\n\n" : "data: {\"choices\":[{\"text\":\"λ\"}]}\n\n";
                string body = first + "\n\n" + second + "data: [DONE]\n\n";
                if (mode == "Eof") body = first + "\n";
                if (mode == "Unicode") body = ": inert comment\n\nevent: message\n" + body + "data: {\"choices\":[{\"text\":\"forbidden-after-done\"}]}\n\n";
                byte[] bodyBytes = Encoding.UTF8.GetBytes(body);
                if (mode == "Headers") await Task.Delay(3000, token).ConfigureAwait(false);
                string status = mode == "HttpError" ? "503 Inert Unavailable" : "200 OK";
                if (mode == "HttpError") bodyBytes = Array.Empty<byte>();
                byte[] responseHeader = Encoding.ASCII.GetBytes("HTTP/1.1 " + status + "\r\nContent-Type: text/event-stream\r\nContent-Length: " + bodyBytes.Length + "\r\nConnection: close\r\n\r\n");
                await stream.WriteAsync(responseHeader.AsMemory(), token).ConfigureAwait(false);
                HeadersSent = true;
                if (mode == "Body") {
                    byte[] prefix = Encoding.UTF8.GetBytes(first);
                    await stream.WriteAsync(prefix.AsMemory(), token).ConfigureAwait(false);
                    await Task.Delay(3000, token).ConfigureAwait(false);
                    await stream.WriteAsync(bodyBytes.AsMemory(prefix.Length), token).ConfigureAwait(false);
                } else if (mode == "SlowFrames") {
                    byte[] prefix = Encoding.UTF8.GetBytes(first + "\n\n");
                    await stream.WriteAsync(prefix.AsMemory(), token).ConfigureAwait(false);
                    await Task.Delay(700, token).ConfigureAwait(false);
                    byte[] middle = Encoding.UTF8.GetBytes(second);
                    await stream.WriteAsync(middle.AsMemory(), token).ConfigureAwait(false);
                    await Task.Delay(700, token).ConfigureAwait(false);
                    await stream.WriteAsync(Encoding.UTF8.GetBytes("data: [DONE]\n\n").AsMemory(), token).ConfigureAwait(false);
                } else if (mode == "Unicode") {
                    int split = Array.IndexOf(bodyBytes, (byte)0xC3) + 1;
                    await stream.WriteAsync(bodyBytes.AsMemory(0, split), token).ConfigureAwait(false);
                    await Task.Delay(10, token).ConfigureAwait(false);
                    await stream.WriteAsync(bodyBytes.AsMemory(split), token).ConfigureAwait(false);
                } else {
                    await stream.WriteAsync(bodyBytes.AsMemory(), token).ConfigureAwait(false);
                }
            } catch (Exception error) {
                if (!lifetime.IsCancellationRequested) FirstFailure = error;
            } finally {
                if (client != null) {
                    try { client.Dispose(); SocketClosed = true; } catch (Exception error) { Record(error); }
                } else { SocketClosed = true; }
            }
        }
        public void CancelCallerAfterHeaders(CancellationTokenSource caller) {
            if (callerCancellation != null) throw new InvalidOperationException("Only one caller cancellation task");
            callerCancellation = CancelCaller(caller);
        }
        private async Task CancelCaller(CancellationTokenSource caller) {
            try {
                while (!HeadersSent) await Task.Delay(5, lifetime.Token).ConfigureAwait(false);
                await Task.Delay(250, lifetime.Token).ConfigureAwait(false);
                caller.Cancel();
                CallerCancelledAfterHeaders = true;
            } catch (Exception error) { if (!lifetime.IsCancellationRequested) Record(error); }
        }
        private void Record(Exception error) { if (FirstFailure == null) FirstFailure = error; else SecondaryFailures.Add(error); }
        public bool Finish() {
            try { lifetime.Cancel(); } catch (Exception error) { Record(error); }
            try { listener.Stop(); ListenerStopped = true; } catch (Exception error) { Record(error); }
            try { if (client != null) client.Dispose(); } catch (Exception error) { Record(error); }
            try { TaskObserved = operation.Wait(2000); } catch (Exception error) { Record(error); }
            try { CallerTaskObserved = callerCancellation == null || callerCancellation.Wait(2000); } catch (Exception error) { Record(error); }
            if (TaskObserved && CallerTaskObserved) {
                try { lifetime.Dispose(); LifetimeDisposed = true; } catch (Exception error) { Record(error); }
            }
            return TaskObserved && CallerTaskObserved && SocketClosed && ListenerStopped && LifetimeDisposed;
        }
    }
}
'@
    function Invoke-StreamingFixture {
        param([ValidateSet('Chat','Completion')][string]$Kind, [string]$Mode, [int]$TimeoutSeconds = 1, [Threading.CancellationToken]$CancellationToken = [Threading.CancellationToken]::None, [switch]$PassCancellation, [Threading.CancellationTokenSource]$CallerCancellationSource)
        $fixture = [PcaiStreamingFixtureR1.Responder]::new($Mode, $Kind -eq 'Chat')
        if ($null -ne $CallerCancellationSource) { $fixture.CancelCallerAfterHeaders($CallerCancellationSource) }
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $caught = $null
        $value = $null
        $closed = $false
        $cleanupErrors = [Collections.Generic.List[object]]::new()
        try {
            $arguments = @{ Model = 'inert-model'; ApiUrl = $fixture.Url; Temperature = 0.25; MaxTokens = 19; TimeoutSeconds = $TimeoutSeconds; ErrorAction = 'Stop' }
            if ($PassCancellation) { $arguments.CancellationToken = $CancellationToken }
            if ($Kind -eq 'Chat') { $value = Invoke-OpenAIChatStream -Messages @(@{role='user';content='inert request'}) @arguments }
            else { $value = Invoke-OpenAICompletionStream -Prompt 'inert request' @arguments }
        } catch { $caught = $_ } finally {
            $watch.Stop()
            try { $closed = $fixture.Finish() } catch { $cleanupErrors.Add($_) }
            if (-not $closed) { $script:PendingStreamingFixtures.Add($fixture) }
        }
        $outcome = [pscustomobject]@{ Kind=$Kind;Mode=$Mode;Value=$value;ErrorRecord=$caught;ElapsedMs=$watch.ElapsedMilliseconds;Fixture=$fixture;FixtureClosed=$closed;CleanupErrors=@($cleanupErrors.ToArray()) }
        $script:StreamingOutcomes.Add($outcome)
        return $outcome
    }
    function Test-CancellationCause {
        param([Management.Automation.ErrorRecord]$Record)
        if ($null -eq $Record) { return $false }
        $errorValue = $Record.Exception
        while ($null -ne $errorValue) {
            if ($errorValue -is [OperationCanceledException]) { return $true }
            $errorValue = $errorValue.InnerException
        }
        return $false
    }
    function Assert-FixtureClosed {
        param($Outcome)
        $Outcome.FixtureClosed | Should -BeTrue
        $Outcome.Fixture.TaskObserved | Should -BeTrue
        $Outcome.Fixture.CallerTaskObserved | Should -BeTrue
        $Outcome.Fixture.SocketClosed | Should -BeTrue
        $Outcome.Fixture.ListenerStopped | Should -BeTrue
        $Outcome.Fixture.LifetimeDisposed | Should -BeTrue
        $Outcome.Fixture.FirstFailure | Should -BeNullOrEmpty
        $Outcome.Fixture.SecondaryFailures.Count | Should -Be 0
        $Outcome.CleanupErrors.Count | Should -Be 0
        $script:PendingStreamingFixtures.Count | Should -Be 0
    }
}

Describe 'Canonical SSE end-to-end deadlines' -Tag 'Unit', 'LLM', 'Portable', 'PredecessorDetecting' {
    # Protects timeout meaning after headers; detects the actual blocking body read.
    It 'bounds chat header-first partial-line reads by the requested deadline' {
        $actual = Invoke-StreamingFixture -Kind Chat -Mode Body
        Assert-FixtureClosed $actual
        $actual.Fixture.HeadersSent | Should -BeTrue
        (Test-CancellationCause $actual.ErrorRecord) | Should -BeTrue
        $actual.ElapsedMs | Should -BeLessThan 2500
        $actual.Value | Should -BeNullOrEmpty
    }
    It 'bounds completion header-first partial-line reads by the requested deadline' {
        $actual = Invoke-StreamingFixture -Kind Completion -Mode Body
        Assert-FixtureClosed $actual
        $actual.Fixture.HeadersSent | Should -BeTrue
        (Test-CancellationCause $actual.ErrorRecord) | Should -BeTrue
        $actual.ElapsedMs | Should -BeLessThan 2500
        $actual.Value | Should -BeNullOrEmpty
    }
    # Protects one request deadline, rather than resetting it for every received line.
    It 'does not renew the chat deadline when earlier frames make progress' {
        $actual = Invoke-StreamingFixture -Kind Chat -Mode SlowFrames
        Assert-FixtureClosed $actual
        (Test-CancellationCause $actual.ErrorRecord) | Should -BeTrue
        $actual.ElapsedMs | Should -BeLessThan 2500
        $actual.Value | Should -BeNullOrEmpty
    }
    It 'retains the existing header deadline for completion requests' {
        $actual = Invoke-StreamingFixture -Kind Completion -Mode Headers
        Assert-FixtureClosed $actual
        (Test-CancellationCause $actual.ErrorRecord) | Should -BeTrue
        $actual.ElapsedMs | Should -BeLessThan 2500
        $actual.Fixture.HeadersSent | Should -BeFalse
    }
    # Protects external cancellation and its original cancellation cause.
    It 'cancels chat body reads through the supplied caller token' {
        $caller = [Threading.CancellationTokenSource]::new()
        try {
            $actual = Invoke-StreamingFixture -Kind Chat -Mode Body -TimeoutSeconds 4 -CancellationToken $caller.Token -PassCancellation -CallerCancellationSource $caller
            Assert-FixtureClosed $actual
            $actual.Fixture.Accepted | Should -BeTrue
            $actual.Fixture.HeadersSent | Should -BeTrue
            $actual.Fixture.CallerCancelledAfterHeaders | Should -BeTrue
            (Test-CancellationCause $actual.ErrorRecord) | Should -BeTrue
            $actual.ElapsedMs | Should -BeLessThan 2000
            $actual.Value | Should -BeNullOrEmpty
        } finally { $caller.Dispose() }
    }
    It 'cancels completion body reads through the supplied caller token' {
        $caller = [Threading.CancellationTokenSource]::new()
        try {
            $actual = Invoke-StreamingFixture -Kind Completion -Mode Body -TimeoutSeconds 4 -CancellationToken $caller.Token -PassCancellation -CallerCancellationSource $caller
            Assert-FixtureClosed $actual
            $actual.Fixture.Accepted | Should -BeTrue
            $actual.Fixture.HeadersSent | Should -BeTrue
            $actual.Fixture.CallerCancelledAfterHeaders | Should -BeTrue
            (Test-CancellationCause $actual.ErrorRecord) | Should -BeTrue
            $actual.ElapsedMs | Should -BeLessThan 2000
            $actual.Value | Should -BeNullOrEmpty
        } finally { $caller.Dispose() }
    }
    # Protects the caller-visible string and literal request fields without a mocked backend.
    It 'keeps chat Unicode frame assembly, DONE boundary and request options unchanged' {
        $actual = Invoke-StreamingFixture -Kind Chat -Mode Unicode -TimeoutSeconds 4
        Assert-FixtureClosed $actual
        $actual.ErrorRecord | Should -BeNullOrEmpty
        $actual.Value | Should -BeExactly 'héλ'
        $actual.Fixture.RequestPath | Should -BeExactly '/v1/chat/completions'
        $request = $actual.Fixture.RequestBody | ConvertFrom-Json
        $request.model | Should -BeExactly 'inert-model'
        $request.messages[0].content | Should -BeExactly 'inert request'
        $request.temperature | Should -Be 0.25
        $request.max_tokens | Should -Be 19
        $request.stream | Should -BeTrue
        (Get-Command Invoke-OpenAIChatStream).ScriptBlock.File | Should -Be $script:StreamingHelperPath
    }
    It 'keeps completion Unicode frame assembly, DONE boundary and request options unchanged' {
        $actual = Invoke-StreamingFixture -Kind Completion -Mode Unicode -TimeoutSeconds 4
        Assert-FixtureClosed $actual
        $actual.ErrorRecord | Should -BeNullOrEmpty
        $actual.Value | Should -BeExactly 'héλ'
        $actual.Fixture.RequestPath | Should -BeExactly '/v1/completions'
        $request = $actual.Fixture.RequestBody | ConvertFrom-Json
        $request.model | Should -BeExactly 'inert-model'
        $request.prompt | Should -BeExactly 'inert request'
        $request.temperature | Should -Be 0.25
        $request.max_tokens | Should -Be 19
        $request.stream | Should -BeTrue
    }
    It 'keeps the existing completion EOF behavior without requiring DONE' {
        $actual = Invoke-StreamingFixture -Kind Completion -Mode Eof -TimeoutSeconds 4
        Assert-FixtureClosed $actual
        $actual.ErrorRecord | Should -BeNullOrEmpty
        $actual.Value | Should -BeExactly 'hé'
    }
    It 'preserves the actual HTTP failure instead of returning partial success' {
        $actual = Invoke-StreamingFixture -Kind Chat -Mode HttpError -TimeoutSeconds 4
        Assert-FixtureClosed $actual
        $actual.ErrorRecord | Should -Not -BeNullOrEmpty
        $actual.ErrorRecord.Exception.ToString() | Should -Match '503'
        $actual.Value | Should -BeNullOrEmpty
    }
}
