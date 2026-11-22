"""Run source-bound USB4 TCP duplex checks on explicitly enrolled Windows peers."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
import json
import math
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = 'Tests/Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1'
PROCESS_RUNNER = 'Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1'
FILES = (Path(__file__).resolve().relative_to(ROOT).as_posix(), ENDPOINT, PROCESS_RUNNER)
SCRIPT_LOADER = r'''
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try {
    $payload = [Console]::In.ReadToEnd() | ConvertFrom-Json -AsHashtable
    & ([scriptblock]::Create($payload.endpoint)) -Request $payload.request -ProcessRunnerSource $payload.runner
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
'''
MANAGEMENT = re.compile(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]{0,126}\Z')
HOST = re.compile(r'[A-Za-z0-9][A-Za-z0-9-]{0,62}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum, maximum, field):
    require(type(value) is int and minimum <= value <= maximum, f'{field} must be an integer in {minimum}..{maximum}')
    return value


def number(value, field):
    require(type(value) in (int, float) and 0 < value <= 1e18 and math.isfinite(value), f'{field} must be a finite positive number')
    return value


def endpoint(value):
    require(isinstance(value, dict), 'Endpoint must be an object')
    host, management = value['host'], value.get('management')
    require(isinstance(host, str) and HOST.fullmatch(host), 'Endpoint host must be its Windows computer name')
    require(management is None or (isinstance(management, str) and MANAGEMENT.fullmatch(management)),
            'Invalid SSH management target')
    address = ipaddress.IPv6Address(value['ip'])
    require(address.is_link_local and address.scope_id is None and str(address) == value['ip'],
            'Endpoint requires a canonical link-local IPv6 address without a scope')
    adapter = uuid.UUID(value['adapter_guid'])
    require(adapter.int != 0 and str(adapter) == value['adapter_guid'], 'Endpoint requires a canonical adapter GUID')
    mac = value['mac']
    require(isinstance(mac, str) and re.fullmatch(r'(?:[0-9A-F]{2}-){5}[0-9A-F]{2}', mac)
            and not (int(mac[:2], 16) & 1) and mac != '00-00-00-00-00-00', 'Invalid unicast adapter MAC')
    path = value['iperf_path']
    exe = PureWindowsPath(path)
    require(re.match(r'^[A-Za-z]:[\\/]', path) and exe.name.lower() == 'iperf3.exe',
            'Endpoint requires an absolute Windows iperf3.exe path')
    return dict(host=host, management=management, adapter_guid=str(adapter), mac=mac, ip=str(address), iperf_path=path)


def profile(data):
    require(isinstance(data, dict) and data.get('schema') == 'pcai.usb4-tcp-hil-profile/v1', 'Unsupported profile schema')
    definitions = data.get('edges')
    require(isinstance(definitions, list) and 1 <= len(definitions) <= 16, 'Profile must contain 1..16 edges')
    edges = {}
    pairs = set()
    for item in definitions:
        require(isinstance(item, dict), 'Edge must be an object')
        name = item['id']
        require(isinstance(name, str) and HOST.fullmatch(name) and name.casefold() not in edges, 'Invalid or duplicate edge ID')
        left, right = endpoint(item['client']), endpoint(item['server'])
        require(left['host'].casefold() != right['host'].casefold(), 'Endpoints must be distinct machines')
        for key in ('ip', 'mac', 'adapter_guid'):
            require(left[key] != right[key], f'Endpoints must have distinct {key} values')
        require(left['management'] is not None or right['management'] is not None, 'Both endpoints cannot be local')
        pair = frozenset((ep['host'].casefold(), ep['adapter_guid']) for ep in (left, right))
        require(pair not in pairs, 'Duplicate physical edge')
        pairs.add(pair)
        edges[name.casefold()] = {
            'id': name, 'client': left, 'server': right,
            'seconds': integer(item.get('seconds', 3), 2, 15, 'seconds'),
            'streams': integer(item.get('streams', 1), 1, 8, 'streams'),
            'tcp_port': integer(item['tcp_port'], 1024, 65535, 'tcp_port'),
            'min_gbps_each_way': number(item['min_gbps_each_way'], 'min_gbps_each_way'),
        }
    return list(edges.values())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_sources():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=10).strip()
    artifacts = {}
    for path in FILES:
        committed = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT, timeout=10)
        actual = (ROOT / path).read_bytes().replace(b'\r\n', b'\n')
        require(committed == actual, f'Running file differs from {revision}: {path}')
        artifacts[path] = committed
    return revision, artifacts


def run_endpoint(artifacts, edge, role, allow_firewall):
    local, peer = ('client', 'server') if role == 'client' else ('server', 'client')
    destination = edge[local]['management']
    prefix = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=6', destination] if destination else []
    loader = base64.b64encode(SCRIPT_LOADER.encode('utf-16-le')).decode('ascii')
    request = {k: edge[k] for k in ('tcp_port', 'seconds', 'streams')}
    request.update(role=role, local=edge[local], peer=edge[peer], allow_temporary_firewall=allow_firewall)
    payload = json.dumps({'endpoint': artifacts[ENDPOINT].decode('utf-8-sig'),
                          'runner': artifacts[PROCESS_RUNNER].decode('utf-8-sig'), 'request': request})
    try:
        response = subprocess.run(prefix + ['pwsh', '-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', loader],
                                  input=payload.encode('utf-8'), capture_output=True, timeout=edge['seconds'] + 48)
    except subprocess.TimeoutExpired:
        raise ValueError(f'{role} endpoint did not finish within its outer deadline') from None
    if response.returncode:
        # PowerShell may prepend CLIXML progress; preserve useful bounded diagnostics.
        output = response.stderr.decode('utf-8', errors='replace').strip()
        raise ValueError(f'{role} endpoint exited {response.returncode}: {output[-750:]}')
    require(0 < len(response.stdout) <= 1_048_576, f'{role} endpoint returned no result or oversized JSON')
    return json.loads(response.stdout)


def measure(edge, side, evidence):
    require(evidence['role'] == ('listen' if side == 'server' else side), f'{side} role mismatch')
    require(evidence['host'].casefold() == edge[side]['host'].casefold(), f'{side} identity mismatch')
    remote = edge['server' if side == 'client' else 'client']
    index = integer(evidence['index'], 1, 2**31 - 1, 'adapter index')
    require(evidence['source'] == f"{edge[side]['ip']}%{index}" and evidence['peer'] == f"{remote['ip']}%{index}",
            f'{side} expected source and peer scopes are missing')
    require(re.fullmatch(r'[a-fA-F0-9]{64}', evidence['binary_sha256']), f'{side} binary digest missing')
    result = evidence['result']
    require(isinstance(result, dict) and 'error' not in result, f'{side} iperf3 reported an error')
    sockets = result['start']['connected']
    require(isinstance(sockets, list) and len(sockets) == edge['streams'] * 2, f'{side} did not report all duplex sockets')
    for connection in sockets:
        require(connection['local_host'].split('%')[0] == edge[side]['ip'] and
                connection['remote_host'].split('%')[0] == remote['ip'], f'{side} socket was not bound to the declared peer link')
    return result['end']


def qualify(edge, client, server):
    client_end = measure(edge, 'client', client)
    server_end = measure(edge, 'server', server)
    directions = [(server_end['sum_received'], client, server),
                  (client_end['sum_received_bidir_reverse'], server, client)]
    rates = []
    for receiver, source, destination in directions:
        elapsed = number(receiver['seconds'], 'receiver seconds')
        require(elapsed >= edge['seconds'] * 0.8, 'Receiver sample stopped early')
        count = integer(receiver['bytes'], 1, 2**63 - 1, 'receiver bytes')
        rate = number(receiver['bits_per_second'], 'receiver rate')
        require(0.99 <= (count * 8 / elapsed) / rate <= 1.01, 'Inconsistent receiver rate and byte totals')
        integer(source['tx_delta'], count, 2**64 - 1, 'USB4 sender counter')
        integer(destination['rx_delta'], count, 2**64 - 1, 'USB4 receiver counter')
        rates.append(rate / 1e9)
    return dict(client_to_server_gbps=rates[0], server_to_client_gbps=rates[1],
                minimum_met=min(rates) >= edge['min_gbps_each_way'])


def test_edge(artifacts, edge, allow_firewall):
    test = {'id': edge['id'], 'state': 'Failed', 'threshold_gbps_each_way': edge['min_gbps_each_way']}
    try:
        with ThreadPoolExecutor(max_workers=1) as worker:
            listening = worker.submit(run_endpoint, artifacts, edge, 'listen', allow_firewall)
            ready = run_endpoint(artifacts, edge, 'ready', False)
            require(ready.get('ready') is True, 'USB4 benchmark listener was not ready on its exact address')
            client = run_endpoint(artifacts, edge, 'client', False)
            server = listening.result()
        test.update(client=client, server=server)
        test['rates'] = qualify(edge, client, server)
        test['state'] = 'Passed' if test['rates']['minimum_met'] else 'Failed'
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        test['error'] = str(error)
    return test


def verify_receipt(receipt, revision, source_digests, profile_digest, edges, now, max_age):
    require(receipt.get('schema') == 'pcai.usb4-tcp-hil/v1' and receipt.get('state') == 'Passed',
            'Receipt must be a passing USB4 TCP HIL run')
    require(receipt.get('source_revision') == revision and receipt.get('source_sha256') == source_digests,
            'Receipt does not bind the expected revision and test sources')
    require(receipt.get('profile_sha256') == profile_digest, 'Receipt does not bind the given profile')
    started = datetime.fromisoformat(receipt['started_utc'])
    completed = datetime.fromisoformat(receipt['completed_utc'])
    require(started.tzinfo is not None and completed.tzinfo is not None and
            now - max_age <= started <= completed <= now, 'Receipt timing is stale or invalid')
    tests = receipt['edges']
    require(isinstance(tests, list) and [test['id'] for test in tests] == [edge['id'] for edge in edges],
            'Receipt does not cover every declared edge in order')
    for edge, test in zip(edges, tests):
        require(test['state'] == 'Passed' and qualify(edge, test['client'], test['server'])['minimum_met'],
                f"Receipt fails the threshold on {edge['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path, help='Local JSON inventory of USB4 edges to test')
    parser.add_argument('--output', type=Path, help='New JSON receipt file, required for a live run')
    parser.add_argument('--dry-run', action='store_true', help='Validate the local profile without contacting endpoints or writing files')
    parser.add_argument('--allow-temporary-firewall', action='store_true', help='Allow an exact inbound TCP rule during each receiver trial')
    parser.add_argument('--verify-receipt', type=Path, help='Validate a trusted receipt offline')
    parser.add_argument('--expected-revision', help='Exact GitHub commit required for receipt verification')
    parser.add_argument('--max-age-minutes', type=int, default=1440, help='Maximum acceptable receipt age for offline verification')
    args = parser.parse_args()
    try:
        with args.profile.open('rb') as file:
            content = file.read(131072 + 1)
        require(len(content) <= 131072, 'Profile exceeds 128 KiB')
        edges = profile(json.loads(content))
        if args.verify_receipt:
            require(not args.dry_run and args.output is None and not args.allow_temporary_firewall,
                    'Receipt verification cannot start a live or dry-run test')
            require(args.expected_revision and re.fullmatch(r'[0-9a-f]{40}', args.expected_revision),
                    'Receipt verification requires an exact --expected-revision')
            max_age = integer(args.max_age_minutes, 1, 10080, 'max-age-minutes')
            digests = {p: sha256(subprocess.check_output(['git', 'show', f'{args.expected_revision}:{p}'],
                                                       cwd=ROOT, timeout=10)) for p in FILES}
            with args.verify_receipt.open('rb') as file:
                record = file.read(32 * 1024 * 1024 + 1)
            require(len(record) <= 32 * 1024 * 1024, 'Receipt is oversized')
            verify_receipt(json.loads(record), args.expected_revision, digests, sha256(content), edges,
                           datetime.now(timezone.utc), timedelta(minutes=max_age))
            print(json.dumps({'state': 'Verified', 'source_revision': args.expected_revision}))
            return 0
        require(args.expected_revision is None, '--expected-revision requires --verify-receipt')
        if args.dry_run:
            print(json.dumps({'state': 'Planned', 'edges': [edge['id'] for edge in edges]}))
            return 0
        require(args.output is not None, '--output is required for a live run')
        revision, artifacts = load_sources()
        # Reserve the evidence path before contacting any peer. An existing receipt is never overwritten.
        with args.output.open('x', encoding='utf-8', newline='\n') as output:
            started = datetime.now(timezone.utc).isoformat()
            tests = [test_edge(artifacts, edge, args.allow_temporary_firewall) for edge in edges]
            passed = all(test['state'] == 'Passed' for test in tests)
            receipt = {'schema': 'pcai.usb4-tcp-hil/v1', 'source_revision': revision,
                       'source_sha256': {p: sha256(data) for p, data in artifacts.items()},
                       'profile_sha256': sha256(content), 'started_utc': started,
                       'completed_utc': datetime.now(timezone.utc).isoformat(),
                       'state': 'Passed' if passed else 'Failed', 'edges': tests}
            json.dump(receipt, output, indent=2, allow_nan=False)
            output.write('\n')
        print(json.dumps({'state': receipt['state'], 'output': str(args.output),
                          'edges': [{'id': test['id'], 'state': test['state'], 'rates': test.get('rates'),
                                     'error': test.get('error')} for test in tests]}))
        return 0 if passed else 1
    except (OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
