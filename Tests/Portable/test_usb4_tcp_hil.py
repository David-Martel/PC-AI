"""USB4 TCP hardware-profile and release-evidence controls without network access."""

import copy
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


source = Path(__file__).resolve().parents[1] / 'Hardware' / 'usb4_tcp_hil.py'
spec = importlib.util.spec_from_file_location('usb4_tcp_hil', source)
hil = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hil)


def sample_profile():
    return {'schema': 'pcai.usb4-tcp-hil-profile/v1', 'edges': [{
        'id': 'bench-pair', 'tcp_port': 53001, 'min_gbps_each_way': 1.0,
        'client': {'host': 'workbench-a', 'management': 'test@management-a', 'ip': 'fe80::a',
                   'adapter_guid': 'd455da7b-3e9e-4532-bc86-0ea794e05150', 'mac': '02-00-00-00-00-01',
                   'iperf_path': 'C:/testing/iperf3.exe'},
        'server': {'host': 'workbench-b', 'ip': 'fe80::b',
                   'adapter_guid': '0ad95aab-90bf-4ab1-9bd3-3999379b62af', 'mac': '02-00-00-00-00-02',
                   'iperf_path': 'D:/testing/iperf3.exe'},
    }]}


def evidence(edge, side):
    local = edge[side]
    peer = edge['server' if side == 'client' else 'client']
    index = 42 if side == 'client' else 76
    sample = {'bytes': 750_000_000, 'seconds': 3.0, 'bits_per_second': 2e9}
    sockets = [{'local_host': local['ip'], 'remote_host': peer['ip']}] * 2
    return {'role': side if side == 'client' else 'listen', 'host': local['host'], 'index': index,
            'source': f"{local['ip']}%{index}", 'peer': f"{peer['ip']}%{index}", 'binary_sha256': 'a' * 64,
            'rx_delta': 800_000_000, 'tx_delta': 800_000_000,
            'result': {'start': {'connected': sockets}, 'end': {
                'sum_received': dict(sample), 'sum_received_bidir_reverse': dict(sample)}}}


class Usb4HilControls(unittest.TestCase):
    # Protects directional profiling. Detects sender samples, omitted and partial
    # intervals, and relabeling absent link evidence as zero. Needs synthetic
    # iperf JSON. Breadcrumb: Tests/Hardware/usb4_tcp_hil.py::describe_edge.
    def test_profile_uses_only_full_receiver_intervals_and_preserves_link_metadata(self):
        edge = hil.profile(sample_profile())[0]
        client, server = evidence(edge, 'client'), evidence(edge, 'server')

        def interval(rate, *, elapsed=1.0, sender=False, omitted=False):
            return {'seconds': elapsed, 'bits_per_second': rate * 1e9,
                    'sender': sender, 'omitted': omitted}

        client['result']['intervals'] = [{'sum_bidir_reverse': sample} for sample in (
            interval(3), interval(5), interval(30, elapsed=0.1), interval(90, sender=True),
            interval(77, omitted=True), interval(0))]
        server['result']['intervals'] = [{'sum': sample} for sample in (interval(2), interval(6))]
        client['link'] = {'tx_bps': 20_000_000_000, 'rx_bps': 20_000_000_000,
                          'driver_version': '10.0.2.3', 'randomize_identifiers': 'Enabled'}
        profile = hil.describe_edge(client, server)
        self.assertEqual(profile['client']['link'], client['link'])
        self.assertIsNone(profile['server']['link'])
        self.assertEqual(profile['client_to_server_receiver_intervals_gbps'], [2.0, 6.0])
        self.assertEqual(profile['server_to_client_receiver_intervals_gbps'], [3.0, 5.0, 0.0])
        del client['result']['intervals']
        self.assertIsNone(hil.describe_edge(client, server)['server_to_client_receiver_intervals_gbps'])

    # Protects machine-neutral enrollment. Detects inferred endpoints, unsafe SSH arguments,
    # non-link-local addresses and duplicate physical edges. Needs Python, no adapters.
    # Breadcrumb: Tests/Hardware/usb4_tcp_hil.py::profile and endpoint.
    def test_profile_checks_enrolled_roles_and_physical_identity(self):
        data = sample_profile()
        self.assertEqual(hil.profile(data)[0]['client']['host'], 'workbench-a')
        for key, value in [('management', '-oProxyCommand=x'), ('ip', 'fd00::a'), ('ip', 'fe80::a%9'),
                           ('host', 'workbench-b'), ('mac', '02-00-00-00-00-02')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                invalid = copy.deepcopy(data)
                invalid['edges'][0]['client'][key] = value
                hil.profile(invalid)
        data['edges'].append({**data['edges'][0], 'id': 'repeated'})
        with self.assertRaisesRegex(ValueError, 'Duplicate physical edge'):
            hil.profile(data)

    # Protects bilateral payload proof. Detects management address, incomplete sockets,
    # no counter progress, early termination, inconsistent byte totals and false ceilings.
    # Needs synthetic endpoint JSON. Breadcrumb: Tests/Hardware/usb4_tcp_hil.py::qualify.
    def test_qualification_requires_both_receivers_and_counters(self):
        edge = hil.profile(sample_profile())[0]
        a, b = evidence(edge, 'client'), evidence(edge, 'server')
        self.assertEqual(hil.qualify(edge, a, b), {
            'client_to_server_gbps': 2.0, 'server_to_client_gbps': 2.0, 'minimum_met': True})
        self.assertFalse(hil.qualify({**edge, 'min_gbps_each_way': 2.1}, a, b)['minimum_met'])
        for removed_from_client in False, True:
            with self.subTest(removed_from_client=removed_from_client):
                client, server = copy.deepcopy((a, b))
                receiver = client if removed_from_client else server
                key = 'sum_received_bidir_reverse' if removed_from_client else 'sum_received'
                del receiver['result']['end'][key]
                with self.assertRaisesRegex(ValueError, 'Missing USB4 receiver sample'):
                    hil.qualify(edge, client, server)
        for field, value in [('tx_delta', 100), ('source', 'fe80::a%7'), ('host', 'some-other-host')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                hil.qualify(edge, {**a, field: value}, b)
        for key, value in [('seconds', 0.1), ('bits_per_second', 9e9), ('bytes', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                bad = copy.deepcopy(a)
                bad['result']['end']['sum_received_bidir_reverse'][key] = value
                hil.qualify(edge, bad, b)
        wrong = copy.deepcopy(b)
        wrong['result']['start']['connected'][0]['remote_host'] = '192.0.2.2'
        with self.assertRaisesRegex(ValueError, 'not bound'):
            hil.qualify(edge, a, wrong)

    # Protects release evidence binding. Detects wrong Git head, profile, evidence digest,
    # stale samples, missing topology and claimed threshold success without receiver proof.
    # Needs synthetic receipts and UTC timestamps. Breadcrumb: usb4_tcp_hil.py::verify_receipt.
    def test_release_verification_fails_closed(self):
        edge = hil.profile(sample_profile())[0]
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        receipt = {'schema': 'pcai.usb4-tcp-hil/v1', 'state': 'Passed', 'source_revision': 'c' * 40,
                   'source_sha256': {'runner': 'a' * 64}, 'profile_sha256': 'b' * 64,
                   'started_utc': (now - timedelta(minutes=1)).isoformat(), 'completed_utc': now.isoformat(),
                   'edges': [{'id': edge['id'], 'state': 'Passed', 'client': evidence(edge, 'client'),
                              'server': evidence(edge, 'server')}]}
        args = ('c' * 40, {'runner': 'a' * 64}, 'b' * 64, [edge], now, timedelta(minutes=5))
        verified, = hil.verify_receipt(receipt, *args)
        self.assertEqual(verified['id'], edge['id'])
        self.assertEqual(verified['threshold_gbps_each_way'], 1.0)
        self.assertEqual(verified['rates']['server_to_client_gbps'], 2.0)
        for field, value in [('source_revision', '1' * 40), ('source_sha256', {}), ('profile_sha256', '1' * 64),
                             ('started_utc', (now - timedelta(minutes=6)).isoformat()), ('edges', [])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                hil.verify_receipt({**receipt, field: value}, *args)
        false_green = copy.deepcopy(receipt)
        false_green['edges'][0]['client']['rx_delta'] = 0
        with self.assertRaises(ValueError):
            hil.verify_receipt(false_green, *args)

    # Protects harness admission. Detects starting traffic after listener failure or
    # applying temporary hardware policy to the client. Detects lost receiver admission
    # errors. Needs a process mock, no network.
    # Breadcrumb: Tests/Hardware/usb4_tcp_hil.py::test_edge.
    def test_listener_must_be_ready_and_only_receiver_can_open_firewall(self):
        edge = hil.profile(sample_profile())[0]
        with patch.object(hil, 'run_endpoint', side_effect=lambda _, __, role, ___: {'ready': role != 'ready'}) as run:
            self.assertEqual(hil.test_edge({}, edge, True)['state'], 'Failed')
            self.assertNotIn('client', [call.args[2] for call in run.call_args_list])
            self.assertIn(('listen', True), [(call.args[2], call.args[3]) for call in run.call_args_list])
        values = {'client': evidence(edge, 'client'), 'listen': evidence(edge, 'server'), 'ready': {'ready': True}}
        with patch.object(hil, 'run_endpoint', side_effect=lambda _, __, role, ___: values[role]) as run:
            self.assertEqual(hil.test_edge({}, edge, True)['state'], 'Passed')
            self.assertIn(('client', False), [(call.args[2], call.args[3]) for call in run.call_args_list])

        def deny_listener(_, __, role, ___):
            if role == 'listen':
                raise ValueError('abandoned firewall rule fixture-name')
            return {'ready': False}

        with patch.object(hil, 'run_endpoint', side_effect=deny_listener) as run:
            result = hil.test_edge({}, edge, True)
            self.assertEqual(result['state'], 'Failed')
            self.assertIn('fixture-name', result['error'])
            self.assertNotIn('client', [call.args[2] for call in run.call_args_list])

    # Protects: the remote transport gives the owned listener time to exit and clean up.
    # Detects: the outer timeout cutting off the remote job or leaving too little time for cleanup.
    # Needs: PowerShell endpoint source, a synthetic profile and a subprocess mock. No network.
    # Breadcrumb: Tests/Hardware/usb4_tcp_hil.py::run_endpoint and Invoke-PcaiUsb4TcpEndpoint.ps1::$processTimeout.
    def test_remote_listener_deadline_leaves_time_for_owned_cleanup(self):
        endpoint_source = (source.parents[2] / hil.ENDPOINT).read_bytes()
        job_graces = re.findall(r'^\s*\$processTimeout = \$seconds \+ (\d+)\s*$',
                                endpoint_source.decode('utf-8-sig'), re.MULTILINE)
        self.assertEqual(len(job_graces), 2, 'Find both PowerShell role budgets to check the outer deadline.')
        longest_job_grace = max(map(int, job_graces))
        artifacts = {hil.ENDPOINT: endpoint_source, hil.PROCESS_RUNNER: b'fixture runner source'}
        data = sample_profile()
        data['edges'][0]['server']['management'] = 'test@management-b'
        response = subprocess.CompletedProcess([], 0, stdout=b'{}', stderr=b'')
        for duration in (2, 15):
            with self.subTest(duration=duration), patch.object(hil.subprocess, 'run', return_value=response) as run:
                data['edges'][0]['seconds'] = duration
                edge, = hil.profile(data)
                self.assertEqual(hil.run_endpoint(artifacts, edge, 'listen', True), {})
                run.assert_called_once()
                command = run.call_args.args[0]
                self.assertEqual(command[0], 'ssh')
                self.assertIn('test@management-b', command)
                connect_option = next((arg for arg in command if arg.startswith('ConnectTimeout=')), '')
                self.assertRegex(connect_option, r'^ConnectTimeout=[1-9]\d*$')
                connect_timeout = int(connect_option.split('=', 1)[1])
                outer_timeout = run.call_args.kwargs['timeout']
                self.assertIsInstance(outer_timeout, int)
                # Reserve 30 s for remote PowerShell setup, CIM calls and firewall cleanup after the owned job.
                self.assertGreaterEqual(outer_timeout, duration + longest_job_grace + connect_timeout + 30)

    # Protects planning as a read-only diagnostic. Detects network execution,
    # output writes or a false qualified status. Needs a temporary profile only.
    # Breadcrumb: Tests/Hardware/usb4_tcp_hil.py::main dry-run branch.
    def test_dry_run_only_reports_planned_edges(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            path.write_text(json.dumps(sample_profile()), encoding='utf-8')
            receipt = path.with_name('receipt.json')
            args = ['usb4_tcp_hil.py', '--profile', str(path), '--output', str(receipt), '--dry-run']
            stdout = io.StringIO()
            with (patch.object(hil.sys, 'argv', args), patch.object(hil, 'load_sources') as load,
                  patch.object(hil, 'run_endpoint') as run, redirect_stdout(stdout)):
                self.assertEqual(hil.main(), 0)
            load.assert_not_called()
            run.assert_not_called()
            self.assertFalse(receipt.exists())
            self.assertEqual(json.loads(stdout.getvalue()), {'state': 'Planned', 'edges': ['bench-pair']})


if __name__ == '__main__':
    unittest.main()
