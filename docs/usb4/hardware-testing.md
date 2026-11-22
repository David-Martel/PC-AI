# USB4 TCP hardware testing

Use the [HIL coordinator](../../Tests/Hardware/usb4_tcp_hil.py) from a trusted PC-AI checkout to qualify Windows USB4NET cables for simultaneous bidirectional TCP. CI runs the [portable controls](../../Tests/Portable/test_usb4_tcp_hil.py) on Windows and Linux. The physical run needs Python 3.10+, Git, OpenSSH for remote management, and PowerShell 7 and iperf3 3.11+ on each endpoint.

## Describe the cables

Keep the site profile and run receipts on an access-controlled local path. Use the actual Windows computer name, interface GUID, MAC and preferred IPv6 link-local address of each USB4NET adapter. `Get-NetAdapter -IncludeHidden`, `Get-NetIPAddress -AddressFamily IPv6`, and `Get-NetNeighbor -AddressFamily IPv6` provide these values. Supply a local iperf3 path on each endpoint and a free TCP port approved for the trial. The optional `management` field names a pinned SSH endpoint. Omit it when the coordinator is running on that endpoint. Management can run over an existing LAN connection. TCP measurements bind to the USB4 link-local addresses and use the live Windows interface index to scope them.

```json
{
  "schema": "pcai.usb4-tcp-hil-profile/v1",
  "edges": [{
    "id": "link-a",
    "client": {
      "host": "peer-a", "management": "operator@peer-a",
      "ip": "fe80::1", "adapter_guid": "11111111-1111-4111-8111-111111111111",
      "mac": "02-00-00-00-00-01", "iperf_path": "C:/Tools/iperf3.exe"
    },
    "server": {
      "host": "peer-b", "management": "operator@peer-b",
      "ip": "fe80::2", "adapter_guid": "22222222-2222-4222-8222-222222222222",
      "mac": "02-00-00-00-00-02", "iperf_path": "C:/Tools/iperf3.exe"
    },
    "tcp_port": 49718, "min_gbps_each_way": 1, "seconds": 3, "streams": 4
  }]
}
```

In a hub or mesh, describe each cable as a separate edge. Use a measured baseline for each edge's floor. The sample floor detects connectivity failure and is too low to qualify a 40 Gbps-class link for a throughput release. Run from a host with a stable management route and schedule the trial to avoid contention with active transfers.

## Run and review

From the checkout root on an authorized coordinator, use the exact reviewed Git revision and private file paths. These commands use PowerShell syntax.

```powershell
python Tests/Hardware/usb4_tcp_hil.py --profile $profile --dry-run
python Tests/Hardware/usb4_tcp_hil.py --profile $profile --output $receipt
python Tests/Hardware/usb4_tcp_hil.py --profile $profile --verify-receipt $receipt --expected-revision $reviewedCommit
```

If the receiver needs a firewall exception, use `--allow-temporary-firewall` with an appropriately privileged receiver login. The [Windows endpoint](../../Tests/Hardware/Invoke-PcaiUsb4TcpEndpoint.ps1) creates a rule scoped to the peer, adapter, executable and port, then removes it when the trial ends. Without that flag, existing host policy governs the connection. Each trial requires reciprocal neighbor identity, an on-link route, scoped sockets, positive counters on both USB4 adapters, and receiver throughput above the declared floor in each direction. Any incomplete edge fails the run. Exit codes are `0` for success, `1` for a failed edge, and `2` for an invalid input or receipt.

For release review, retain the trusted original receipt, profile, and source revision. Offline verification checks the revision and source hashes, profile, observation age, every edge, and both measured directions. Capture and distribute receipts through an authenticated channel. The receipt qualifies direct Windows USB4NET TCP for the recorded conditions. Record reconnect, multi-hop, authenticated file transfers and RDMA with separate trials. See [deployment and measurement](fabric-deployment.md) for the broader release sequence and [Microsoft's USB4 throughput method](https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/usb4-interdomain-connections#throughput-testing) for transport testing guidance.
