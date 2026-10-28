# Captured evidence byte custody

Five historical captures contain mixed line endings in their committed blobs.
The general `*.txt text eol=crlf` rule in [Git attributes](../.gitattributes)
rewrites those bytes during Windows checkout. The files immediately appear
modified, including after `git stash`, although their differences consist only
of CRLF versus LF. This obstructs clean-checkout integration and changes the
bytes of retained evidence.

Exact-path `-text -eol` exceptions preserve the committed captures:

- [Intel Serial IO controllers](../Reports/input-stack-investigation-20260606/H3_intel_serialio_controllers.txt)
- [Process Lasso](../Reports/input-stack-investigation-20260606/H4_processlasso.txt)
- [I/O and CPU load](../Reports/input-stack-investigation-20260606/H5_io_cpu_load.txt)
- [Vendor services](../Reports/input-stack-investigation-20260606/H7_vendor_services.txt)
- [OneDrive step C](../Reports/onedrive-triage-20260509/27-stepC.txt)

The report blobs stay unchanged. The exceptions follow the existing Milly and
workload-evidence rules without changing normalization for other text files.

## Verification

Compare each capture's raw SHA256 with `git cat-file blob HEAD:<path>` output.
Use binary-safe output handling. `git check-attr text eol -- <path>` must report
both attributes as unset, and `git hash-object --path=<path> <path>` must equal
`git rev-parse HEAD:<path>`. A fresh Windows checkout must be clean, preserve the
same hashes and leave the five reports absent from the PR diff.

Before correcting an existing normalized working copy, preserve its bytes,
patch and hashes. Confirm raw comparison after CRLF-to-LF conversion finds no
other changes. Restore only these verified captures from their committed blobs;
do not normalize or reconstruct the evidence through text encoding APIs.

This verifies byte custody and checkout behavior. It establishes no new
workstation measurement or runtime qualification. See the historical
[fleet closeout](windows-fleet-closeout.md) for the separate acceptance boundaries.
