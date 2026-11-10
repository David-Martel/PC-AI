# Independent installer history review

The [bound receipt](usb4-history-install-review.json) records exact normalized-LF source, test and documentation hashes. All15 installer groups passed in a fresh profile-free Windows PowerShell process, including the actual scheduler/schema and file-ACL controls.

The review found a mixed-case bypass: the root allowlist admitted uppercase history while a case-sensitive selector skipped its validation. A negative control restoring only that comparison fails the unchanged new test with the expected missing-refusal error. The reviewed revision uses canonical case-sensitive admission.

Valid owned history remains read-only during upgrade admission. The existing marker, protected directory and file ACLs, canonical content-addressed leaves, bounded complete bytes, exact hashes and object/pending shape are required. Foreign recovery archives remain outside the installation root. No remaining blocker was found in this narrow delta.

This qualifies source and detecting fixture coverage. It does not qualify deployment, a real future upgrade, arbitrary privileged races or power-loss behavior, and does not rewrite the earlier publication/retirement review.
