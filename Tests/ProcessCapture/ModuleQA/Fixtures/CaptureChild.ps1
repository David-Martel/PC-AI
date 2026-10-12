[CmdletBinding(PositionalBinding=$false)]
param([switch]$Emit,[switch]$DryRun,[Alias('h')][switch]$Help,[Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments)
if($Help-or($null-ne$RemainingArguments-and$RemainingArguments-ccontains'--help')){'CaptureChild.ps1 -Emit emits two fixed public markers; default/DryRun performs no IO.';return}
if($DryRun-or-not$Emit){'NO_IO';return}
if($null-ne$RemainingArguments-and$RemainingArguments.Count){throw 'Unknown arguments'}
[Console]::Out.Write('PCAI_MODULE_CAPTURE_OK')
[Console]::Error.Write('PCAI_MODULE_CAPTURE_ERR')
exit 0
