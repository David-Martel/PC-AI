#Requires -Version 7.5
# Definitions only. Pins bind the complete actual production files and controls.
function Read-ProcessCaptureSources {
    param([string]$ModuleRoot,[string]$TestRoot,[string]$RegisterPath,[string]$RegisterSHA256)
    if($RegisterSHA256-cnotmatch'^[A-F0-9]{64}$'){throw 'Exact source register hash required'}
    $registerLeaf=Get-Item -LiteralPath $RegisterPath -Force
    if($registerLeaf-isnot[IO.FileInfo]-or$registerLeaf.Length-gt131072-or($registerLeaf.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Bounded ordinary register leaf required'}
    $registerAncestor=$registerLeaf
    for($depth=0;$null-ne$registerAncestor;$depth++){if($depth-ge32-or($registerAncestor.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Ordinary register ancestry required'};$registerAncestor=if($registerAncestor-is[IO.DirectoryInfo]){$registerAncestor.Parent}else{$registerAncestor.Directory}}
    $bytes=[IO.File]::ReadAllBytes($RegisterPath)
    if($bytes.Length-gt131072-or[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))-cne$RegisterSHA256){throw 'Bounded same-byte source register differs'}
    $register=[Text.UTF8Encoding]::new($false,$true).GetString($bytes)|ConvertFrom-Json -AsHashtable -DateKind String
    if($register.State-cne'MAINTAINED_SOURCE_BINDING_ONLY_NOT_RUNTIME_ACCEPTANCE'-or$register.Files-isnot[Array]-or$register.Files.Count-ne9-or$register.TestFiles-isnot[Array]-or$register.TestFiles.Count-ne5){throw 'Exact maintained source inventory required'}
    $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach($group in @(@{Root=$ModuleRoot;Rows=$register.Files},@{Root=$TestRoot;Rows=$register.TestFiles})){
        foreach($row in $group.Rows){
            if($row.Path-isnot[string]-or$row.Path-cnotmatch'^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$'-or$row.Path.Split('/')-ccontains'..'-or$row.SHA256-cnotmatch'^[A-F0-9]{64}$'-or$row.Bytes-isnot[long]-and$row.Bytes-isnot[int]){throw 'Malformed source pin'}
            $path=[IO.Path]::GetFullPath((Join-Path $group.Root $row.Path));if(-not$seen.Add($path)){throw 'Duplicate source leaf'}
            $item=Get-Item -LiteralPath $path -Force;$leaf=$item
            for($depth=0;$null-ne$item;$depth++){if($depth-ge32-or($item.Attributes-band[IO.FileAttributes]::ReparsePoint)){throw 'Ordinary source ancestry required'};$item=if($item-is[IO.DirectoryInfo]){$item.Parent}else{$item.Directory}}
            if($leaf-isnot[IO.FileInfo]-or$leaf.Length-ne$row.Bytes-or(Get-FileHash -LiteralPath $path).Hash-cne$row.SHA256){throw 'Actual source/fixture pin differs'}
        }
    }
    $register
}
