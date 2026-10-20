using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Win32.SafeHandles;

namespace Pcai.Storage
{
    /// <summary>Holds immutable local bytes through archive verification and identity-bound deletion.</summary>
    public sealed class PcaiArchiveSource : IDisposable
    {
        private const uint Read = 0x80000000, Write = 0x40000000, Delete = 0x00010000;
        private const uint ShareRead = 1, ShareWrite = 2, ShareDelete = 4;
        private const uint OpenExisting = 3, OpenAlways = 4, OpenReparse = 0x00200000, DirectoryFlag = 0x02000000, WriteThrough = 0x80000000;
        private const uint Reparse = 0x400, Offline = 0x1000, Recall = 0x400000;
        private readonly List<SafeFileHandle> directories = new List<SafeFileHandle>();
        private FileStream stream;
        private SafeFileHandle renameGuard;
        private ReceiptDirectoryCustody receiptDirectory;
        private ReceiptWriter receiptWriter;
        private bool receiptDurable, deleted, disposed;
        public string SourcePath { get; private set; }
        public string CanonicalGuidPath { get; private set; }
        public string VolumeSerialHex { get; private set; }
        public string FileId128 { get; private set; }
        public long Length { get; private set; }
        public string Md5 { get; private set; }
        public string Sha256 { get; private set; }
        public uint HardLinkCount { get; private set; }
        public uint HardLinkCountBeforeDeletion { get; private set; }
        public long AllocatedBytes { get; private set; }
        public string ReceiptGuidPath { get { return receiptWriter == null ? null : receiptWriter.CanonicalGuidPath; } }
        public string ReceiptVolumeSerial { get { return receiptWriter == null ? null : receiptWriter.VolumeSerialHex; } }
        public string ReceiptFileId { get { return receiptWriter == null ? null : receiptWriter.FileId128; } }

        [StructLayout(LayoutKind.Sequential)]
        private struct Identity { public ulong VolumeSerial, Low, High; }
        [StructLayout(LayoutKind.Sequential)]
        private struct Attributes { public uint Flags, Tag; }
        [StructLayout(LayoutKind.Sequential)]
        private struct Disposition { public byte DeleteFile; }
        [StructLayout(LayoutKind.Sequential)]
        private struct Standard { public long Allocation, EndOfFile; public uint Links; public byte Pending, Directory; }
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern SafeFileHandle CreateFileW(string path, uint access, uint share, IntPtr security, uint creation, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetFileInformationByHandleEx(SafeFileHandle handle, int kind, out Identity info, uint bytes);
        [DllImport("kernel32.dll", SetLastError = true, EntryPoint = "GetFileInformationByHandleEx")]
        private static extern bool GetAttributesByHandle(SafeFileHandle handle, int kind, out Attributes info, uint bytes);
        [DllImport("kernel32.dll", SetLastError = true, EntryPoint = "GetFileInformationByHandleEx")]
        private static extern bool GetStandardByHandle(SafeFileHandle handle, int kind, out Standard info, uint bytes);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern uint GetFinalPathNameByHandleW(SafeFileHandle handle, StringBuilder path, uint size, uint flags);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetFileInformationByHandle(SafeFileHandle handle, int kind, ref Disposition info, uint bytes);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool CreateDirectoryW(string path, IntPtr security);

        private static SafeFileHandle Open(string path, uint access, uint share, uint flags)
        {
            SafeFileHandle handle = CreateFileW(path, access, share, IntPtr.Zero, OpenExisting, flags, IntPtr.Zero);
            if (handle.IsInvalid) { int error = Marshal.GetLastWin32Error(); handle.Dispose(); throw new Win32Exception(error, "Cannot acquire archive source custody: " + path); }
            return handle;
        }

        private static Identity ReadIdentity(SafeFileHandle handle)
        {
            Identity value;
            if (!GetFileInformationByHandleEx(handle, 18, out value, 24)) throw new Win32Exception(Marshal.GetLastWin32Error(), "FileIdInfo is required for archive custody.");
            return value;
        }

        private static string IdText(Identity value)
        {
            byte[] bytes = new byte[16];
            Buffer.BlockCopy(BitConverter.GetBytes(value.Low), 0, bytes, 0, 8);
            Buffer.BlockCopy(BitConverter.GetBytes(value.High), 0, bytes, 8, 8);
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }

        private static void RejectReparse(SafeFileHandle handle)
        {
            Attributes value;
            if (!GetAttributesByHandle(handle, 9, out value, 8)) throw new Win32Exception(Marshal.GetLastWin32Error(), "Cannot verify source attributes.");
            if ((value.Flags & (Reparse | Offline | Recall)) != 0) throw new IOException("Cloud, offline and reparse archive sources are rejected.");
        }

        private static string GuidPath(SafeFileHandle handle)
        {
            StringBuilder buffer = new StringBuilder(1024);
            uint count = GetFinalPathNameByHandleW(handle, buffer, (uint)buffer.Capacity, 1);
            if (count >= buffer.Capacity && count < 32768) { buffer = new StringBuilder((int)count + 1); count = GetFinalPathNameByHandleW(handle, buffer, (uint)buffer.Capacity, 1); }
            if (count == 0 || count >= buffer.Capacity || !buffer.ToString().StartsWith(@"\\?\Volume{", StringComparison.OrdinalIgnoreCase)) throw new IOException("A local volume GUID path is required for archive custody.");
            return buffer.ToString();
        }

        public static string ValidateLocalPath(string path)
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) throw new PlatformNotSupportedException("Archive source custody requires Windows.");
            if (String.IsNullOrWhiteSpace(path)) throw new ArgumentException("Archive path is empty.");
            if (!Path.IsPathFullyQualified(path)) throw new IOException("Archive paths must be absolute.");
            // Windows normalization can silently strip a trailing dot or space.
            foreach (string raw in path.Replace('/', '\\').Split('\\'))
                if (raw.EndsWith(".", StringComparison.Ordinal) || raw.EndsWith(" ", StringComparison.Ordinal)) throw new IOException("Ambiguous archive path is rejected before normalization.");
            string full = Path.GetFullPath(path);
            if (full.Length < 4 || full[1] != ':' || full[2] != '\\' || full.StartsWith(@"\\", StringComparison.Ordinal)) throw new IOException("Only absolute local drive paths are supported.");
            foreach (string part in full.Substring(3).Split('\\'))
            {
                if (part.Length == 0 || part.EndsWith(".", StringComparison.Ordinal) || part.EndsWith(" ", StringComparison.Ordinal) || part.Contains(":")) throw new IOException("Ambiguous or streamed archive path is rejected.");
                if (String.Equals(part, "$null", StringComparison.OrdinalIgnoreCase) || Regex.IsMatch(part.Split('.')[0], @"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", RegexOptions.IgnoreCase)) throw new IOException("Reserved archive path is rejected.");
            }
            return full;
        }

        public PcaiArchiveSource(string path)
        {
            SourcePath = ValidateLocalPath(path);
            try
            {
                // Hold ancestors against renaming; directory writes remain shared.
                Stack<string> parents = new Stack<string>();
                for (DirectoryInfo parent = new FileInfo(SourcePath).Directory; parent != null; parent = parent.Parent) parents.Push(parent.FullName);
                foreach (string parent in parents)
                {
                    SafeFileHandle guard = Open(parent, 1, ShareRead | ShareWrite, DirectoryFlag | OpenReparse);
                    directories.Add(guard); RejectReparse(guard);
                }
                SafeFileHandle readHandle = Open(SourcePath, Read, ShareRead | ShareDelete, OpenReparse);
                try { RejectReparse(readHandle); stream = new FileStream(readHandle, FileAccess.Read, 131072, false); }
                catch { readHandle.Dispose(); throw; }
                // Real read access participates in Windows sharing checks; a
                // zero-access metadata handle does not reliably block renames.
                // No DELETE access is held, keeping ordinary rclone readers compatible.
                renameGuard = Open(SourcePath, Read, ShareRead, OpenReparse);
                RejectReparse(renameGuard);
                Identity identity = ReadIdentity(stream.SafeFileHandle), guarded = ReadIdentity(renameGuard);
                if (identity.VolumeSerial != guarded.VolumeSerial || identity.Low != guarded.Low || identity.High != guarded.High) throw new IOException("Source identity changed while acquiring custody.");
                CanonicalGuidPath = GuidPath(stream.SafeFileHandle);
                VolumeSerialHex = identity.VolumeSerial.ToString("x16"); FileId128 = IdText(identity);
                Length = stream.Length;
                Standard standard;
                if (!GetStandardByHandle(stream.SafeFileHandle, 1, out standard, 24)) throw new Win32Exception(Marshal.GetLastWin32Error(), "Source allocation and link count are required.");
                HardLinkCount = standard.Links; AllocatedBytes = standard.Allocation;
                using (IncrementalHash md5 = IncrementalHash.CreateHash(HashAlgorithmName.MD5))
                using (IncrementalHash sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
                {
                    byte[] buffer = new byte[1048576]; int count;
                    while ((count = stream.Read(buffer, 0, buffer.Length)) != 0) { md5.AppendData(buffer, 0, count); sha.AppendData(buffer, 0, count); }
                    Md5 = Convert.ToHexString(md5.GetHashAndReset()).ToLowerInvariant(); Sha256 = Convert.ToHexString(sha.GetHashAndReset()).ToLowerInvariant();
                    stream.Position = 0;
                }
            }
            catch { Dispose(); throw; }
        }

        /// <summary>Append and flush a receipt before authorizing exact-identity deletion.</summary>
        public void PersistVerifiedReceipt(string manifestPath, string recordJson, string remoteMd5, long remoteSize, long remoteCount)
        {
            if (disposed || deleted) throw new ObjectDisposedException(nameof(PcaiArchiveSource));
            if (!String.Equals(remoteMd5, Md5, StringComparison.OrdinalIgnoreCase) || remoteSize != Length || remoteCount != 1) throw new IOException("Archive hash, size or count does not match held source bytes.");
            AcquireReceipt(manifestPath);
            receiptWriter.Append(recordJson);
            receiptDurable = true;
        }

        /// <summary>Hold the exact receipt leaf and ancestors until source custody ends.</summary>
        public void AcquireReceipt(string manifestPath)
        {
            if (disposed) throw new ObjectDisposedException(nameof(PcaiArchiveSource));
            string full = ValidateLocalPath(manifestPath);
            if (receiptDirectory != null)
            {
                if (!String.Equals(full, receiptDirectory.ManifestPath, StringComparison.OrdinalIgnoreCase)) throw new IOException("Verified receipt path cannot change during source custody.");
                return;
            }
            ReceiptDirectoryCustody parent = new ReceiptDirectoryCustody(full);
            try { receiptWriter = parent.OpenWriter(); receiptDirectory = parent; }
            catch { parent.Dispose(); throw; }
        }

        /// <summary>Append subsequent events through the same held verified receipt.</summary>
        public void AppendReceipt(string manifestPath, string recordJson)
        {
            if (disposed) throw new ObjectDisposedException(nameof(PcaiArchiveSource));
            AcquireReceipt(manifestPath);
            receiptWriter.Append(recordJson);
        }

        public static void AppendDurableReceipt(string manifestPath, string recordJson)
        {
            using (ReceiptDirectoryCustody parent = new ReceiptDirectoryCustody(manifestPath))
            using (ReceiptWriter writer = parent.OpenWriter()) writer.Append(recordJson);
        }

        /// <summary>Hold receipt ancestors before any directory creation or leaf open.</summary>
        public sealed class ReceiptDirectoryCustody : IDisposable
        {
            private readonly List<SafeFileHandle> guards = new List<SafeFileHandle>();
            private bool disposed;
            public string ManifestPath { get; private set; }
            public string CanonicalDirectoryGuidPath { get; private set; }
            public string VolumeSerialHex { get; private set; }

            public ReceiptDirectoryCustody(string manifestPath)
            {
                ManifestPath = ValidateLocalPath(manifestPath);
                try
                {
                    Stack<string> parents = new Stack<string>();
                    for (DirectoryInfo parent = new FileInfo(ManifestPath).Directory; parent != null; parent = parent.Parent) parents.Push(parent.FullName);
                    string expected = null;
                    while (parents.Count != 0)
                    {
                        string parent = parents.Pop();
                        // Only missing descendants of an already held parent can be created.
                        // A rival junction or file is opened without following reparse points.
                        SafeFileHandle guard = CreateFileW(parent, 1, ShareRead, IntPtr.Zero, OpenExisting, DirectoryFlag | OpenReparse, IntPtr.Zero);
                        if (guard.IsInvalid)
                        {
                            int error = Marshal.GetLastWin32Error(); guard.Dispose();
                            if ((error != 2 && error != 3) || guards.Count == 0) throw new Win32Exception(error, "Cannot hold receipt ancestor: " + parent);
                            if (!CreateDirectoryW(parent, IntPtr.Zero))
                            {
                                error = Marshal.GetLastWin32Error();
                                if (error != 183) throw new Win32Exception(error, "Cannot create receipt directory under held parent: " + parent);
                            }
                            guard = Open(parent, 1, ShareRead, DirectoryFlag | OpenReparse);
                        }
                        guards.Add(guard);
                        RejectReparse(guard);
                        Standard standard;
                        if (!GetStandardByHandle(guard, 1, out standard, 24) || standard.Directory == 0) throw new IOException("Receipt ancestor must be a held directory.");
                        string actual = GuidPath(guard).TrimEnd('\\');
                        if (expected != null && !String.Equals(actual, expected, StringComparison.OrdinalIgnoreCase)) throw new IOException("Receipt ancestor GUID path changed during custody acquisition: expected " + expected + "; actual " + actual);
                        CanonicalDirectoryGuidPath = actual;
                        VolumeSerialHex = ReadIdentity(guard).VolumeSerial.ToString("x16");
                        if (parents.Count != 0) expected = actual + "\\" + Path.GetFileName(parents.Peek().TrimEnd('\\'));
                    }
                }
                catch { Dispose(); throw; }
            }

            public ReceiptWriter OpenWriter()
            {
                if (disposed) throw new ObjectDisposedException(nameof(ReceiptDirectoryCustody));
                foreach (SafeFileHandle guard in guards) RejectReparse(guard);
                return new ReceiptWriter(this);
            }

            public void Dispose()
            {
                if (disposed) return;
                disposed = true;
                for (int index = guards.Count - 1; index >= 0; index--) guards[index].Dispose();
            }
        }

        /// <summary>Append only through a validated native receipt handle without overwrite.</summary>
        public sealed class ReceiptWriter : IDisposable
        {
            private readonly ReceiptDirectoryCustody parent;
            private FileStream output;
            public string CanonicalGuidPath { get; private set; }
            public string VolumeSerialHex { get; private set; }
            public string FileId128 { get; private set; }

            internal ReceiptWriter(ReceiptDirectoryCustody directory)
            {
                parent = directory;
                SafeFileHandle handle = CreateFileW(parent.ManifestPath, Read | Write, ShareRead, IntPtr.Zero, OpenAlways, OpenReparse | WriteThrough, IntPtr.Zero);
                if (handle.IsInvalid) { int error = Marshal.GetLastWin32Error(); handle.Dispose(); throw new Win32Exception(error, "Cannot acquire receipt leaf custody."); }
                try
                {
                    RejectReparse(handle);
                    Identity identity = ReadIdentity(handle);
                    CanonicalGuidPath = GuidPath(handle);
                    VolumeSerialHex = identity.VolumeSerial.ToString("x16"); FileId128 = IdText(identity);
                    if (!String.Equals(CanonicalGuidPath, parent.CanonicalDirectoryGuidPath + "\\" + Path.GetFileName(parent.ManifestPath), StringComparison.OrdinalIgnoreCase) || VolumeSerialHex != parent.VolumeSerialHex) throw new IOException("Receipt leaf is outside the held GUID parent.");
                    Standard standard;
                    if (!GetStandardByHandle(handle, 1, out standard, 24) || standard.Directory != 0 || standard.Pending != 0 || standard.Links != 1) throw new IOException("Receipt leaf must be one ordinary, unlinked file.");
                    output = new FileStream(handle, FileAccess.ReadWrite, 4096, false);
                }
                catch { handle.Dispose(); throw; }
            }

            public void Append(string recordJson)
            {
                if (output == null) throw new ObjectDisposedException(nameof(ReceiptWriter));
                RejectReparse(output.SafeFileHandle);
                Identity actual = ReadIdentity(output.SafeFileHandle);
                if (actual.VolumeSerial.ToString("x16") != VolumeSerialHex || IdText(actual) != FileId128 || !String.Equals(GuidPath(output.SafeFileHandle), CanonicalGuidPath, StringComparison.OrdinalIgnoreCase)) throw new IOException("Receipt identity changed; no bytes appended.");
                byte[] bytes = new UTF8Encoding(false, true).GetBytes(recordJson + "\n");
                output.Seek(0, SeekOrigin.End);
                output.Write(bytes, 0, bytes.Length);
                output.Flush(true);
            }

            public void Dispose() { if (output != null) { output.Dispose(); output = null; } }
        }

        /// <summary>Release rename protection while retaining immutable bytes for deletion admission.</summary>
        public void PrepareForDeletion()
        {
            if (!receiptDurable || disposed || deleted) throw new InvalidOperationException("A durable verified receipt is required before deletion.");
            if (renameGuard != null) { renameGuard.Dispose(); renameGuard = null; }
        }

        /// <summary>Delete only the file instance whose held bytes were verified.</summary>
        public void DeleteVerified()
        {
            PrepareForDeletion();
            using (SafeFileHandle target = Open(SourcePath, Delete, ShareRead, OpenReparse))
            {
                RejectReparse(target);
                Identity actual = ReadIdentity(target);
                if (actual.VolumeSerial.ToString("x16") != VolumeSerialHex || IdText(actual) != FileId128 || !String.Equals(GuidPath(target), CanonicalGuidPath, StringComparison.OrdinalIgnoreCase)) throw new IOException("Source identity changed before deletion; replacement preserved.");
                Standard standard;
                if (!GetStandardByHandle(target, 1, out standard, 24)) throw new Win32Exception(Marshal.GetLastWin32Error(), "Cannot verify deletion link count.");
                HardLinkCountBeforeDeletion = standard.Links;
                Disposition disposition = new Disposition { DeleteFile = 1 };
                if (!SetFileInformationByHandle(target, 4, ref disposition, 1)) throw new Win32Exception(Marshal.GetLastWin32Error(), "Identity-bound archive deletion failed.");
                stream.Dispose(); stream = null; deleted = true;
            }
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            if (renameGuard != null) renameGuard.Dispose();
            if (stream != null) stream.Dispose();
            if (receiptWriter != null) receiptWriter.Dispose();
            if (receiptDirectory != null) receiptDirectory.Dispose();
            for (int i = directories.Count - 1; i >= 0; i--) directories[i].Dispose();
        }
    }
}
