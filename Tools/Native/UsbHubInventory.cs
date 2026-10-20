// Read-only USB hub queries. Layouts verified against Windows SDK 10.0.26100.0,
// usbioctl.h (pack1) and usbiodef.h. No port reset or vendor IOCTL is used.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace PcAi.Diagnostics {
 public sealed class PortInfo {
  public int Port {get;set;} public uint? ConnectionStatus {get;set;}
  public string VendorId {get;set;} public string ProductId {get;set;}
  public byte? DeviceClass {get;set;} public byte? Configuration {get;set;}
  public byte? LegacySpeed {get;set;} public bool? IsHub {get;set;}
  public ushort? Address {get;set;} public uint? OpenPipes {get;set;}
  public bool? SuperSpeedOrHigher {get;set;} public string SpeedQueryError {get;set;}
  public byte? MaxPowerRaw {get;set;} public int? DeclaredMaxBusCurrentMa {get;set;}
  public bool? SelfPowered {get;set;} public bool? RemoteWakeCapable {get;set;}
  public byte DescriptorConfigurationIndex {get;set;}
  public byte? DescriptorConfigurationValue {get;set;}
  public bool? MatchesActiveConfiguration {get;set;}
  public List<string> InterfaceClasses {get;set;} = new List<string>();
  public string ConfigurationQueryError {get;set;}
  public string Error {get;set;}
 }
 public sealed class HubInfo {
  public int HubIndex {get;set;} public int? PortCount {get;set;}
  public bool? BusPowered {get;set;} public string Error {get;set;}
  public List<PortInfo> Ports {get;set;} = new List<PortInfo>();
 }
 public static class UsbInventory {
  [StructLayout(LayoutKind.Sequential)] struct InterfaceData {
   public uint Size; public Guid ClassGuid; public uint Flags; public IntPtr Reserved;
  }
  [DllImport("setupapi.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  static extern IntPtr SetupDiGetClassDevs(ref Guid g, string e, IntPtr p, uint flags);
  [DllImport("setupapi.dll", SetLastError=true)]
  static extern bool SetupDiEnumDeviceInterfaces(IntPtr set, IntPtr d, ref Guid g, uint index, ref InterfaceData data);
  [DllImport("setupapi.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  static extern bool SetupDiGetDeviceInterfaceDetail(IntPtr set, ref InterfaceData data, IntPtr detail, uint size, out uint needed, IntPtr device);
  [DllImport("setupapi.dll", SetLastError=true)] static extern bool SetupDiDestroyDeviceInfoList(IntPtr set);
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  static extern SafeFileHandle CreateFile(string path, uint access, uint share, IntPtr sec, uint creation, uint flags, IntPtr template);
  [DllImport("kernel32.dll", SetLastError=true)]
  static extern bool DeviceIoControl(SafeFileHandle h, uint code, byte[] input, uint inputSize, byte[] output, uint outputSize, out uint received, IntPtr overlapped);
  static string ErrorCode(Exception error) { var win32=error as Win32Exception; return win32==null ? error.GetType().Name : "Win32Error:"+win32.NativeErrorCode; }
  static string LastError() { int n=Marshal.GetLastWin32Error(); return "Win32Error:"+n; }
  static byte[] Query(SafeFileHandle h, uint code, int port, int minSize) {
   byte[] input=new byte[4096], output=new byte[4096];
   Array.Copy(BitConverter.GetBytes(port),input,4);
   uint received;
   if(!DeviceIoControl(h,code,input,(uint)input.Length,output,(uint)output.Length,out received,IntPtr.Zero)) throw new Win32Exception(Marshal.GetLastWin32Error());
   if(received<minSize || received>output.Length) throw new InvalidOperationException("Invalid hub response length: "+received);
   return output;
  }
  static byte[] Descriptor(SafeFileHandle h, int port, int length) {
   byte[] input=new byte[12+length], output=new byte[12+length];
   Array.Copy(BitConverter.GetBytes(port),input,4);
   input[4]=0x80; input[5]=6; input[7]=2; // GET_DESCRIPTOR configuration index 0
   Array.Copy(BitConverter.GetBytes((ushort)length),0,input,10,2);
   uint received;
   if(!DeviceIoControl(h,0x220410,input,(uint)input.Length,output,(uint)output.Length,out received,IntPtr.Zero)) throw new Win32Exception(Marshal.GetLastWin32Error());
   if(received<21 || received>output.Length) throw new InvalidOperationException("Invalid configuration descriptor response length: "+received);
   var result=new byte[received-12]; Array.Copy(output,12,result,0,result.Length); return result;
  }
  static void PowerAndInterfaces(SafeFileHandle h, int port, PortInfo info) {
   try {
    byte[] input=new byte[16], output=new byte[16];
    Array.Copy(BitConverter.GetBytes(port),input,4); input[4]=16; input[8]=7;
    uint received;
    if(!DeviceIoControl(h,0x22045C,input,16,output,16,out received,IntPtr.Zero)) throw new Win32Exception(Marshal.GetLastWin32Error());
    if(received<16) throw new InvalidOperationException("Short EX_V2 response");
    info.SuperSpeedOrHigher=(BitConverter.ToUInt32(output,12)&1)!=0;
   } catch(Exception e) {info.SpeedQueryError=ErrorCode(e);}
   try {
    var head=Descriptor(h,port,9);
    if(head[0]!=9 || head[1]!=2) throw new InvalidOperationException("Invalid configuration header");
    int length=BitConverter.ToUInt16(head,2);
    if(length<9 || length>16384) throw new InvalidOperationException("Unsupported configuration size "+length);
    var config=Descriptor(h,port,length);
    DecodeConfiguration(config,length,info);
   } catch(Exception e) {info.ConfigurationQueryError=ErrorCode(e);}
  }
  // Pure descriptor decoding used by the real reader and synthetic safety tests.
  public static void DecodeConfiguration(byte[] config, int length, PortInfo info) {
   if(info==null) throw new ArgumentNullException(nameof(info));
   if(config==null) throw new ArgumentNullException(nameof(config));
   if(length<9 || length>16384 || config.Length<length) throw new InvalidOperationException("Truncated or unsupported configuration descriptor");
   if(config[0]!=9 || config[1]!=2 || config[5]==0 || BitConverter.ToUInt16(config,2)!=length || (config[7]&128)==0) throw new InvalidOperationException("Malformed configuration header");
   var classes=new List<string>();
   for(int offset=0;offset<length;) {
    int size=config[offset];
    if(size<2 || offset+size>length) throw new InvalidOperationException("Malformed configuration descriptor chain");
    if(config[offset+1]==4) {
     if(size<9) throw new InvalidOperationException("Short interface descriptor");
     classes.Add(config[offset+5].ToString("X2")+":"+config[offset+6].ToString("X2")+":"+config[offset+7].ToString("X2"));
    }
    offset+=size;
   }
   // Publish only after the entire chain validates. Unknown speed/configuration cannot imply a budget.
   info.DescriptorConfigurationIndex=0; info.DescriptorConfigurationValue=config[5];
   info.MatchesActiveConfiguration=info.Configuration.HasValue ? (bool?)(config[5]==info.Configuration.Value) : null;
   info.MaxPowerRaw=config[8]; info.SelfPowered=(config[7]&64)!=0; info.RemoteWakeCapable=(config[7]&32)!=0;
   info.InterfaceClasses=classes; info.DeclaredMaxBusCurrentMa=null;
   if(info.Configuration.HasValue && info.Configuration.Value>0 && info.MatchesActiveConfiguration==true && info.SuperSpeedOrHigher.HasValue) info.DeclaredMaxBusCurrentMa=config[8]*(info.SuperSpeedOrHigher.Value?8:2);
  }
  public static List<HubInfo> Read() {
   var hubs=new List<HubInfo>(); Guid guid=new Guid("f18a0e88-c30c-11d0-8815-00a0c906bed8");
   IntPtr set=SetupDiGetClassDevs(ref guid,null,IntPtr.Zero,18);
   if(set==new IntPtr(-1)) throw new Win32Exception(Marshal.GetLastWin32Error());
   try {
    for(uint index=0;;index++) {
     var data=new InterfaceData { Size=(uint)Marshal.SizeOf<InterfaceData>() };
     if(!SetupDiEnumDeviceInterfaces(set,IntPtr.Zero,ref guid,index,ref data)) {
      int error=Marshal.GetLastWin32Error(); if(error==259) break; throw new Win32Exception(error);
     }
     uint needed; SetupDiGetDeviceInterfaceDetail(set,ref data,IntPtr.Zero,0,out needed,IntPtr.Zero);
     if(needed<8 || needed>65536) throw new InvalidOperationException("Invalid interface detail size: "+needed);
     IntPtr detail=Marshal.AllocHGlobal((int)needed);
     string path;
     try {
      Marshal.WriteInt32(detail,IntPtr.Size==8?8:6);
      if(!SetupDiGetDeviceInterfaceDetail(set,ref data,detail,needed,out needed,IntPtr.Zero)) throw new Win32Exception(Marshal.GetLastWin32Error());
      path=Marshal.PtrToStringUni(IntPtr.Add(detail,4));
     } finally {Marshal.FreeHGlobal(detail);}
     var hub=new HubInfo {HubIndex=hubs.Count+1}; hubs.Add(hub);
     using(var h=CreateFile(path,0xC0000000,3,IntPtr.Zero,3,0,IntPtr.Zero)) {
      if(h.IsInvalid) {hub.Error=LastError();continue;}
      try {
       var node=Query(h,0x220408,0,76); hub.PortCount=node[6]; hub.BusPowered=node[75]!=0;
       for(int port=1;port<=hub.PortCount.Value;port++) {
        var info=new PortInfo {Port=port}; hub.Ports.Add(info);
        try {
         var b=Query(h,0x220448,port,35);
         info.ConnectionStatus=BitConverter.ToUInt32(b,31);
         if(info.ConnectionStatus!=1) continue;
         if(b[4]!=18 || b[5]!=1) throw new InvalidOperationException("Invalid device descriptor header");
         info.VendorId=BitConverter.ToUInt16(b,12).ToString("X4");
         info.ProductId=BitConverter.ToUInt16(b,14).ToString("X4");
         info.DeviceClass=b[8]; info.Configuration=b[22]; info.LegacySpeed=b[23];
         info.IsHub=b[24]!=0; info.Address=BitConverter.ToUInt16(b,25); info.OpenPipes=BitConverter.ToUInt32(b,27);
         PowerAndInterfaces(h,port,info);
        } catch(Exception e) {info.Error=ErrorCode(e);}
       }
      } catch(Exception e) {hub.Error=ErrorCode(e);}
     }
    }
   } finally {SetupDiDestroyDeviceInfoList(set);}
   return hubs;
  }
 }
}
