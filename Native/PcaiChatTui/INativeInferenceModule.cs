using System;
using System.Runtime.InteropServices;
using PcaiNative;

namespace PcaiChatTui;

/// <summary>
/// Abstraction for the native P/Invoke calls to pcai_inference.dll for testability.
/// </summary>
public interface INativeInferenceModule
{
    int pcai_init(string backendName);
    int pcai_load_model(string modelPath, int gpuLayers);
    string? Generate(string prompt, uint maxTokens = 512, float temperature = 0.7f);
    string? GetLastError();
    int pcai_generate_streaming(string prompt, uint maxTokens, float temperature, InferenceModule.TokenCallback callback, IntPtr userData);
    void pcai_shutdown();
}

/// <summary>
/// Real implementation of INativeInferenceModule that calls the static methods in PcaiNative.InferenceModule.
/// </summary>
public class NativeInferenceModuleWrapper : INativeInferenceModule
{
    public int pcai_init(string backendName) => InferenceModule.pcai_init(backendName);
    public int pcai_load_model(string modelPath, int gpuLayers) => InferenceModule.pcai_load_model(modelPath, gpuLayers);
    public string? Generate(string prompt, uint maxTokens = 512, float temperature = 0.7f) => InferenceModule.Generate(prompt, maxTokens, temperature);
    public string? GetLastError() => InferenceModule.GetLastError();
    public int pcai_generate_streaming(string prompt, uint maxTokens, float temperature, InferenceModule.TokenCallback callback, IntPtr userData) => InferenceModule.pcai_generate_streaming(prompt, maxTokens, temperature, callback, userData);
    public void pcai_shutdown() => InferenceModule.pcai_shutdown();
}
