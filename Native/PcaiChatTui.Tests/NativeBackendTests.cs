using System;
using System.Threading.Tasks;
using Moq;
using Xunit;
using PcaiChatTui;

namespace PcaiChatTui.Tests;

public class NativeBackendTests
{
    [Fact]
    public async Task CheckAvailabilityAsync_Success_ReturnsTrue()
    {
        // Arrange
        var mockModule = new Mock<INativeInferenceModule>();
        // Mock init to return 0 (success)
        mockModule.Setup(m => m.pcai_init("llamacpp")).Returns(0);
        
        var backend = new NativeBackend("llamacpp", mockModule.Object);

        // Act
        var result = await backend.CheckAvailabilityAsync();

        // Assert
        Assert.True(result);
        mockModule.Verify(m => m.pcai_init("llamacpp"), Times.Once);
    }

    [Fact]
    public async Task CheckAvailabilityAsync_InitFails_ReturnsFalse()
    {
        // Arrange
        var mockModule = new Mock<INativeInferenceModule>();
        // Mock init to return non-zero (failure)
        mockModule.Setup(m => m.pcai_init("llamacpp")).Returns(1);
        
        var backend = new NativeBackend("llamacpp", mockModule.Object);

        // Act
        var result = await backend.CheckAvailabilityAsync();

        // Assert
        Assert.False(result);
        mockModule.Verify(m => m.pcai_init("llamacpp"), Times.Once);
    }

    [Fact]
    public async Task CheckAvailabilityAsync_DllNotFoundException_ReturnsFalse()
    {
        // Arrange
        var mockModule = new Mock<INativeInferenceModule>();
        mockModule.Setup(m => m.pcai_init(It.IsAny<string>()))
                  .Throws(new DllNotFoundException("pcai_inference.dll not found"));
        
        var backend = new NativeBackend("mistralrs", mockModule.Object);

        // Act
        var result = await backend.CheckAvailabilityAsync();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task CheckAvailabilityAsync_EntryPointNotFoundException_ReturnsFalse()
    {
        // Arrange
        var mockModule = new Mock<INativeInferenceModule>();
        mockModule.Setup(m => m.pcai_init(It.IsAny<string>()))
                  .Throws(new EntryPointNotFoundException("pcai_init not found"));
        
        var backend = new NativeBackend("mistralrs", mockModule.Object);

        // Act
        var result = await backend.CheckAvailabilityAsync();

        // Assert
        Assert.False(result);
    }
}
