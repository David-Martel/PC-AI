using System;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Moq;
using Moq.Protected;
using Xunit;
using PcaiChatTui;

namespace PcaiChatTui.Tests;

public class HttpBackendTests
{
    private const string TestEndpoint = "http://localhost:8080";

    [Fact]
    public async Task CheckAvailabilityAsync_SuccessResponse_ReturnsTrue()
    {
        // Arrange
        var handlerMock = new Mock<HttpMessageHandler>(MockBehavior.Strict);
        handlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.Is<HttpRequestMessage>(req => req.Method == HttpMethod.Get && req.RequestUri != null && req.RequestUri.ToString() == $"{TestEndpoint}/v1/models"),
                ItExpr.IsAny<CancellationToken>()
            )
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK
            });

        var backend = new HttpBackend(TestEndpoint, handlerMock.Object);

        // Act
        var result = await backend.CheckAvailabilityAsync();

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task CheckAvailabilityAsync_ErrorResponse_ReturnsFalse()
    {
        // Arrange
        var handlerMock = new Mock<HttpMessageHandler>(MockBehavior.Strict);
        handlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.Is<HttpRequestMessage>(req => req.Method == HttpMethod.Get && req.RequestUri != null && req.RequestUri.ToString() == $"{TestEndpoint}/v1/models"),
                ItExpr.IsAny<CancellationToken>()
            )
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.InternalServerError
            });

        var backend = new HttpBackend(TestEndpoint, handlerMock.Object);

        // Act
        var result = await backend.CheckAvailabilityAsync();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task CheckAvailabilityAsync_NetworkError_ReturnsFalse()
    {
        // Arrange
        var handlerMock = new Mock<HttpMessageHandler>(MockBehavior.Strict);
        handlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>()
            )
            .ThrowsAsync(new HttpRequestException("Connection refused"));

        var backend = new HttpBackend(TestEndpoint, handlerMock.Object);

        // Act
        var result = await backend.CheckAvailabilityAsync();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task CheckAvailabilityAsync_TaskCancelled_ThrowsOrReturnsFalse()
    {
        // Arrange
        var handlerMock = new Mock<HttpMessageHandler>(MockBehavior.Strict);
        handlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>()
            )
            .ThrowsAsync(new TaskCanceledException("A task was canceled."));

        var backend = new HttpBackend(TestEndpoint, handlerMock.Object);

        // Act
        var result = await backend.CheckAvailabilityAsync();

        // Assert
        Assert.False(result);
    }
    
    [Fact]
    public async Task CheckAvailabilityAsync_CancellationTokenCancelled_ReturnsFalse()
    {
        // Arrange
        var handlerMock = new Mock<HttpMessageHandler>(MockBehavior.Strict);
        // We simulate a long running task that observes the cancellation token
        handlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>()
            )
            .Returns<HttpRequestMessage, CancellationToken>(async (req, ct) =>
            {
                await Task.Delay(5000, ct); // Should throw OperationCanceledException
                return new HttpResponseMessage(HttpStatusCode.OK);
            });

        var backend = new HttpBackend(TestEndpoint, handlerMock.Object);
        using var cts = new CancellationTokenSource();
        cts.Cancel(); // Cancel immediately

        // Act
        var result = await backend.CheckAvailabilityAsync(cts.Token);

        // Assert
        Assert.False(result);
    }
}
