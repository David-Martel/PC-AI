using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;

namespace PcaiNative.Tests;

public class NativeResolverRegistrationTests
{
    [Fact]
    public void FailedRegistrationDoesNotPublishAFalseSuccess()
    {
        // Keep resolver state isolated from the actual native integration fixture.
        var context = new AssemblyLoadContext("resolver-failure-fixture", isCollectible: true);
        try
        {
            var assembly = context.LoadFromAssemblyPath(typeof(PcaiCore).Assembly.Location);
            NativeLibrary.SetDllImportResolver(assembly, (_, _, _) => IntPtr.Zero);
            var resolver = assembly.GetType("PcaiNative.NativeResolver", throwOnError: true)!;
            var register = resolver.GetMethod("EnsureRegistered", BindingFlags.Static | BindingFlags.NonPublic)!;
            // Both attempts must report the real registration conflict. Previously
            // the first failure left the published flag set and the second succeeded.
            for (var attempt = 0; attempt < 2; attempt++)
            {
                var error = Assert.Throws<TargetInvocationException>(() => register.Invoke(null, new object[] { assembly }));
                Assert.IsType<InvalidOperationException>(error.InnerException);
            }
        }
        finally
        {
            context.Unload();
        }
    }
}
