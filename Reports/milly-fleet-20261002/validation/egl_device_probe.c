#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    PFNEGLQUERYDEVICESEXTPROC query = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC platform = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!query || !platform) { fprintf(stderr, "EGL device extension unavailable\n"); return 2; }
    EGLDeviceEXT devices[16];
    EGLint count = 0;
    if (!query(16, devices, &count)) { fprintf(stderr, "Device enumeration failed\n"); return 3; }
    int nvidia_verified = 0;
    for (EGLint i = 0; i < count; ++i) {
        EGLDisplay display = platform(EGL_PLATFORM_DEVICE_EXT, devices[i], NULL);
        EGLint major = 0, minor = 0;
        if (!eglInitialize(display, &major, &minor)) continue;
        const char *vendor = eglQueryString(display, EGL_VENDOR);
        if (!vendor || strstr(vendor, "NVIDIA") == NULL) { eglTerminate(display); continue; }
        const EGLint config_attributes[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE };
        EGLConfig config;
        EGLint found = 0;
        if (!eglChooseConfig(display, config_attributes, &config, 1, &found) || found != 1 || !eglBindAPI(EGL_OPENGL_API)) {
            fprintf(stderr, "NVIDIA EGL config/API failure\n"); eglTerminate(display); return 4;
        }
        const EGLint size[] = { EGL_WIDTH, 8, EGL_HEIGHT, 8, EGL_NONE };
        EGLSurface surface = eglCreatePbufferSurface(display, config, size);
        EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
        if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT || !eglMakeCurrent(display, surface, surface, context)) {
            fprintf(stderr, "NVIDIA EGL context failure\n"); eglTerminate(display); return 5;
        }
        glViewport(0, 0, 8, 8);
        glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        unsigned char pixel[3] = {0, 0, 0};
        glReadPixels(4, 4, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pixel);
        GLenum error = glGetError();
        printf("EGL %d.%d vendor=%s renderer=%s GL=%s pixel=%u,%u,%u error=%u\n",
            major, minor, vendor, glGetString(GL_RENDERER), glGetString(GL_VERSION),
            pixel[0], pixel[1], pixel[2], error);
        nvidia_verified = error == GL_NO_ERROR && pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0;
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(display, context);
        eglDestroySurface(display, surface);
        eglTerminate(display);
        if (!nvidia_verified) return 6;
    }
    if (!nvidia_verified) { fprintf(stderr, "No NVIDIA EGL device rendered the expected pixel\n"); return 7; }
    return 0;
}
