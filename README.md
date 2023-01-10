# ReSTIR

## External resources

- CUDA (my version is 11.3)
- OptiX 7 SDK
- GLFW (code is in common/3rdParty)
- stb_image (code is in common/3rdParty/stb_image.h)
- TinyObjLoader (code is in common/3rdParty/tiny_obj_loader.h)
- gdt (gpu developer tools, includes basic vec3 calculation, code is in common/gdt)
- glfWindow (a basic window class provided by optix7course, code is in common/glfWindow)
- code base: optix7course
    - provide a basic framework of how to complie, run with optix and cuda.
    - we write our own code include construct scene, control camera, manage gpu memory, and code of ReSTIR in .cu
    - Allowed by TA (QAQ)

## Executable

Executable is in "build/Release/restir.exe", please don't move due to the relative path.

We compile this program by MSVC from VS2019 community.

If you can run this program, compress "f" to unlock camera, "w, a, s, d" to move camera.

## Addition materials

See our result in "results" directory for detail, big figure.

