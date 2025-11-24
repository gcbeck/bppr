# BPPR - Build Instructions

The BPPR executable has been tested on Arch Linux running kernel version 6.16.4 
on old Intel Core 2 Duo hardware at 1.86GHz. 
Compilation was performed with the Intel oneAPI DPC++/C++ Compiler version 2025.0.4.

The build structure is simple enough to justify a straightforward `tasks.json` entry in `vscode` rather than the machinery of a makefile. Some guidelines found below.  

## Prerequisites

1. Intel Math Kernel Library, part of the [oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html). The build options for this library are quite involved; start from the suggestions here but it's worth reading the documentation too.  
2. The `cx_math.h`  header in the [constexpr](https://github.com/elbeno/constexpr) library for compile-time math utilities. This just implies `-I${CXROOT}/src/include` in your build args. 
3. The low-level logging library [L3](https://github.com/undoio/l3) which needs some more care as we extend its capability to floating point scalars. Along with the obligatory `-I${L3ROOT}`, `${L3ROOT}/l3.c` and `${L3ROOT}/l3.S` should be added to the object compilation list. Additionally the binary file parser `l3_dump.py` requires the following modifications:
    1. Add this section near the top of the file, just below the declaration section for `Enum types defined in src/l3.c for L3_LOG()->{platform, loc_type} fields`:
    ```
    # #############################################################################
    # Allowing for negative representation of C++ uint64 inputs 
    DETERMINANT = (2 ** 63)
    MAX_UINT64    = (2 ** 64) - 1
    _XX, _XXX           = [2**63,  2**63+2**62]
    DD                      = _XX+9
    DDD, ODD         = [_XXX+99,  _XXX+9]

    def decimalize(uint:int):
        if uint >= _XXX:
            return ('' if uint > DDD else ('0' if uint > ODD else '00')) + str(uint-_XXX)
        elif uint >= _XX:
            return ('' if uint > DD else '0') + str(uint-_XX)
        return str(uint)
    ```
    2. Modify the function `do_c_print`:
    ```
    if arg1 == DETERMINANT:
        msg_text = re.sub(r"%[nu]\.%u", "-inf", msg_text)
    format_string = re.sub(r"%([nu])\.%u", r"%\g<1>." + decimalize(arg2), msg_text)
    format_string = re.sub(r"(%[und][^\.]+)%n", (r"\g<1>"+str(MAX_UINT64-arg2)) if (DETERMINANT < arg2 <= MAX_UINT64) else r"\g<1>"+str(arg2), format_string, count=1)
    format_string = re.sub(r"%n", (r"-"+str(MAX_UINT64-arg1)) if (DETERMINANT < arg1 <= MAX_UINT64) else str(arg1), format_string, count=1)
    format_string = fmtstr_replace(format_string)

    args = format_string.count("%")
    if args == 1: return format_string % arg1 
    if args == 2: return format_string % (arg1, arg2) 
    return format_string
    ```
    3. (Possibly unnecessary; try this if you get `KeyError`) Modify the following clause in `do_main`:
    ```
    if OS_UNAME_S == 'Linux':
        offs = msgptr - rodata_offs  # Force offset calc to ignore fibase for l3_init in header
    ```
4. The environment variables `CXROOT` and `L3ROOT` must be set and visible to the compiler. One way to achieve this, assuming the prerequisites at location `$GITROOT/{l3,cx}`, is with the following added to `~./bashrc`:
```
alias l3="export L3ROOT=${GITROOT}/l3 && export L3_LOC_ENABLED=0"
alias cx="export CXROOT=${GITROOT}/cx"
l3log () {
    locarg=$([[ $# -gt 2 ]] && echo "--loc-binary $3" || echo "")
    L3_LOC_ENABLED=0 ${L3ROOT}/l3_dump.py --log-file $1 --binary $2 ${locarg} 
}
```
so that the environment variables are in vscode's launch context when starting with the command
```
l3 && cx && code &
```
The environment variable `MKLROOT` is often set via the Intel oneAPI script `setvars.sh`.

## Example

Putting it all together for a build target `bppr`, an admittedly vscode-centric build line would look something like this: 
```
{
    "type": "cppbuild",
    "label": "C/C++: icpx build active file",
    "command": "/opt/intel/oneapi/compiler/latest/bin/icpx",
    //"dependsOn": "l3locmv",
    "args": [
        "-fdiagnostics-color=always",
        "-std=c++20",
        //"-I${workspaceFolder}/loc",
        "-I${L3ROOT}",
        "-I${CXROOT}/src/include",
        "-I${MKLROOT}/include",
        "-g",
        "-static-intel",
        "-static-libgcc",
        "-static-libstdc++",
        "-L${MKLROOT}/lib",
        "-Wl,--start-group",
        "-lmkl_intel_lp64",
        "-lmkl_intel_thread",
        "-lmkl_core",
        "-Wl,--end-group",
        "-liomp5",
        "-lpthread",
        "-lm",
        "-ldl",
        // "-DL3_LOC_ENABLED",
        // "-DLOC_FILE_INDEX=LOC_${fileBasenameNoExtension}_cpp", //${fileExtname/^\.//g}",
        "-DLOGLEVEL=2",
        "${file}",
        "${L3ROOT}/l3.c",
        //"${fileDirname}/loc/loc_filenames.c",
        "${L3ROOT}/l3.S",
        "-o",
        "${workspaceFolder}/${fileBasenameNoExtension}"
    ],
    "options": {
        "cwd": "${workspaceFolder}",
    },
    "problemMatcher": [
        "$gcc"
    ],
    "group": {
        "kind": "build",
        "isDefault": true
    },
    "detail": "compiler: /opt/intel/oneapi/compiler/latest/bin/icpx"
}
```