
# Debugging


```shell
sudo apt install cmake-curses-gui
ccmake -B build
```

Set `CMAKE_BUILD_TYPE` to `Debug` and re-generate
Default value Empty/`Release`

Now build again with desired debug target

`cmake --build build --target theNextWeekOptix`


## Using gdb (cuda-gdb) in emacs

```shell
$ cat .cuda-gdbinit
cwd /path/top/top/dir
```

using the `cuda-gdb` for `gdb` in emacs

```lisp
(setq gud-gdb-command-name "cuda-gdb -i=mi --args ")
```

Invoke `gdb` in emacs using <kbd>M-x</kbd>`gdb`

add program to execute at end of prompt:

`cuda-gdb -i=mi --args` **`build/theNextWeekOptix`**

### Emacs gdb mode to inspect segfault stack trace, etc.

`gdb>`<kbd>r</kbd>

the `r` runs the program. If no breakpoints are set, and there is a segfault termination, it will trap to the debugger at that stack frame.

`gdb>`<kbd>C-c &lt;</kbd>    `# goes up one level in stack`

Repeat until you get to user code. Now inspect other state

`gdb>`<kbd>p var1</kbd>, etc.

`gdb>`<kbd>info locals</kbd>, etc.
