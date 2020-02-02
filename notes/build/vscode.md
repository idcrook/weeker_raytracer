# Use Visual Studio code to edit

## VS Code Extensions

- `vscode-clangd`
- `vscode-cudacpp`

## Ubuntu

```shell
# install for clangd
sudo apt-get install clang-tools-9
```

## macOS Catalina

```shell
# install for clangd (this was llvm 9 at the time)
brew install llvm
```

 **(*Code* -> Preferences -> Extensions)**

- `vscode-clangd` gear
  - Clangd: Path

`/usr/local/opt/llvm/bin/clangd`
