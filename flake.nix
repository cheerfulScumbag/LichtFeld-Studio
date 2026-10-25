# SPDX-License-Identifier: GPL-3.0-or-later
{
  description = "LichtFeld Studio application and development environment for NixOS";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # NVIDIA CUDA Toolkit
      };
      # Nixpkgs' merged toolkit omits nvJPEG, which nvImageCodec links statically.
      cuda = pkgs.symlinkJoin {
        name = "lichtfeld-cuda-12.8";
        paths = [
          pkgs.cudaPackages_12_8.cudatoolkit
        ]
        ++ map (output: pkgs.cudaPackages_12_8.libnvjpeg.${output}) [
          "dev"
          "include"
          "lib"
          "static"
        ];
      };
      libvtermArchive = pkgs.fetchurl {
        url = "https://github.com/neovim/libvterm/archive/934bc2fbf21800ac3458a499df8820ca5fb45fd3.tar.gz";
        hash = "sha256-8JUl6yoCZ5vg61C8HClFaejLqktZ+4Z9YGI23igwBF8=";
      };
      libvtermSource = pkgs.runCommand "lichtfeld-libvterm-source" { } ''
        mkdir -p "$out"
        tar xf ${libvtermArchive} --strip-components=1 -C "$out"
      '';
      setupProgress = pkgs.writeText "lichtfeld-setup-progress.py" ''
        import fcntl
        from pathlib import Path
        import sys
        import time

        log_path, lock_path, ready_path = map(Path, sys.argv[1:])
        print("LichtFeld Studio setup", flush=True)
        print("First-time compilation can take a long time.", flush=True)
        print("Closing this window does not cancel setup.", flush=True)
        print(f"Build log: {log_path}\n", flush=True)
        with log_path.open(errors="replace") as log, lock_path.open() as lock:
            while True:
                text = log.read()
                if text:
                    print(text, end="", flush=True)
                try:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(0.1)
                    continue
                fcntl.flock(lock, fcntl.LOCK_UN)
                print(log.read(), end="", flush=True)
                break
        if ready_path.is_file():
            print("\nSetup complete. Starting LichtFeld Studio.", flush=True)
        else:
            print(f"\nSetup did not finish successfully. See {log_path}", flush=True)
            try:
                input("Press Enter to close this window.")
            except (EOFError, KeyboardInterrupt):
                pass
            sys.exit(1)
      '';
      # vcpkg resolves and builds dependencies during configure. Keep that
      # networked build outside the Nix sandbox and inside a versioned cache.
      runtime = pkgs.writeText "lichtfeld-studio-runtime.py" ''
        import fcntl
        import os
        from pathlib import Path
        import shutil
        import subprocess
        import sys

        source = Path("${self}")
        desktop_env = os.environ.copy()
        # Native terminal emulators must not inherit the FHS library overrides
        # or a Python home belonging to the cached application.
        for variable in ("LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH"):
            desktop_env.pop(variable, None)
        cache_base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        if not cache_base.is_absolute():
            cache_base = Path.home() / ".cache"
        cache = cache_base / "lichtfeld-studio"
        version = cache / "${
          builtins.substring 0 20 (builtins.hashString "sha256" (toString self.outPath))
        }"
        checkout = version / "source"
        build = checkout / "build-linux-release"
        payload = version / "install"
        ready = version / ".ready"
        cache.mkdir(parents=True, exist_ok=True)
        vcpkg = cache / "vcpkg"
        os.environ["VCPKG_ROOT"] = str(vcpkg)
        os.environ["XDG_CACHE_HOME"] = str(cache / "dependencies")
        os.environ["CCACHE_DIR"] = str(cache / "ccache")
        os.environ["CUDA_CACHE_PATH"] = str(cache / "cuda")
        binary_cache = cache / "vcpkg-archives"
        binary_cache.mkdir(exist_ok=True)
        os.environ["VCPKG_DEFAULT_BINARY_CACHE"] = str(binary_cache)
        jobs = min(6, os.cpu_count() or 1)

        def make_writable(root):
            for directory, dirs, files in os.walk(root):
                for name in ["."] + dirs + files:
                    path = Path(directory) / name
                    if not path.is_symlink():
                        path.chmod(path.stat().st_mode | 0o200)

        def library_path(prefix):
            paths = [str(prefix / "lib"), str(prefix / "lib/extensions"),
                     "/run/opengl-driver/lib", "/usr/lib"]
            inherited = os.environ.get("LD_LIBRARY_PATH")
            if inherited:
                paths.append(inherited)
            return ":".join(paths)

        def notify(summary, body, urgency="normal"):
            try:
                subprocess.run(
                    ["notify-send", "-a", "LichtFeld Studio", "-u", urgency, summary, body],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        with (cache / "build.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                message = f"Setup is already running. Build log: {cache / 'build.log'}"
                print(message, file=sys.stderr, flush=True)
                notify("LichtFeld Studio", message)
                sys.exit(0)
            if not ready.exists():
                version.mkdir(parents=True, exist_ok=True)
                print("Building LichtFeld Studio for first use. This can take a long time.", file=sys.stderr)
                print(f"Build log: {cache / 'build.log'}", file=sys.stderr)
                notify("LichtFeld Studio", f"Building application for first use. Log: {cache / 'build.log'}")
                with (cache / "build.log").open("w") as log:
                    def emit(message):
                        print(message, file=log, flush=True)
                        print(message, file=sys.stderr, flush=True)

                    def run(*args, cwd=None, env=None):
                        command = list(map(str, args))
                        emit("Running: " + " ".join(command))
                        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True, errors="replace") as process:
                            for line in process.stdout:
                                emit(line.rstrip("\n"))
                            if process.wait() != 0:
                                raise subprocess.CalledProcessError(process.returncode, command)

                    terminal = None
                    try:
                        if not sys.stderr.isatty() and (desktop_env.get("DISPLAY") or desktop_env.get("WAYLAND_DISPLAY")):
                            # xdg-terminal-exec honors the user's configured terminal.
                            subprocess.run(["${pkgs.xdg-terminal-exec}/bin/xdg-terminal-exec", "--print-id"],
                                           env=desktop_env, stdout=subprocess.DEVNULL, check=True, timeout=10)
                            terminal = subprocess.Popen([
                                "${pkgs.xdg-terminal-exec}/bin/xdg-terminal-exec",
                                "--title=LichtFeld Studio setup", "--",
                                "${pkgs.python3.interpreter}", "${setupProgress}",
                                str(cache / "build.log"), str(cache / "build.lock"), str(ready),
                            ], env=desktop_env, stdout=subprocess.DEVNULL, stderr=log, start_new_session=True)
                        emit("\n[1/5] Preparing source")
                        if not checkout.exists():
                            shutil.copytree(source, checkout, symlinks=True)
                            make_writable(checkout)
                            shutil.copytree("${libvtermSource}", checkout / "external/libvterm",
                                            dirs_exist_ok=True)
                            make_writable(checkout / "external/libvterm")
                        emit("\n[2/5] Resolving dependencies")
                        if not vcpkg.exists():
                            run("git", "clone", "https://github.com/microsoft/vcpkg.git", vcpkg)
                        revision = os.environ["LFS_VCPKG_REV"]
                        current = subprocess.check_output(["git", "-C", str(vcpkg), "rev-parse", "HEAD"], text=True).strip()
                        if current != revision:
                            run("git", "-C", vcpkg, "fetch", "origin", revision)
                            run("git", "-C", vcpkg, "checkout", "--detach", revision)
                        if not (vcpkg / "vcpkg").exists():
                            run("bash", vcpkg / "bootstrap-vcpkg.sh", "-disableMetrics")
                        run("cmake", "--preset", "linux-release", "-DBUILD_CUDA_PTX_ONLY=ON",
                            "-DBUILD_CUDA_MIN_SM=75", "-DBUILD_PYTHON_STUBS=OFF",
                            "-DLFS_DEV_IMPORT_SOURCE_PYTHON=OFF", "-DLFS_DEV_IMPORT_SOURCE_RESOURCES=OFF",
                            "-DCMAKE_INSTALL_BINDIR=bin", "-DCMAKE_INSTALL_LIBDIR=lib",
                            "-DCMAKE_INSTALL_DATADIR=share",
                            "-DLFS_DOWNLOAD_CACHE_DIR=" + str(cache / "downloads"), cwd=checkout)
                        emit("\n[3/5] Compiling LichtFeld Studio")
                        run("cmake", "--build", "--preset", "linux-release", "--parallel", jobs, cwd=checkout)
                        emit("\n[4/5] Installing application and runtime libraries")
                        (payload / "lib").mkdir(parents=True, exist_ok=True)
                        # Upstream install targets omit lfs_rmlui and OpenMesh libraries.
                        # Stage all source-built shared libraries before cmake install.
                        for library_dir in [build, build / "Build/lib"]:
                            if library_dir.exists():
                                for library in library_dir.glob("*.so*"):
                                    destination = payload / "lib" / library.name
                                    if library.is_symlink():
                                        if destination.exists() or destination.is_symlink():
                                            destination.unlink()
                                        os.symlink(os.readlink(library), destination)
                                    elif library.is_file():
                                        shutil.copy2(library, destination)
                        # The GUI discovers its Python home through this production
                        # layout. Use this build's interpreter and stdlib, not a
                        # compiled-in prefix retained in a vcpkg binary cache.
                        python_prefix = build / "vcpkg_installed/x64-linux"
                        python_stdlib = python_prefix / "lib/python3.12"
                        python_executable = python_prefix / "tools/python3/python3.12"
                        if not (python_stdlib / "encodings/__init__.py").is_file() or not python_executable.is_file():
                            raise FileNotFoundError("The vcpkg Python 3.12 runtime is incomplete")
                        (payload / "bin").mkdir(parents=True, exist_ok=True)
                        for destination, target in [
                            (payload / "lib/python3.12", python_stdlib),
                            (payload / "bin/python3", python_executable),
                        ]:
                            if destination.is_symlink():
                                destination.unlink()
                            if not destination.exists():
                                destination.symlink_to(os.path.relpath(target, destination.parent),
                                                       target_is_directory=target.is_dir())
                        run("cmake", "--install", build, "--prefix", payload)
                        emit("\n[5/5] Validating Python and native bindings")
                        check_env = os.environ.copy()
                        check_env["LD_LIBRARY_PATH"] = library_path(payload)
                        check_env["PYTHONHOME"] = str(payload)
                        check_env["PYTHONDONTWRITEBYTECODE"] = "1"
                        run(payload / "bin/LichtFeld-Studio", "--version", env=check_env)
                        run(payload / "bin/python3", "-s", "-c",
                            "import sys, importlib.machinery; sys.path.insert(0, sys.argv[1]); "
                            "import encodings, ssl, lichtfeld\n"
                            "if not isinstance(lichtfeld.__loader__, importlib.machinery.ExtensionFileLoader):\n"
                            "    raise RuntimeError('lichtfeld must load its native extension, not a namespace package')\n"
                            "import lfs_plugins; print('Python runtime, native binding, and plugin imports passed')",
                            payload / "lib/python", env=check_env)
                        ready.write_text("Build, install, and matching Python runtime validation completed successfully.\n")
                        emit("\nSetup complete. Starting LichtFeld Studio.")
                        notify("LichtFeld Studio", "Build complete. Starting application.")
                    except (OSError, subprocess.SubprocessError) as error:
                        error_msg = f"Setup failed: {error}. See {cache / 'build.log'}"
                        emit(error_msg)
                        notify("LichtFeld Studio", error_msg, urgency="critical")
                        if terminal is None and sys.stdin.isatty():
                            try:
                                input("Press Enter to close this window.")
                            except (EOFError, KeyboardInterrupt):
                                pass
                        sys.exit(1)
                    finally:
                        if terminal is not None:
                            terminal.poll()

        os.environ["LD_LIBRARY_PATH"] = library_path(payload)
        os.environ["CUDA_PATH"] = "${cuda}"
        os.environ["CUDAToolkit_ROOT"] = "${cuda}"
        # The executable locates resources in ../share/LichtFeld-Studio and
        # the installed Python plugins in ../lib/python relative to itself.
        executable = payload / "bin/LichtFeld-Studio"
        os.execv(str(executable), [str(executable), *sys.argv[1:]])
      '';
      setupTests = pkgs.writeText "lichtfeld-setup-tests.py" ''
        import ast
        import fcntl
        import io
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import threading
        import time
        import unittest
        from unittest.mock import MagicMock, patch

        runtime = Path("${runtime}")
        tree = ast.parse(runtime.read_text())
        key = next(node.value.right.value for node in tree.body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "version" for target in node.targets))

        class SetupTests(unittest.TestCase):
            def test_duplicate_and_cached_launch(self):
                with tempfile.TemporaryDirectory() as directory:
                    env = os.environ.copy()
                    env.update(HOME=directory, XDG_CACHE_HOME=directory)
                    cache = Path(directory) / "lichtfeld-studio"
                    cache.mkdir()
                    log = cache / "build.log"
                    log.write_text("existing build output")
                    with (cache / "build.lock").open("a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        result = subprocess.run([sys.executable, str(runtime)], env=env,
                                                capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Setup is already running", result.stderr)
                    self.assertEqual(log.read_text(), "existing build output")
                    self.assertFalse((cache / key).exists())
                    version = cache / key
                    binary = version / "install/bin/LichtFeld-Studio"
                    binary.parent.mkdir(parents=True)
                    binary.write_text("#!${pkgs.runtimeShell}\nprintf 'GUI_READY %s\\n' \"$1\"\n")
                    binary.chmod(0o755)
                    (version / ".ready").touch()
                    result = subprocess.run([sys.executable, str(runtime), "--help"], env=env,
                                            capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), "GUI_READY --help")
                    self.assertEqual(log.read_text(), "existing build output")

            def test_first_setup_stages_and_terminal_failure(self):
                for missing_terminal in (False, True):
                    with self.subTest(missing_terminal=missing_terminal), tempfile.TemporaryDirectory() as directory:
                        cache = Path(directory) / "lichtfeld-studio"
                        version = cache / key
                        prefix = version / "source/build-linux-release/vcpkg_installed/x64-linux"
                        for name in ("lib/python3.12/encodings/__init__.py", "tools/python3/python3.12"):
                            file = prefix / name
                            file.parent.mkdir(parents=True, exist_ok=True)
                            file.touch()
                        def start_process(*args, **kwargs):
                            process = MagicMock()
                            process.__enter__.return_value = process
                            process.stdout = io.StringIO("[1/1] fixture build output\n")
                            process.wait.return_value = process.returncode = 0
                            return process
                        def run_command(command, **kwargs):
                            if missing_terminal and "--print-id" in command:
                                raise subprocess.CalledProcessError(2, command)
                            return subprocess.CompletedProcess(command, 0)
                        with patch.dict(os.environ, {
                            "HOME": directory, "XDG_CACHE_HOME": directory, "DISPLAY": ":fixture",
                            "LFS_VCPKG_REV": "pinned", "LD_LIBRARY_PATH": "fhs-library-path",
                        }), patch("sys.stderr", io.StringIO()), patch("sys.stdin.isatty", return_value=False), \
                             patch("subprocess.run", side_effect=run_command), \
                             patch("subprocess.check_output", return_value="pinned\n"), \
                             patch("subprocess.Popen", side_effect=start_process) as popen, \
                             patch("os.execv") as launch:
                            if missing_terminal:
                                with self.assertRaises(SystemExit) as failure:
                                    exec(compile(tree, str(runtime), "exec"), {"__name__": "__main__"})
                                self.assertEqual(failure.exception.code, 1)
                                launch.assert_not_called()
                                self.assertFalse((version / ".ready").exists())
                            else:
                                exec(compile(tree, str(runtime), "exec"), {"__name__": "__main__"})
                                launch.assert_called_once()
                                self.assertTrue((version / ".ready").is_file())
                                terminal_call = popen.call_args_list[0]
                                self.assertIn("--title=LichtFeld Studio setup", terminal_call.args[0])
                                self.assertNotIn("LD_LIBRARY_PATH", terminal_call.kwargs["env"])
                                self.assertEqual(terminal_call.kwargs["env"]["XDG_CACHE_HOME"], directory)
                                log = (cache / "build.log").read_text()
                                for stage in range(1, 6):
                                    self.assertIn(f"[{stage}/5]", log)
                                self.assertIn("Setup complete", log)

            def test_live_output_and_command_failure(self):
                functions = [node for node in ast.walk(tree)
                             if isinstance(node, ast.FunctionDef) and node.name in ("emit", "run")]
                log, terminal = io.StringIO(), io.StringIO()
                namespace = {"subprocess": subprocess, "sys": sys, "log": log}
                exec(compile(ast.Module(body=functions, type_ignores=[]), "runtime-functions", "exec"), namespace)
                with tempfile.TemporaryDirectory() as directory, patch("sys.stderr", terminal):
                    gate = Path(directory) / "continue"
                    command = "import pathlib,time; print('[1/' + '2] compiling',flush=True); "
                    command += "\nwhile not pathlib.Path(" + repr(str(gate)) + ").exists(): time.sleep(0.01)"
                    worker = threading.Thread(target=namespace["run"], args=(sys.executable, "-c", command))
                    worker.start()
                    try:
                        deadline = time.monotonic() + 10
                        while "[1/2] compiling" not in terminal.getvalue() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertIn("[1/2] compiling", terminal.getvalue())
                        self.assertIn("[1/2] compiling", log.getvalue())
                        self.assertTrue(worker.is_alive())
                    finally:
                        gate.touch()
                        worker.join(timeout=10)
                    self.assertFalse(worker.is_alive())
                    with self.assertRaises(subprocess.CalledProcessError) as failure:
                        namespace["run"](sys.executable, "-c", "print('compiler error'); raise SystemExit(7)")
                    self.assertEqual(failure.exception.returncode, 7)
                    self.assertIn("compiler error", terminal.getvalue())
                    self.assertIn("compiler error", log.getvalue())

            def test_progress_window_completion_and_failure(self):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    log, lock, ready = [root / name for name in ("build.log", "build.lock", ".ready")]
                    log.write_text("[2/2] linking\n")
                    lock.touch()
                    command = [sys.executable, "${setupProgress}", str(log), str(lock), str(ready)]
                    result = subprocess.run(command, input="\n", capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("Press Enter to close", result.stdout)
                    self.assertIn("[2/2] linking", result.stdout)
                    with lock.open() as held_lock:
                        fcntl.flock(held_lock, fcntl.LOCK_EX)
                        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        try:
                            self.assertIn("LichtFeld Studio setup", process.stdout.readline())
                            self.assertIsNone(process.poll())
                            ready.touch()
                        finally:
                            fcntl.flock(held_lock, fcntl.LOCK_UN)
                        output, errors = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, errors)
                    self.assertIn("Setup complete", output)

        unittest.main()
      '';
      launcher = pkgs.writeShellScript "lichtfeld-studio" ''
        exec ${environment}/bin/lichtfeld-dev -c 'exec python3 "$@"' lichtfeld-studio ${runtime} "$@"
      '';
      desktopItem = pkgs.makeDesktopItem {
        name = "lichtfeld-studio";
        desktopName = "LichtFeld Studio";
        genericName = "3D Gaussian splat editor";
        exec = "lichtfeld-studio";
        icon = "lichtfeld-studio";
        categories = [
          "Graphics"
          "3DGraphics"
          "Science"
        ];
        terminal = false;
      };
      application =
        pkgs.runCommand "lichtfeld-studio"
          {
            nativeBuildInputs = [ pkgs.desktop-file-utils ];
            meta = {
              description = "LichtFeld Studio with a cached source build on first use";
              homepage = "https://lichtfeld.io";
              license = pkgs.lib.licenses.gpl3Plus;
              platforms = [ system ];
              mainProgram = "lichtfeld-studio";
            };
          }
          ''
            mkdir -p "$out/bin" "$out/lib" "$out/share/applications" "$out/share/icons/hicolor/scalable/apps"
            install -m755 ${launcher} "$out/bin/lichtfeld-studio"
            cp ${desktopItem}/share/applications/lichtfeld-studio.desktop "$out/share/applications/"
            cp ${self}/src/visualizer/gui/assets/logo/lichtfeld-logo.svg \
              "$out/share/icons/hicolor/scalable/apps/lichtfeld-studio.svg"
            desktop-file-validate "$out/share/applications/lichtfeld-studio.desktop"
          '';
      environment = pkgs.buildFHSEnv {
        name = "lichtfeld-dev";
        targetPkgs =
          p: with p; [
            gcc14
            gnumake
            cmake
            ninja
            pkg-config
            git
            git-lfs
            curl
            cacert
            zip
            unzip
            which
            file
            patch
            python3
            perl
            ccache
            nasm
            autoconf
            autoconf-archive
            automake
            libtool
            m4
            bison
            flex
            gettext
            gperf
            cuda
            libnotify
            gtk3
            glib
            pango
            harfbuzz
            cairo
            gdk-pixbuf
            atk
            at-spi2-atk
            at-spi2-core
            fontconfig
            freetype
            fribidi
            libepoxy
            pixman
            libthai
            libdatrie
            libffi
            pcre2
            zlib
            bzip2
            brotli
            libpng
            libjpeg
            libtiff
            expat
            graphite2
            libsysprof-capture
            libxft
            libdeflate
            lerc
            zstd
            libwebp
            util-linux
            libselinux
            libsepol
            dbus
            systemd
            libGL
            libGLU
            vulkan-loader
            wayland
            wayland-scanner
            wayland-protocols
            libxkbcommon
            libdecor
            ibus
            libx11
            libxext
            libxcursor
            libxi
            libxinerama
            libxrandr
            libxrender
            libxfixes
            libxdamage
            libxcomposite
            libxtst
            libxau
            libxdmcp
            libxscrnsaver
            libxxf86vm
            libxcb
            xorgproto
            xauth
            xorg-server
          ];
        multiPkgs = null;
        extraOutputsToInstall = [ "dev" ];
        profile = ''
          export PKG_CONFIG_PATH="/usr/lib/pkgconfig:/usr/share/pkgconfig"
          export CC=gcc
          export CXX=g++
          export CUDAHOSTCXX=g++
          export CUDA_PATH=${cuda}
          export CUDAToolkit_ROOT=${cuda}
          export CUDACXX=${cuda}/bin/nvcc
          export VCPKG_ROOT="''${VCPKG_ROOT:-$PWD/vcpkg}"
          export VCPKG_FORCE_SYSTEM_BINARIES=1
          export VCPKG_DISABLE_METRICS=1
          export LFS_VCPKG_REV=817892b0ecbc6af994dd6df999c73faf795adbf2
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:/usr/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
        runScript = "bash";
      };
    in
    {
      devShells.${system}.default = environment.env;
      packages.${system} = {
        default = self.packages.${system}.lichtfeld-studio;
        lichtfeld-studio = application;
        dev-environment = environment;
      };
      apps.${system}.default = {
        type = "app";
        program = "${application}/bin/lichtfeld-studio";
        meta = application.meta;
      };
      checks.${system}.setup = pkgs.runCommand "lichtfeld-setup-checks" { } ''
        ${pkgs.python3.interpreter} ${setupTests}
        touch "$out"
      '';
      formatter.${system} = pkgs.nixfmt;
    };
}
