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
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not ready.exists():
                version.mkdir(parents=True, exist_ok=True)
                print("Building LichtFeld Studio for first use. This can take a long time.", file=sys.stderr)
                print(f"Build log: {cache / 'build.log'}", file=sys.stderr)
                notify("LichtFeld Studio", f"Building application for first use. Log: {cache / 'build.log'}")
                with (cache / "build.log").open("w") as log:
                    def run(*args, cwd=None, env=None):
                        print("Running:", *map(str, args), file=log, flush=True)
                        subprocess.run(list(map(str, args)), cwd=cwd, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)

                    try:
                        if not checkout.exists():
                            shutil.copytree(source, checkout, symlinks=True)
                            make_writable(checkout)
                            shutil.copytree("${libvtermSource}", checkout / "external/libvterm",
                                            dirs_exist_ok=True)
                            make_writable(checkout / "external/libvterm")
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
                        run("cmake", "--build", "--preset", "linux-release", "--parallel", jobs, cwd=checkout)
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
                        run("cmake", "--install", build, "--prefix", payload)
                        check_env = os.environ.copy()
                        check_env["LD_LIBRARY_PATH"] = library_path(payload)
                        run(payload / "bin/LichtFeld-Studio", "--version", env=check_env)
                        run("python3", "-c",
                            f"import sys; sys.path.insert(0, '{payload / 'lib/python'}'); import lichtfeld",
                            env=check_env)
                        ready.write_text("Build and install completed successfully.\n")
                        notify("LichtFeld Studio", "Build complete. Starting application.")
                    except (OSError, subprocess.CalledProcessError) as error:
                        error_msg = f"Build failed: {error}. See {cache / 'build.log'}"
                        print(f"LichtFeld Studio {error_msg}", file=sys.stderr)
                        notify("LichtFeld Studio", error_msg, urgency="critical")
                        sys.exit(1)

        os.environ["LD_LIBRARY_PATH"] = library_path(payload)
        os.environ["CUDA_PATH"] = "${cuda}"
        os.environ["CUDAToolkit_ROOT"] = "${cuda}"
        # The executable locates resources in ../share/LichtFeld-Studio and
        # the installed Python plugins in ../lib/python relative to itself.
        executable = payload / "bin/LichtFeld-Studio"
        os.execv(str(executable), [str(executable), *sys.argv[1:]])
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
      formatter.${system} = pkgs.nixfmt;
    };
}
