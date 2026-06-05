use serde::Serialize;
use std::io::Read;
#[cfg(target_os = "windows")]
use std::net::TcpStream;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
#[cfg(target_os = "windows")]
use std::time::{Duration, Instant};
use tauri::Manager;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

fn cmd(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut c = Command::new(program);
    #[cfg(target_os = "windows")]
    c.creation_flags(CREATE_NO_WINDOW);
    c
}

struct ApiProcess(Mutex<Option<Child>>);

#[derive(Clone, Serialize)]
struct SetupPayload {
    step: String,
    message: String,
    progress: u8,
}

struct SetupStatus(Mutex<SetupPayload>);

fn update_status(app: &tauri::AppHandle, step: &str, msg: &str, progress: u8) {
    let state = app.state::<SetupStatus>();
    let mut guard = state.0.lock().unwrap();
    guard.step = step.to_string();
    guard.message = msg.to_string();
    guard.progress = progress;
}

#[tauri::command]
fn get_setup_status(state: tauri::State<SetupStatus>) -> SetupPayload {
    state.0.lock().unwrap().clone()
}

// ── Dev mode (used during `cargo tauri dev`) ────────────────────

fn find_project_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_default();
    let candidates = [
        cwd.join("gladiator_api.py"),
        cwd.join("../gladiator_api.py"),
        cwd.join("../../gladiator_api.py"),
    ];
    for p in &candidates {
        if p.exists() {
            return p.parent().unwrap_or(&cwd).to_path_buf();
        }
    }
    cwd
}

fn find_python() -> String {
    let root = find_project_root();
    let venv_python = root.join("gui").join(".venv").join("bin").join("python3");
    if venv_python.exists() {
        return venv_python.to_string_lossy().to_string();
    }
    for name in &["python3", "python"] {
        if Command::new(name)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return name.to_string();
        }
    }
    "python3".to_string()
}

fn kill_port(port: u16) {
    if cfg!(target_os = "windows") {
        cmd("powershell")
            .args([
                "-Command",
                &format!("Get-Process -Id (Get-NetTCPConnection -LocalPort {}).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force", port),
            ])
            .status()
            .ok();
    } else {
        Command::new("sh")
            .arg("-c")
            .arg(format!("fuser -k {}/tcp 2>/dev/null; exit 0", port))
            .status()
            .ok();
    }
}

fn start_dev_server(app_handle: &tauri::AppHandle) {
    let root = find_project_root();
    let python = find_python();
    let script = root.join("gladiator_api.py");

    eprintln!(
        "[gladiator-gui] Starting API server: {} {}",
        python,
        script.display()
    );

    kill_port(8742);

    let child = cmd(&python)
        .arg(&script)
        .arg("8742")
        .spawn()
        .expect("Failed to start Python API server");

    *app_handle.state::<ApiProcess>().0.lock().unwrap() = Some(child);

    update_status(app_handle, "ready", "Ready!", 100);
}

// ── Production mode (used in release builds) ────────────────────

fn data_dir() -> PathBuf {
    let base = if cfg!(target_os = "windows") {
        std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    } else if cfg!(target_os = "macos") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home)
            .join("Library")
            .join("Application Support")
    } else {
        std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                PathBuf::from(home).join(".local").join("share")
            })
    };
    base.join("gladiator")
}

fn download_file(url: &str, dest: &PathBuf, app: &tauri::AppHandle) -> bool {
    let _ = std::fs::remove_file(dest);

    let tmp = dest.with_extension("tmp");
    let _ = std::fs::remove_file(&tmp);

    let result = (|| -> Result<(), String> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(600))
            .build();

        let response = agent
            .get(url)
            .call()
            .map_err(|e| format!("Download failed: {e}"))?;

        let mut out =
            std::fs::File::create(&tmp).map_err(|e| format!("Failed to create temp file: {e}"))?;

        let mut reader = response.into_reader();
        std::io::copy(&mut reader, &mut out)
            .map_err(|e| format!("Failed to write response to file: {e}"))?;

        drop(out);

        let len = std::fs::metadata(&tmp)
            .map_err(|e| format!("Failed to stat temp file: {e}"))?
            .len();

        if len <= 1_000_000 {
            return Err(format!(
                "Downloaded backend is too small ({len} bytes). The release asset may be missing or corrupted."
            ));
        }

        std::fs::rename(&tmp, dest).map_err(|e| format!("Failed to rename temp file: {e}"))?;

        Ok(())
    })();

    match result {
        Ok(()) => true,
        Err(msg) => {
            update_status(app, "error", &msg, 0);
            let _ = std::fs::remove_file(&tmp);
            false
        }
    }
}

// ── Windows: download and run pre-built PyInstaller binary ──────

#[cfg(target_os = "windows")]
fn backend_binary_name() -> &'static str {
    "gladiator-backend-x86_64-windows.exe"
}

#[cfg(target_os = "windows")]
fn backend_download_url(version: &str) -> String {
    format!(
        "https://github.com/Nicholas-Wilkins/Gladiator/releases/download/v{}/{}",
        version,
        backend_binary_name()
    )
}

#[cfg(target_os = "windows")]
fn is_backend_installed(dir: &PathBuf) -> bool {
    let installed = dir.join(".installed");
    if !installed.exists() {
        return false;
    }
    if let Ok(stored) = std::fs::read_to_string(&installed) {
        if stored.trim() != env!("CARGO_PKG_VERSION") {
            return false;
        }
    }
    let backend_path = dir.join("backend").join(backend_binary_name());
    backend_path.exists()
        && std::fs::metadata(&backend_path)
            .map(|m| m.len() > 1_000_000)
            .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn install_backend(dir: &PathBuf, app: &tauri::AppHandle) -> bool {
    eprintln!("[gladiator-gui] Installing backend to {}", dir.display());

    let backend_dir = dir.join("backend");
    std::fs::create_dir_all(&backend_dir).ok();

    update_status(
        app,
        "downloading",
        "Downloading backend from GitHub\u{2026}",
        5,
    );

    let binary_name = backend_binary_name();
    let binary_path = backend_dir.join(binary_name);
    let url = backend_download_url(env!("CARGO_PKG_VERSION"));

    if !download_file(&url, &binary_path, app) {
        return false;
    }

    if std::fs::write(dir.join(".installed"), env!("CARGO_PKG_VERSION")).is_err() {
        update_status(app, "error", "Failed to create installation marker.", 0);
        return false;
    }
    eprintln!("[gladiator-gui] Backend installation complete!");
    true
}

#[cfg(target_os = "windows")]
fn start_production_server(app_handle: &tauri::AppHandle) {
    let dir = data_dir();
    let binary_path = dir.join("backend").join(backend_binary_name());

    update_status(
        app_handle,
        "starting",
        "Starting backend server\u{2026}",
        95,
    );

    if !binary_path.exists() {
        let msg = format!(
            "Backend binary not found at {}. Try reinstalling.",
            binary_path.display()
        );
        update_status(app_handle, "error", &msg, 0);
        return;
    }

    kill_port(8742);

    let mut child = match cmd(&binary_path.to_string_lossy())
        .arg("8742")
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("Failed to start backend: {}", e);
            update_status(app_handle, "error", &msg, 0);
            return;
        }
    };

    let deadline = Instant::now() + Duration::from_secs(120);
    let addr = "127.0.0.1:8742";
    let mut ready = false;

    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code().map(|c| c.to_string()).unwrap_or_default();
                let mut stderr_buf = String::new();
                if let Some(mut stderr_pipe) = child.stderr.take() {
                    let _ = stderr_pipe.read_to_string(&mut stderr_buf);
                }
                let stderr = stderr_buf.trim();
                if stderr.is_empty() {
                    update_status(
                        app_handle,
                        "error",
                        &format!("Backend exited immediately (code {})", code),
                        0,
                    );
                } else {
                    update_status(
                        app_handle,
                        "error",
                        &format!("Backend exited immediately (code {}):\n{}", code, stderr),
                        0,
                    );
                }
                return;
            }
            Ok(None) => {
                if let Ok(stream) =
                    TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(500))
                {
                    drop(stream);
                    ready = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(e) => {
                update_status(
                    app_handle,
                    "error",
                    &format!("Process check failed: {}", e),
                    0,
                );
                return;
            }
        }
    }

    if ready {
        *app_handle.state::<ApiProcess>().0.lock().unwrap() = Some(child);
        update_status(app_handle, "ready", "Ready!", 100);
    } else {
        let mut stderr_buf = String::new();
        if let Some(mut stderr_pipe) = child.stderr.take() {
            let _ = stderr_pipe.read_to_string(&mut stderr_buf);
        }
        let _ = child.kill();
        let _ = child.wait();
        let stderr = stderr_buf.trim();
        if stderr.is_empty() {
            update_status(
                app_handle,
                "error",
                "Backend did not start within 2 minutes. Check antivirus or try reinstalling.",
                0,
            );
        } else {
            update_status(
                app_handle,
                "error",
                &format!("Backend failed to start:\n{}", stderr),
                0,
            );
        }
    }
}

// ── Linux/macOS: venv/pip approach ──────────────────────────────

#[cfg(not(target_os = "windows"))]
fn venv_python(venv_dir: &PathBuf) -> PathBuf {
    let bin = venv_dir.join("bin");
    let candidates = ["python3", "python"];
    for name in &candidates {
        let p = bin.join(name);
        if p.exists() {
            return p;
        }
    }
    if let Ok(entries) = std::fs::read_dir(&bin) {
        let mut pythons: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                s.starts_with("python") || s.starts_with("python3")
            })
            .map(|e| e.path())
            .collect();
        pythons.sort();
        if let Some(p) = pythons.first() {
            return p.clone();
        }
    }
    venv_dir.join("bin").join("python3")
}

#[cfg(not(target_os = "windows"))]
fn venv_pip(venv_dir: &PathBuf) -> PathBuf {
    venv_dir.join("bin").join("pip")
}

#[cfg(not(target_os = "windows"))]
fn python_stdlib_ok(python: &PathBuf) -> bool {
    Command::new(python)
        .args(["-c", "import encodings"])
        .env_remove("PYTHONHOME")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(not(target_os = "windows"))]
fn python_name() -> String {
    let py3 = PathBuf::from("python3");
    if python_stdlib_ok(&py3) {
        return "python3".to_string();
    }
    let py = PathBuf::from("python");
    if python_stdlib_ok(&py) {
        return "python".to_string();
    }
    "python3".to_string()
}

#[cfg(not(target_os = "windows"))]
fn pip_install(venv_dir: &PathBuf, pkgs: &[&str]) -> Result<String, String> {
    let mut cmd = Command::new(&venv_pip(venv_dir));
    cmd.args(["install"]);
    cmd.args(pkgs);
    cmd.env_remove("PYTHONHOME");
    let out = cmd
        .output()
        .map_err(|e| format!("Failed to run pip: {}", e))?;
    if out.status.success() {
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        Ok(text)
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(stderr.trim().to_string())
    }
}

#[cfg(not(target_os = "windows"))]
fn can_import(python: &PathBuf, module: &str) -> bool {
    Command::new(python)
        .args(["-c", &format!("import {}", module)])
        .env_remove("PYTHONHOME")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(not(target_os = "windows"))]
fn extract_zip(zip_path: &PathBuf, dest: &PathBuf, app: &tauri::AppHandle) -> bool {
    let result = Command::new("unzip")
        .args([
            "-o",
            &zip_path.to_string_lossy(),
            "-d",
            &dest.to_string_lossy(),
        ])
        .output();

    match result {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            update_status(
                app,
                "error",
                &format!("Extraction failed:\n{}", stderr.trim()),
                0,
            );
            false
        }
        Err(e) => {
            update_status(app, "error", &format!("Failed to run unzip: {}", e), 0);
            false
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn copy_dir(src: &PathBuf, dst: &PathBuf) -> Result<(), String> {
    let _ = std::fs::remove_dir_all(dst);
    std::fs::create_dir_all(dst).map_err(|e| format!("create_dir_all: {}", e))?;
    let entries = std::fs::read_dir(src).map_err(|e| format!("read_dir: {}", e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("entry: {}", e))?;
        let ty = entry.file_type().map_err(|e| format!("file_type: {}", e))?;
        let name = entry.file_name();
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        if ty.is_dir() {
            copy_dir(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)
                .map_err(|e| format!("copy {}: {}", name.to_string_lossy(), e))?;
        }
    }
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn find_venv_site_packages(venv_dir: &PathBuf) -> Option<PathBuf> {
    let lib = venv_dir.join("lib");
    let ok = std::fs::read_dir(&lib).ok()?;
    let py_ver = ok
        .filter_map(|e| e.ok())
        .map(|e| e.file_name())
        .find(|n| n.to_string_lossy().starts_with("python3"))?;
    let sp = lib.join(&py_ver).join("site-packages");
    if sp.exists() {
        Some(sp)
    } else {
        None
    }
}

#[cfg(not(target_os = "windows"))]
fn try_start_server(
    bin: &PathBuf,
    script: &PathBuf,
    site_packages: &Option<PathBuf>,
) -> Result<Child, String> {
    kill_port(8742);
    let mut cmd = Command::new(bin);
    cmd.arg(script).arg("8742");
    cmd.stderr(Stdio::piped());
    if let Some(sp) = site_packages {
        let existing = std::env::var("PYTHONPATH").unwrap_or_default();
        let combined = if existing.is_empty() {
            sp.to_string_lossy().to_string()
        } else {
            format!("{}:{}", sp.display(), existing)
        };
        cmd.env("PYTHONPATH", combined);
    }
    cmd.env_remove("PYTHONHOME");
    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn: {}", e))?;
    std::thread::sleep(std::time::Duration::from_millis(400));
    match child.try_wait() {
        Ok(Some(status)) => {
            let code = status.code().map(|c| c.to_string()).unwrap_or_default();
            let mut stderr_buf = String::new();
            if let Some(mut stderr_pipe) = child.stderr.take() {
                let _ = stderr_pipe.read_to_string(&mut stderr_buf);
            }
            let stderr = stderr_buf.trim();
            if stderr.is_empty() {
                Err(format!("exited immediately (code {})", code))
            } else {
                Err(format!("exited immediately (code {}):\n{}", code, stderr))
            }
        }
        Ok(None) => Ok(child),
        Err(e) => Err(format!("process check failed: {}", e)),
    }
}

#[cfg(not(target_os = "windows"))]
fn preserve_user_data(src: &PathBuf, dst: &PathBuf) {
    let _ = std::fs::create_dir_all(dst);
    if let Ok(entries) = std::fs::read_dir(src) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "db") {
                let _ = std::fs::copy(&path, dst.join(path.file_name().unwrap()));
            }
        }
    }
    let exported = src.join("exported_bots");
    if exported.exists() {
        let _ = copy_dir(&exported, &dst.join("exported_bots"));
    }
}

#[cfg(not(target_os = "windows"))]
fn try_restore_user_data(backup: &PathBuf, dest: &PathBuf) {
    if backup.exists() {
        let _ = std::fs::create_dir_all(dest);
        preserve_user_data(backup, dest);
        let _ = std::fs::remove_dir_all(backup);
    }
}

#[cfg(not(target_os = "windows"))]
fn is_backend_installed(dir: &PathBuf) -> bool {
    let installed = dir.join(".installed");
    if !installed.exists() {
        return false;
    }
    if let Ok(stored) = std::fs::read_to_string(&installed) {
        if stored.trim() != env!("CARGO_PKG_VERSION") {
            return false;
        }
    }
    let py = venv_python(&dir.join(".venv"));
    if !py.exists() {
        return false;
    }
    can_import(&py, "fastapi")
}

#[cfg(not(target_os = "windows"))]
fn install_backend(dir: &PathBuf, app: &tauri::AppHandle) -> bool {
    eprintln!("[gladiator-gui] Installing backend to {}", dir.display());

    let backup = std::env::temp_dir().join("gladiator_preserve");
    let _ = std::fs::remove_dir_all(&backup);
    let old_backend = dir.join("backend");
    let had_data = old_backend.exists();
    if had_data {
        preserve_user_data(&old_backend, &backup);
    }

    update_status(
        app,
        "downloading",
        "Downloading backend from GitHub\u{2026}",
        5,
    );

    let zip_path = dir.join("repo.zip");
    let url = "https://github.com/Nicholas-Wilkins/Gladiator/archive/refs/heads/main.zip";
    if !download_file(url, &zip_path, app) {
        try_restore_user_data(&backup, &old_backend);
        return false;
    }

    update_status(app, "extracting", "Extracting backend\u{2026}", 15);
    let tmp_extract = std::env::temp_dir().join("gladiator_extract");
    let _ = std::fs::remove_dir_all(&tmp_extract);
    let _ = std::fs::create_dir_all(&tmp_extract);
    if !extract_zip(&zip_path, &tmp_extract, app) {
        try_restore_user_data(&backup, &old_backend);
        let _ = std::fs::remove_dir_all(&tmp_extract);
        return false;
    }

    let _ = std::fs::remove_dir_all(dir);
    if std::fs::create_dir_all(dir).is_err() {
        try_restore_user_data(&backup, &old_backend);
        return false;
    }

    let extracted = tmp_extract.join("Gladiator-main");
    let backend = dir.join("backend");
    if let Err(e) = std::fs::rename(&extracted, &backend) {
        match copy_dir(&extracted, &backend) {
            Ok(()) => {
                let _ = std::fs::remove_dir_all(&extracted);
            }
            Err(copy_err) => {
                try_restore_user_data(&backup, &backend);
                update_status(
                    app,
                    "error",
                    &format!(
                        "Failed to move backend files:\nrename: {}\ncopy: {}",
                        e, copy_err
                    ),
                    0,
                );
                return false;
            }
        }
    }
    let _ = std::fs::remove_file(&zip_path);
    let _ = std::fs::remove_dir_all(&tmp_extract);
    try_restore_user_data(&backup, &backend);

    update_status(app, "venv", "Setting up Python environment\u{2026}", 30);
    let venv_dir = dir.join(".venv");
    let py_name = python_name();
    let py_path = PathBuf::from(&py_name);
    if !python_stdlib_ok(&py_path) {
        update_status(
            app,
            "error",
            &format!(
                "Python was not found. Make sure python3 is installed and on the PATH.\n\
                 Tried: '{}', 'python3', 'python'",
                py_name
            ),
            0,
        );
        return false;
    }
    let mut venv_cmd = Command::new(&py_name);
    venv_cmd.args(["-m", "venv", &venv_dir.to_string_lossy()]);
    venv_cmd.env_remove("PYTHONHOME");
    match venv_cmd.output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            update_status(
                app,
                "error",
                &format!(
                    "Failed to create Python virtual environment:\n{}",
                    stderr.trim()
                ),
                0,
            );
            return false;
        }
        Err(e) => {
            update_status(
                app,
                "error",
                &format!("Failed to run Python ({}): {}", py_name, e),
                0,
            );
            return false;
        }
    }

    let python_bin = venv_python(&venv_dir);
    if !python_bin.exists() {
        let bin_dir = venv_dir.join("bin");
        let contents = match std::fs::read_dir(&bin_dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
                .collect::<Vec<_>>()
                .join(", "),
            Err(_) => "directory not found".to_string(),
        };
        update_status(
            app,
            "error",
            &format!(
                "Virtual environment created but no Python binary found in {}.\n\
                 Contents: {}",
                bin_dir.display(),
                contents
            ),
            0,
        );
        return false;
    }

    update_status(app, "deps", "Installing Python packages\u{2026}", 40);
    let req_path = backend.join("requirements.txt");
    let mut pip_req = Command::new(&venv_pip(&venv_dir));
    pip_req.args(["install", "-r", &req_path.to_string_lossy()]);
    pip_req.env_remove("PYTHONHOME");
    match pip_req.output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            update_status(
                app,
                "error",
                &format!("Failed to install Python packages:\n{}", stderr.trim()),
                0,
            );
            return false;
        }
        Err(e) => {
            update_status(app, "error", &format!("Failed to run pip: {}", e), 0);
            return false;
        }
    }

    update_status(
        app,
        "torch",
        "Detecting GPU and installing PyTorch\u{2026}",
        65,
    );
    let mut torch_cmd = Command::new(&venv_python(&venv_dir));
    torch_cmd.arg(backend.join("install.py"));
    torch_cmd.current_dir(&backend);
    torch_cmd.env_remove("PYTHONHOME");
    match torch_cmd.output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let combined = format!("{}\n{}", stdout.trim(), stderr.trim());
            let msg = if combined.trim().is_empty() {
                "PyTorch installation failed with no error output.".to_string()
            } else {
                format!("PyTorch installation failed:\n{}", combined.trim())
            };
            update_status(app, "error", &msg, 0);
            return false;
        }
        Err(e) => {
            update_status(
                app,
                "error",
                &format!("Failed to run PyTorch installer: {}", e),
                0,
            );
            return false;
        }
    }

    if std::fs::write(dir.join(".installed"), env!("CARGO_PKG_VERSION")).is_err() {
        update_status(app, "error", "Failed to create installation marker.", 0);
        return false;
    }
    eprintln!("[gladiator-gui] Backend installation complete!");
    true
}

#[cfg(not(target_os = "windows"))]
fn start_production_server(app_handle: &tauri::AppHandle) {
    let dir = data_dir();
    let venv_dir = dir.join(".venv");
    let script = dir.join("backend").join("gladiator_api.py");

    update_status(
        app_handle,
        "starting",
        "Starting backend server\u{2026}",
        95,
    );

    let venv_py = venv_python(&venv_dir);
    let system_py = PathBuf::from(python_name());
    let mut sp = find_venv_site_packages(&venv_dir);

    let venv_ok = venv_py.exists() && python_stdlib_ok(&venv_py);
    let sys_ok = python_stdlib_ok(&system_py);

    let (bin, label) = if venv_ok {
        (venv_py, "venv")
    } else if sys_ok {
        eprintln!("[gladiator-gui] venv Python broken (stdlib check failed), using system python");
        (system_py, "system")
    } else {
        let msg = "No working Python interpreter found. Make sure Python 3 is installed and its standard library is intact.".to_string();
        update_status(app_handle, "error", &msg, 0);
        return;
    };

    let site_pkgs = if label == "system" { sp.take() } else { None };

    let python = &bin;
    for mod_name in &["fastapi"] {
        if !can_import(python, mod_name) {
            eprintln!("[gladiator-gui] {} not importable, installing...", mod_name);
            update_status(
                app_handle,
                "deps",
                "Installing missing Python packages\u{2026}",
                85,
            );

            let installed = if venv_dir.join("bin").join("pip").exists() {
                pip_install(&venv_dir, &["fastapi", "uvicorn[standard]"]).is_ok()
            } else {
                false
            };
            if !installed {
                let sys_py = PathBuf::from(python_name());
                let mut pip_cmd = Command::new(&sys_py);
                pip_cmd.args(["-m", "pip", "install", "fastapi", "uvicorn[standard]"]);
                pip_cmd.env_remove("PYTHONHOME");
                if label == "venv" {
                    if let Some(sp) = find_venv_site_packages(&venv_dir) {
                        pip_cmd.arg("--target");
                        pip_cmd.arg(sp.to_string_lossy().to_string());
                    }
                }
                let result = pip_cmd.output();
                match result {
                    Ok(out) if out.status.success() => {}
                    Ok(out) => {
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        let msg = format!("Failed to install fastapi/uvicorn:\n{}", stderr.trim());
                        update_status(app_handle, "error", &msg, 0);
                        return;
                    }
                    Err(e) => {
                        let msg = format!("Failed to run pip: {}", e);
                        update_status(app_handle, "error", &msg, 0);
                        return;
                    }
                }
            }
            if !can_import(python, mod_name) {
                let msg = format!(
                    "{} still not importable after install. Try running:\n  pip install fastapi uvicorn[standard]",
                    mod_name
                );
                update_status(app_handle, "error", &msg, 0);
                return;
            }
        }
    }

    eprintln!(
        "[gladiator-gui] Starting server with {}: {}",
        label,
        python.display()
    );
    match try_start_server(python, &script, &site_pkgs) {
        Ok(child) => {
            *app_handle.state::<ApiProcess>().0.lock().unwrap() = Some(child);
            update_status(app_handle, "ready", "Ready!", 100);
        }
        Err(e) => {
            let _ = std::fs::remove_file(dir.join(".installed"));
            let msg = format!("Python server {}: {}", e, python.display());
            update_status(app_handle, "error", &msg, 0);
        }
    }
}

// ── Entry point ─────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            app.manage(ApiProcess(Mutex::new(None)));
            app.manage(SetupStatus(Mutex::new(SetupPayload {
                step: "init".to_string(),
                message: "Preparing dependencies\u{2026}".to_string(),
                progress: 0,
            })));
            let handle = app.handle().clone();

            if cfg!(debug_assertions) {
                start_dev_server(&handle);
                std::thread::sleep(std::time::Duration::from_secs(2));
            } else {
                std::thread::spawn(move || {
                    let dir = data_dir();
                    if !is_backend_installed(&dir) {
                        if !install_backend(&dir, &handle) {
                            return;
                        }
                    }
                    start_production_server(&handle);
                });
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.try_state::<ApiProcess>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(mut child) = guard.take() {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![get_setup_status])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
