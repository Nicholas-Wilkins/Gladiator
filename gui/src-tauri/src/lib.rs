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
use tauri_plugin_updater::UpdaterExt;

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

#[tauri::command]
fn get_pending_update_notes() -> Option<serde_json::Value> {
    let path = data_dir().join("last_update.json");
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&path).ok()?;
    let notes: serde_json::Value = serde_json::from_str(&content).ok()?;
    let _ = std::fs::remove_file(&path);
    Some(notes)
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

        // Try rename first (atomic); fall back to copy+delete if the
        // file is locked by antivirus scanning (common on Windows).
        if std::fs::rename(&tmp, dest).is_err() {
            std::fs::copy(&tmp, dest).map_err(|e| format!("Failed to copy temp file: {e}"))?;
            let _ = std::fs::remove_file(&tmp);
        }

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

// ── Windows: venv/pip from bundled source ZIP ──────────────────

#[cfg(target_os = "windows")]
fn venv_python(venv_dir: &PathBuf) -> PathBuf {
    venv_dir.join("Scripts").join("python.exe")
}

#[cfg(target_os = "windows")]
fn python_stdlib_ok(python: &PathBuf) -> bool {
    cmd(&python.to_string_lossy())
        .args(["-c", "import encodings"])
        .env_remove("PYTHONHOME")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn can_import(python: &PathBuf, module: &str) -> bool {
    cmd(&python.to_string_lossy())
        .args(["-c", &format!("import {}", module)])
        .env_remove("PYTHONHOME")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn extract_zip(zip_path: &PathBuf, dest: &PathBuf, app: &tauri::AppHandle) -> bool {
    let _ = std::fs::create_dir_all(dest);
    let result = cmd("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                zip_path.to_string_lossy(),
                dest.to_string_lossy()
            ),
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
            update_status(
                app,
                "error",
                &format!("Failed to run PowerShell Expand-Archive: {}", e),
                0,
            );
            false
        }
    }
}

#[cfg(target_os = "windows")]
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

#[cfg(target_os = "windows")]
fn try_start_server(bin: &PathBuf, script: &PathBuf) -> Result<Child, String> {
    kill_port(8742);
    let mut c = cmd(&bin.to_string_lossy());
    c.arg(script).arg("8742");
    c.stderr(Stdio::piped());
    c.env_remove("PYTHONHOME");
    let mut child = c.spawn().map_err(|e| format!("Failed to spawn: {}", e))?;
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

#[cfg(target_os = "windows")]
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

#[cfg(target_os = "windows")]
fn try_restore_user_data(backup: &PathBuf, dest: &PathBuf) {
    if backup.exists() {
        let _ = std::fs::create_dir_all(dest);
        preserve_user_data(backup, dest);
        let _ = std::fs::remove_dir_all(backup);
    }
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
    let py = venv_python(&dir.join(".venv"));
    py.exists() && can_import(&py, "fastapi")
}

#[cfg(target_os = "windows")]
fn install_backend(dir: &PathBuf, app: &tauri::AppHandle) -> bool {
    eprintln!("[gladiator-gui] Installing backend to {}", dir.display());

    if let Err(e) = std::fs::create_dir_all(dir) {
        update_status(
            app,
            "error",
            &format!("Failed to create data directory: {}", e),
            0,
        );
        return false;
    }

    let backup = std::env::temp_dir().join("gladiator_preserve");
    let _ = std::fs::remove_dir_all(&backup);
    let old_backend = dir.join("backend");
    let had_data = old_backend.exists();
    if had_data {
        preserve_user_data(&old_backend, &backup);
    }

    update_status(app, "installing", "Installing backend\u{2026}", 5);

    let zip_name = "gladiator-backend-windows.zip";
    let zip_path = dir.join(&zip_name);

    #[cfg(has_backend_zip)]
    {
        let data = include_bytes!("../backend/gladiator-backend-windows.zip");
        if let Err(e) = std::fs::write(&zip_path, data) {
            update_status(
                app,
                "error",
                &format!("Failed to write embedded backend: {}", e),
                0,
            );
            try_restore_user_data(&backup, &old_backend);
            return false;
        }
    }
    #[cfg(not(has_backend_zip))]
    {
        let resource_path = app
            .path()
            .resource_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("backend")
            .join(zip_name);

        if !resource_path.exists() {
            update_status(
                app,
                "error",
                &format!(
                    "Backend bundle not found at {}. The installer may be corrupted.",
                    resource_path.display()
                ),
                0,
            );
            try_restore_user_data(&backup, &old_backend);
            return false;
        }

        let mut last_err = String::new();
        for attempt in 1..=10 {
            match std::fs::OpenOptions::new().read(true).open(&resource_path) {
                Ok(f) => drop(f),
                Err(e) => {
                    last_err = format!("{}", e);
                    eprintln!(
                        "[gladiator-gui] Resource not accessible (attempt {}/10): {}",
                        attempt, last_err
                    );
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    continue;
                }
            }
            match std::fs::copy(&resource_path, &zip_path) {
                Ok(_) => {
                    last_err.clear();
                    break;
                }
                Err(e) => {
                    last_err = format!("{}", e);
                    eprintln!(
                        "[gladiator-gui] Copy attempt {}/10 failed: {}",
                        attempt, last_err
                    );
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
        if !last_err.is_empty() {
            update_status(
                app,
                "error",
                &format!(
                    "Failed to copy backend bundle from installer resources: {}",
                    last_err
                ),
                0,
            );
            try_restore_user_data(&backup, &old_backend);
            return false;
        }
    }

    // Extract the source bundle into the backend directory.
    update_status(app, "extracting", "Extracting backend\u{2026}", 15);
    let backend_dir = dir.join("backend");
    let _ = std::fs::remove_dir_all(&backend_dir);
    if !extract_zip(&zip_path, &backend_dir, app) {
        try_restore_user_data(&backup, &old_backend);
        return false;
    }
    let _ = std::fs::remove_file(&zip_path);
    try_restore_user_data(&backup, &backend_dir);

    let uv_path = dir.join("uv.exe");
    #[cfg(has_uv_exe)]
    {
        let data = include_bytes!("../backend/uv.exe");
        if let Err(e) = std::fs::write(&uv_path, data) {
            update_status(
                app,
                "error",
                &format!("Failed to write embedded uv.exe: {}", e),
                0,
            );
            try_restore_user_data(&backup, &old_backend);
            return false;
        }
    }
    #[cfg(not(has_uv_exe))]
    {
        let uv_resource = app
            .path()
            .resource_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("backend")
            .join("uv.exe");
        if !uv_resource.exists() {
            update_status(
                app,
                "error",
                &format!(
                    "uv.exe not found at {}. The installer may be corrupted.",
                    uv_resource.display()
                ),
                0,
            );
            try_restore_user_data(&backup, &old_backend);
            return false;
        }
        let mut last_err = String::new();
        for attempt in 1..=10 {
            match std::fs::OpenOptions::new().read(true).open(&uv_resource) {
                Ok(f) => drop(f),
                Err(e) => {
                    last_err = format!("{}", e);
                    eprintln!(
                        "[gladiator-gui] uv.exe not accessible (attempt {}/10): {}",
                        attempt, last_err
                    );
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    continue;
                }
            }
            match std::fs::copy(&uv_resource, &uv_path) {
                Ok(_) => {
                    last_err.clear();
                    break;
                }
                Err(e) => {
                    last_err = format!("{}", e);
                    eprintln!(
                        "[gladiator-gui] uv.exe copy attempt {}/10 failed: {}",
                        attempt, last_err
                    );
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
        if !last_err.is_empty() {
            update_status(
                app,
                "error",
                &format!("Failed to copy uv.exe from resources: {}", last_err),
                0,
            );
            try_restore_user_data(&backup, &old_backend);
            return false;
        }
    }

    // Create virtual environment with uv (downloads Python if needed).
    update_status(app, "venv", "Creating virtual environment\u{2026}", 40);
    let venv_dir = dir.join(".venv");
    if venv_dir.exists() {
        let old = dir.join(".venv.old");
        let _ = std::fs::remove_dir_all(&old);
        if std::fs::rename(&venv_dir, &old).is_err() {
            let _ = std::fs::remove_dir_all(&venv_dir);
        }
    }
    let mut venv_cmd = cmd(&uv_path.to_string_lossy());
    venv_cmd.args([
        "venv",
        "--seed",
        "--python",
        "3.11",
        &venv_dir.to_string_lossy(),
    ]);
    venv_cmd.current_dir(&dir);
    venv_cmd.env_remove("PYTHONHOME");
    match venv_cmd.output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            update_status(
                app,
                "error",
                &format!(
                    "Failed to create virtual environment with uv:\n{}",
                    stderr.trim()
                ),
                0,
            );
            try_restore_user_data(&backup, &old_backend);
            return false;
        }
        Err(e) => {
            update_status(app, "error", &format!("Failed to run uv: {}", e), 0);
            try_restore_user_data(&backup, &old_backend);
            return false;
        }
    }
    let python_bin = venv_python(&venv_dir);
    if !python_bin.exists() || !python_stdlib_ok(&python_bin) {
        update_status(
            app,
            "error",
            "uv created the virtual environment but the Python interpreter is not functional.",
            0,
        );
        try_restore_user_data(&backup, &old_backend);
        return false;
    }

    // Install pip dependencies from requirements.txt with uv.
    update_status(app, "deps", "Installing Python packages\u{2026}", 60);

    // Clear uv's distribution cache to avoid Windows untrusted mount point errors
    // (os error 448) that occur when the app has been moved by an installer update.
    let mut cache_clean = cmd(&uv_path.to_string_lossy());
    cache_clean.args(["cache", "clean"]);
    let _ = cache_clean.output();
    // Use a fresh cache dir inside our data dir so we don't inherit stale paths.
    let uv_cache_dir = dir.join(".uv-cache");
    let _ = std::fs::create_dir_all(&uv_cache_dir);

    let req_path = backend_dir.join("requirements.txt");
    let mut pip_cmd = cmd(&uv_path.to_string_lossy());
    pip_cmd.args(["pip", "install", "-r", &req_path.to_string_lossy()]);
    pip_cmd.current_dir(&dir);
    pip_cmd.env("UV_CACHE_DIR", &uv_cache_dir);
    pip_cmd.env_remove("PYTHONHOME");
    match pip_cmd.output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            update_status(
                app,
                "error",
                &format!(
                    "Failed to install Python packages with uv:\n{}",
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
                &format!("Failed to run uv pip install: {}", e),
                0,
            );
            return false;
        }
    }

    // Install PyTorch (optional, for NN engine support).
    update_status(
        app,
        "torch",
        "Detecting GPU and installing PyTorch\u{2026}",
        80,
    );
    let mut torch_cmd = cmd(&python_bin.to_string_lossy());
    torch_cmd.arg(backend_dir.join("install.py"));
    torch_cmd.current_dir(&backend_dir);
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

#[cfg(target_os = "windows")]
fn start_production_server(app_handle: &tauri::AppHandle) {
    let dir = data_dir();
    let uv_path = dir.join("uv.exe");
    let venv_dir = dir.join(".venv");
    let script = dir.join("backend").join("gladiator_api.py");

    update_status(
        app_handle,
        "starting",
        "Starting backend server\u{2026}",
        95,
    );

    let venv_py = venv_python(&venv_dir);

    // Ensure the virtual environment is functional.
    if !venv_py.exists() || !python_stdlib_ok(&venv_py) {
        if uv_path.exists() {
            eprintln!("[gladiator-gui] venv broken, recreating with uv");
            if venv_dir.exists() {
                let old = dir.join(".venv.old");
                let _ = std::fs::remove_dir_all(&old);
                if std::fs::rename(&venv_dir, &old).is_err() {
                    let _ = std::fs::remove_dir_all(&venv_dir);
                }
            }
            let mut venv_cmd = cmd(&uv_path.to_string_lossy());
            venv_cmd.args([
                "venv",
                "--seed",
                "--python",
                "3.11",
                &venv_dir.to_string_lossy(),
            ]);
            venv_cmd.current_dir(&dir);
            venv_cmd.env_remove("PYTHONHOME");
            match venv_cmd.output() {
                Ok(out) if out.status.success() => {}
                _ => {
                    update_status(
                        app_handle,
                        "error",
                        "Failed to recreate virtual environment with uv.",
                        0,
                    );
                    return;
                }
            }
        } else {
            update_status(
                app_handle,
                "error",
                "No working Python virtual environment found and uv is not available.\n\
                 Reinstall the application.",
                0,
            );
            return;
        }
    }

    // Fix missing packages with uv.
    if !can_import(&venv_py, "fastapi") {
        if uv_path.exists() {
            eprintln!("[gladiator-gui] fastapi not importable, installing with uv");
            let mut pip_cmd = cmd(&uv_path.to_string_lossy());
            pip_cmd.args(["pip", "install", "fastapi", "uvicorn[standard]"]);
            pip_cmd.current_dir(&dir);
            pip_cmd.env_remove("PYTHONHOME");
            match pip_cmd.output() {
                Ok(out) if out.status.success() => {}
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    update_status(
                        app_handle,
                        "error",
                        &format!(
                            "Failed to install fastapi/uvicorn with uv:\n{}",
                            stderr.trim()
                        ),
                        0,
                    );
                    return;
                }
                Err(e) => {
                    update_status(app_handle, "error", &format!("Failed to run uv: {}", e), 0);
                    return;
                }
            }
        } else {
            update_status(
                app_handle,
                "error",
                "fastapi is not installed and uv is not available.\n\
                 Reinstall the application.",
                0,
            );
            return;
        }
        if !can_import(&venv_py, "fastapi") {
            update_status(
                app_handle,
                "error",
                "fastapi still not importable after uv install. Try reinstalling.",
                0,
            );
            return;
        }
    }

    eprintln!(
        "[gladiator-gui] Starting server with venv Python: {}",
        venv_py.display()
    );
    match try_start_server(&venv_py, &script) {
        Ok(child) => {
            *app_handle.state::<ApiProcess>().0.lock().unwrap() = Some(child);
            update_status(app_handle, "ready", "Ready!", 100);
        }
        Err(e) => {
            let _ = std::fs::remove_file(dir.join(".installed"));
            let msg = format!("Python server {}: {}", e, venv_py.display());
            update_status(app_handle, "error", &msg, 0);
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
fn python_stdlib_ok(python: &PathBuf) -> bool {
    Command::new(python)
        .args(["-c", "import encodings"])
        .env_remove("PYTHONHOME")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
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
#[cfg(not(target_os = "windows"))]
fn try_start_server(bin: &PathBuf, script: &PathBuf) -> Result<Child, String> {
    kill_port(8742);
    let mut cmd = Command::new(bin);
    cmd.arg(script).arg("8742");
    cmd.stderr(Stdio::piped());
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

    // Download uv binary.
    update_status(app, "downloading", "Downloading uv\u{2026}", 30);
    let uv_path = dir.join("uv");
    let uv_url = if cfg!(target_os = "macos") {
        "https://github.com/astral-sh/uv/releases/download/0.5.0/uv-aarch64-apple-darwin.tar.gz"
    } else {
        "https://github.com/astral-sh/uv/releases/download/0.5.0/uv-x86_64-unknown-linux-gnu.tar.gz"
    };
    let uv_archive = dir.join("uv.tar.gz");
    if !download_file(uv_url, &uv_archive, app) {
        try_restore_user_data(&backup, &backend);
        return false;
    }
    let mut tar_cmd = Command::new("tar");
    tar_cmd.args([
        "-xzf",
        &uv_archive.to_string_lossy(),
        "--strip-components=1",
        "-C",
        &dir.to_string_lossy(),
    ]);
    match tar_cmd.output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            update_status(
                app,
                "error",
                &format!("Failed to extract uv:\n{}", stderr.trim()),
                0,
            );
            try_restore_user_data(&backup, &backend);
            return false;
        }
        Err(e) => {
            update_status(app, "error", &format!("Failed to run tar: {}", e), 0);
            try_restore_user_data(&backup, &backend);
            return false;
        }
    }
    let _ = std::fs::remove_file(&uv_archive);
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(&uv_path, std::fs::Permissions::from_mode(0o755));

    // Create virtual environment with uv (downloads Python if needed).
    update_status(app, "venv", "Creating virtual environment\u{2026}", 45);
    let venv_dir = dir.join(".venv");
    if venv_dir.exists() {
        let old = dir.join(".venv.old");
        let _ = std::fs::remove_dir_all(&old);
        if std::fs::rename(&venv_dir, &old).is_err() {
            let _ = std::fs::remove_dir_all(&venv_dir);
        }
    }
    let mut venv_cmd = Command::new(&uv_path);
    venv_cmd.args([
        "venv",
        "--seed",
        "--python",
        "3.11",
        &venv_dir.to_string_lossy(),
    ]);
    venv_cmd.current_dir(&dir);
    venv_cmd.env_remove("PYTHONHOME");
    match venv_cmd.output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            update_status(
                app,
                "error",
                &format!(
                    "Failed to create virtual environment with uv:\n{}",
                    stderr.trim()
                ),
                0,
            );
            try_restore_user_data(&backup, &backend);
            return false;
        }
        Err(e) => {
            update_status(app, "error", &format!("Failed to run uv: {}", e), 0);
            try_restore_user_data(&backup, &backend);
            return false;
        }
    }
    let python_bin = venv_python(&venv_dir);
    if !python_bin.exists() || !python_stdlib_ok(&python_bin) {
        update_status(
            app,
            "error",
            "uv created the virtual environment but the Python interpreter is not functional.",
            0,
        );
        try_restore_user_data(&backup, &backend);
        return false;
    }

    // Install pip dependencies with uv.
    update_status(app, "deps", "Installing Python packages\u{2026}", 60);
    let req_path = backend.join("requirements.txt");
    let mut pip_cmd = Command::new(&uv_path);
    pip_cmd.args(["pip", "install", "-r", &req_path.to_string_lossy()]);
    pip_cmd.current_dir(&dir);
    pip_cmd.env_remove("PYTHONHOME");
    match pip_cmd.output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            update_status(
                app,
                "error",
                &format!(
                    "Failed to install Python packages with uv:\n{}",
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
                &format!("Failed to run uv pip install: {}", e),
                0,
            );
            return false;
        }
    }

    // Install PyTorch (optional, for NN engine support).
    update_status(
        app,
        "torch",
        "Detecting GPU and installing PyTorch\u{2026}",
        80,
    );
    let mut torch_cmd = Command::new(&python_bin);
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
    let uv_path = dir.join("uv");
    let venv_dir = dir.join(".venv");
    let script = dir.join("backend").join("gladiator_api.py");

    update_status(
        app_handle,
        "starting",
        "Starting backend server\u{2026}",
        95,
    );

    let venv_py = venv_python(&venv_dir);

    // Ensure the virtual environment is functional.
    if !venv_py.exists() || !python_stdlib_ok(&venv_py) {
        if uv_path.exists() {
            eprintln!("[gladiator-gui] venv broken, recreating with uv");
            if venv_dir.exists() {
                let old = dir.join(".venv.old");
                let _ = std::fs::remove_dir_all(&old);
                if std::fs::rename(&venv_dir, &old).is_err() {
                    let _ = std::fs::remove_dir_all(&venv_dir);
                }
            }
            let mut venv_cmd = Command::new(&uv_path);
            venv_cmd.args([
                "venv",
                "--seed",
                "--python",
                "3.11",
                &venv_dir.to_string_lossy(),
            ]);
            venv_cmd.current_dir(&dir);
            venv_cmd.env_remove("PYTHONHOME");
            match venv_cmd.output() {
                Ok(out) if out.status.success() => {}
                _ => {
                    update_status(
                        app_handle,
                        "error",
                        "Failed to recreate virtual environment with uv.",
                        0,
                    );
                    return;
                }
            }
        } else {
            update_status(
                app_handle,
                "error",
                "No working Python virtual environment found and uv is not available.\n\
                 Reinstall the application.",
                0,
            );
            return;
        }
    }

    // Fix missing packages with uv.
    if !can_import(&venv_py, "fastapi") {
        if uv_path.exists() {
            eprintln!("[gladiator-gui] fastapi not importable, installing with uv");
            let mut pip_cmd = Command::new(&uv_path);
            pip_cmd.args(["pip", "install", "fastapi", "uvicorn[standard]"]);
            pip_cmd.current_dir(&dir);
            pip_cmd.env_remove("PYTHONHOME");
            match pip_cmd.output() {
                Ok(out) if out.status.success() => {}
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    update_status(
                        app_handle,
                        "error",
                        &format!(
                            "Failed to install fastapi/uvicorn with uv:\n{}",
                            stderr.trim()
                        ),
                        0,
                    );
                    return;
                }
                Err(e) => {
                    update_status(app_handle, "error", &format!("Failed to run uv: {}", e), 0);
                    return;
                }
            }
        } else {
            update_status(
                app_handle,
                "error",
                "fastapi is not installed and uv is not available.\n\
                 Reinstall the application.",
                0,
            );
            return;
        }
        if !can_import(&venv_py, "fastapi") {
            update_status(
                app_handle,
                "error",
                "fastapi still not importable after uv install. Try reinstalling.",
                0,
            );
            return;
        }
    }

    eprintln!(
        "[gladiator-gui] Starting server with venv Python: {}",
        venv_py.display()
    );
    match try_start_server(&venv_py, &script) {
        Ok(child) => {
            *app_handle.state::<ApiProcess>().0.lock().unwrap() = Some(child);
            update_status(app_handle, "ready", "Ready!", 100);
        }
        Err(e) => {
            let _ = std::fs::remove_file(dir.join(".installed"));
            let msg = format!("Python server {}: {}", e, venv_py.display());
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
                let updater_handle = handle.clone();

                std::thread::spawn(move || {
                    let dir = data_dir();
                    if !is_backend_installed(&dir) {
                        if !install_backend(&dir, &handle) {
                            return;
                        }
                    }
                    start_production_server(&handle);
                });

                // Check for app updates on startup (release builds only)
                tauri::async_runtime::spawn(async move {
                    match updater_handle.updater() {
                        Ok(updater) => {
                            match updater.check().await {
                                Ok(Some(update)) => {
                                    // Save release notes before installing
                                    let notes_dir = data_dir();
                                    let _ = std::fs::create_dir_all(&notes_dir);
                                    let notes = serde_json::json!({
                                        "version": update.version,
                                        "body": update.body,
                                    });
                                    let notes_path = notes_dir.join("last_update.json");
                                    let _ = std::fs::write(
                                        &notes_path,
                                        serde_json::to_string_pretty(&notes).unwrap(),
                                    );

                                    if cfg!(target_os = "windows") {
                                    let url = update.download_url.clone();
                                    let version = update.version.clone();

                                    let temp_dir = std::env::temp_dir()
                                        .join(format!("gladiator_update_{}", version));
                                    let _ = std::fs::create_dir_all(&temp_dir);
                                    let zip_path = temp_dir.join("gladiator_update.zip");
                                    let extract_dir = temp_dir.join("app");

                                    let download_ok = (|| -> Result<(), String> {
                                        let agent = ureq::AgentBuilder::new()
                                            .timeout_connect(
                                                std::time::Duration::from_secs(30),
                                            )
                                            .timeout(std::time::Duration::from_secs(600))
                                            .build();
                                        let response = agent
                                            .get(url.as_str())
                                            .call()
                                            .map_err(|e| format!("Download failed: {e}"))?;
                                        let mut out = std::fs::File::create(&zip_path)
                                            .map_err(|e| format!("Create file: {e}"))?;
                                        let mut reader = response.into_reader();
                                        std::io::copy(&mut reader, &mut out)
                                            .map_err(|e| format!("Write failed: {e}"))?;
                                        drop(out);
                                        Ok(())
                                    })();

                                    if let Err(e) = download_ok {
                                        eprintln!("Update download failed: {}", e);
                                    } else if extract_zip(
                                        &zip_path,
                                        &extract_dir,
                                        &updater_handle,
                                    ) {
                                        if let Ok(exe_path) = std::env::current_exe() {
                                            if let Some(app_dir) = exe_path.parent() {
                                                let pid = std::process::id();
                                                let script = format!(
                                                    "param($p,$s,$d,$e)\n\
                                                     $w=0\n\
                                                     while($w-lt60){{\n\
                                                       $x=Get-Process -Id $p \
                                                     -ErrorAction SilentlyContinue\n\
                                                       if(!$x){{break}}\n\
                                                       Start-Sleep 1\n\
                                                       $w++\n\
                                                     }}\n\
                                                     Start-Sleep 2\n\
                                                     Copy-Item \"$s\\*\" \"$d\\\" \
                                                     -Recurse -Force\n\
                                                     Start-Process $e\n\
                                                     Remove-Item $s -Recurse -Force \
                                                     -ErrorAction SilentlyContinue\n"
                                                );
                                                let script_path = temp_dir
                                                    .join("apply_update.ps1");
                                                if std::fs::write(
                                                    &script_path,
                                                    script.as_bytes(),
                                                )
                                                .is_ok()
                                                {
                                                    let _ = cmd("powershell")
                                                        .args([
                                                            "-NoProfile",
                                                            "-ExecutionPolicy",
                                                            "Bypass",
                                                            "-File",
                                                            &script_path
                                                                .to_string_lossy(),
                                                            &pid.to_string(),
                                                            &extract_dir
                                                                .to_string_lossy(),
                                                            &app_dir
                                                                .to_string_lossy(),
                                                            &exe_path
                                                                .to_string_lossy(),
                                                        ])
                                                        .spawn();
                                                    // Exit so the swap script can
                                                    // replace the running binary
                                                    std::process::exit(0);
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    let _ = update
                                        .download_and_install(|_, _| {}, || {})
                                        .await;
                                }
                            }
                            Ok(None) => {}
                            Err(e) => eprintln!("Update check failed: {}", e),
                        }
                    }
                    Err(e) => eprintln!("Updater init failed: {}", e),
                }
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
        .invoke_handler(tauri::generate_handler![get_setup_status, get_pending_update_notes])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
