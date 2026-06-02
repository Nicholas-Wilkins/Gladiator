use serde::Serialize;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

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
        if Command::new(name).arg("--version").output().is_ok() {
            return name.to_string();
        }
    }
    "python3".to_string()
}

fn kill_port(port: u16) {
    if cfg!(target_os = "windows") {
        Command::new("powershell")
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

    let child = Command::new(&python)
        .arg(&script)
        .arg("8742")
        .spawn()
        .expect("Failed to start Python API server");

    *app_handle.state::<ApiProcess>().0.lock().unwrap() = Some(child);
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

fn venv_python(venv_dir: &PathBuf) -> PathBuf {
    if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("python.exe")
    } else {
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
}

fn venv_pip(venv_dir: &PathBuf) -> PathBuf {
    if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("pip.exe")
    } else {
        venv_dir.join("bin").join("pip")
    }
}

fn python_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "python"
    } else {
        if Command::new("python3").arg("--version").output().is_ok() {
            "python3"
        } else {
            "python"
        }
    }
}

fn pip_install(venv_dir: &PathBuf, pkgs: &[&str]) -> Result<String, String> {
    let out = Command::new(&venv_pip(venv_dir))
        .args(["install"])
        .args(pkgs)
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

fn can_import(python: &PathBuf, module: &str) -> bool {
    Command::new(python)
        .args(["-c", &format!("import {}", module)])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn is_backend_installed(dir: &PathBuf) -> bool {
    let installed = dir.join(".installed").exists();
    if !installed {
        return false;
    }
    let py = venv_python(&dir.join(".venv"));
    if !py.exists() {
        return false;
    }
    // Verify the venv is actually functional
    can_import(&py, "fastapi")
}

fn download_file(url: &str, dest: &PathBuf, app: &tauri::AppHandle) -> bool {
    let result = if cfg!(target_os = "windows") {
        Command::new("powershell")
            .args([
                "-Command",
                &format!(
                    "Invoke-WebRequest -Uri '{}' -OutFile '{}'",
                    url,
                    dest.display()
                ),
            ])
            .output()
    } else {
        Command::new("curl")
            .args(["-L", "-o", &dest.to_string_lossy(), url])
            .output()
    };

    match result {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            update_status(
                app,
                "error",
                &format!("Download failed:\n{}", stderr.trim()),
                0,
            );
            false
        }
        Err(e) => {
            update_status(app, "error", &format!("Failed to run curl: {}", e), 0);
            false
        }
    }
}

fn extract_zip(zip_path: &PathBuf, dest: &PathBuf, app: &tauri::AppHandle) -> bool {
    let result = if cfg!(target_os = "windows") {
        Command::new("powershell")
            .args([
                "-Command",
                &format!(
                    "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                    zip_path.display(),
                    dest.display()
                ),
            ])
            .output()
    } else {
        Command::new("unzip")
            .args([
                "-o",
                &zip_path.to_string_lossy(),
                "-d",
                &dest.to_string_lossy(),
            ])
            .output()
    };

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

fn install_backend(dir: &PathBuf, app: &tauri::AppHandle) {
    eprintln!("[gladiator-gui] Installing backend to {}", dir.display());
    update_status(
        app,
        "downloading",
        "Downloading backend from GitHub\u{2026}",
        5,
    );

    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).expect("Failed to create backend directory");

    let zip_path = dir.join("repo.zip");
    let url = "https://github.com/Nicholas-Wilkins/Gladiator/archive/refs/heads/main.zip";
    if !download_file(url, &zip_path, app) {
        return;
    }

    update_status(app, "extracting", "Extracting files\u{2026}", 20);
    if !extract_zip(&zip_path, dir, app) {
        return;
    }

    let extracted = dir.join("Gladiator-main");
    let backend = dir.join("backend");
    std::fs::rename(&extracted, &backend).expect("Failed to rename extracted directory");
    let _ = std::fs::remove_file(&zip_path);

    update_status(app, "venv", "Setting up Python environment\u{2026}", 30);
    let venv_dir = dir.join(".venv");
    match Command::new(python_name())
        .args(["-m", "venv", &venv_dir.to_string_lossy()])
        .output()
    {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let msg = format!(
                "Failed to create Python virtual environment:\n{}",
                stderr.trim()
            );
            update_status(app, "error", &msg, 0);
            return;
        }
        Err(e) => {
            let msg = format!(
                "Failed to run Python ({}): {}\nMake sure python3 or python is installed.",
                python_name(),
                e
            );
            update_status(app, "error", &msg, 0);
            return;
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
        let msg = format!(
            "Virtual environment created but no Python binary found in {}.\nContents: {}\nThis may indicate a broken or incomplete Python installation.",
            bin_dir.display(),
            contents
        );
        update_status(app, "error", &msg, 0);
        return;
    }

    update_status(app, "deps", "Installing Python packages\u{2026}", 40);
    let req_path = backend.join("requirements.txt");
    match Command::new(&venv_pip(&venv_dir))
        .args(["install", "-r", &req_path.to_string_lossy()])
        .output()
    {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let msg = format!("Failed to install Python packages:\n{}", stderr.trim());
            update_status(app, "error", &msg, 0);
            return;
        }
        Err(e) => {
            let msg = format!("Failed to run pip: {}", e);
            update_status(app, "error", &msg, 0);
            return;
        }
    }

    update_status(app, "torch", "Installing PyTorch (CPU-only)\u{2026}", 65);
    match Command::new(&venv_pip(&venv_dir))
        .args([
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ])
        .output()
    {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let msg = format!("Failed to install PyTorch:\n{}", stderr.trim());
            update_status(app, "error", &msg, 0);
            return;
        }
        Err(e) => {
            let msg = format!("Failed to run pip for PyTorch: {}", e);
            update_status(app, "error", &msg, 0);
            return;
        }
    }

    if std::fs::write(dir.join(".installed"), "").is_err() {
        update_status(app, "error", "Failed to create installation marker.", 0);
        return;
    }
    eprintln!("[gladiator-gui] Backend installation complete!");
}

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

fn try_start_server(
    bin: &PathBuf,
    script: &PathBuf,
    site_packages: &Option<PathBuf>,
) -> Result<Child, String> {
    kill_port(8742);
    let mut cmd = Command::new(bin);
    cmd.arg(script).arg("8742");
    if let Some(sp) = site_packages {
        cmd.env("PYTHONPATH", sp);
    }
    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn: {}", e))?;
    std::thread::sleep(std::time::Duration::from_millis(400));
    match child.try_wait() {
        Ok(Some(status)) => {
            let code = status.code().map(|c| c.to_string()).unwrap_or_default();
            Err(format!("exited immediately (code {})", code))
        }
        Ok(None) => Ok(child),
        Err(e) => Err(format!("process check failed: {}", e)),
    }
}

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

    // ── Find a working python binary ────────────────────────────
    let venv_py = venv_python(&venv_dir);
    let system_py = PathBuf::from(python_name());
    let sp = find_venv_site_packages(&venv_dir);

    let (bin, label) =
        if venv_py.exists() && Command::new(&venv_py).arg("--version").output().is_ok() {
            (venv_py, "venv")
        } else {
            eprintln!("[gladiator-gui] venv Python broken, using system python");
            (system_py, "system")
        };

    // ── Ensure critical deps are importable ─────────────────────
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
            // Try inside the venv first, then globally
            let installed = if venv_dir.join("bin").join("pip").exists() {
                pip_install(&venv_dir, &["fastapi", "uvicorn[standard]"]).is_ok()
            } else {
                false
            };
            if !installed {
                // Fall back to system pip
                let result = Command::new(python)
                    .args(["-m", "pip", "install", "fastapi", "uvicorn[standard]"])
                    .output();
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
        }
    }

    // ── Start the server ────────────────────────────────────────
    eprintln!(
        "[gladiator-gui] Starting server with {}: {}",
        label,
        python.display()
    );
    match try_start_server(python, &script, &sp) {
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
                        install_backend(&dir, &handle);
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
