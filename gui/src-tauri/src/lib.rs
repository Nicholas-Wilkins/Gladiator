use serde::Serialize;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
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

fn backend_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "gladiator-backend-x86_64-windows.exe"
    } else if cfg!(target_os = "macos") {
        "gladiator-backend-aarch64-macos"
    } else {
        "gladiator-backend-x86_64-linux"
    }
}

fn backend_download_url(version: &str) -> String {
    format!(
        "https://github.com/Nicholas-Wilkins/Gladiator/releases/download/v{}/{}",
        version,
        backend_binary_name()
    )
}

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

    // Make executable on Unix
    #[cfg(not(target_os = "windows"))]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&binary_path, std::fs::Permissions::from_mode(0o755)).ok();
    }

    if std::fs::write(dir.join(".installed"), env!("CARGO_PKG_VERSION")).is_err() {
        update_status(app, "error", "Failed to create installation marker.", 0);
        return false;
    }
    eprintln!("[gladiator-gui] Backend installation complete!");
    true
}

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

    let mut child = match Command::new(&binary_path)
        .arg("8742")
        .env("GLADIATOR_DATA_DIR", dir.to_string_lossy().to_string())
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
        }
        Ok(None) => {
            *app_handle.state::<ApiProcess>().0.lock().unwrap() = Some(child);
            update_status(app_handle, "ready", "Ready!", 100);
        }
        Err(e) => {
            update_status(
                app_handle,
                "error",
                &format!("Process check failed: {}", e),
                0,
            );
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
