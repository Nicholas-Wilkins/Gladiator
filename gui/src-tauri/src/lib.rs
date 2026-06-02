use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

struct ApiProcess(Mutex<Option<Child>>);

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let root = find_project_root();
            let python = find_python();
            let script = root.join("gladiator_api.py");

            eprintln!(
                "[gladiator-gui] Starting API server: {} {}",
                python,
                script.display()
            );

            // Kill leftover processes on port 8742 from previous crashes
            Command::new("sh")
                .arg("-c")
                .arg("fuser -k 8742/tcp 2>/dev/null; exit 0")
                .status()
                .ok();

            let child = Command::new(&python)
                .arg(&script)
                .arg("8742")
                .spawn()
                .expect("Failed to start Python API server");

            app.manage(ApiProcess(Mutex::new(Some(child))));
            std::thread::sleep(std::time::Duration::from_secs(2));
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
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
