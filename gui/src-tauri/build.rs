fn main() {
    if std::path::Path::new("backend/gladiator-backend-windows.zip").exists() {
        println!("cargo:rustc-cfg=has_backend_zip");
    }
    if std::path::Path::new("backend/uv.exe").exists() {
        println!("cargo:rustc-cfg=has_uv_exe");
    }
    tauri_build::build()
}
