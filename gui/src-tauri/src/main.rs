#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // WebKitGTK requires X11 backend on Wayland (Arch Linux).
    // Disable compositing to avoid GBM buffer errors on some GPU drivers.
    std::env::set_var("GDK_BACKEND", "x11");
    std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
    gladiator_gui_lib::run();
}
