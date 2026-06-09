## Downloads

### Windows

**Recommended for most users — try this first:**
- [**Gladiator_<version>_x64-setup.exe**](https://github.com/Nicholas-Wilkins/Gladiator/releases/download/v<version>/Gladiator_<version>_x64-setup.exe)
  NSIS installer. Simple, familiar setup wizard with a modern UI.
  1. Download the `.exe` from the link above.
  2. Run the installer.
  3. If Windows Defender shows a warning, click **More info**, then click **Run anyway**.
  4. Follow the prompts — Gladiator will be installed and available from your Start menu.

**Enterprise / IT-admin deployment:**
- [**Gladiator_<version>_x64_en-US.msi**](https://github.com/Nicholas-Wilkins/Gladiator/releases/download/v<version>/Gladiator_<version>_x64_en-US.msi)
  Microsoft Installer. Integrates with Add/Remove Programs, supports silent install (`msiexec /i ...`), and can be deployed via Group Policy or SCCM.

### Linux

- [**Gladiator_<version>_amd64.AppImage**](https://github.com/Nicholas-Wilkins/Gladiator/releases/download/v<version>/Gladiator_<version>_amd64.AppImage)
  Portable AppImage — download, `chmod +x Gladiator_<version>_amd64.AppImage`, then run `./Gladiator_<version>_amd64.AppImage`.
- [**Gladiator_<version>_amd64.deb**](https://github.com/Nicholas-Wilkins/Gladiator/releases/download/v<version>/Gladiator_<version>_amd64.deb)
  Debian/Ubuntu package — `sudo dpkg -i Gladiator_<version>_amd64.deb`.

### macOS

- [**Gladiator_<version>_aarch64.dmg**](https://github.com/Nicholas-Wilkins/Gladiator/releases/download/v<version>/Gladiator_<version>_aarch64.dmg) (Apple Silicon)

  1. Download the DMG from the link above.
  2. Open the DMG — drag the Gladiator app to your **Applications** folder.
  3. The first time you open it, macOS may show a warning that the app is from an unidentified developer. To bypass:
     - **Right-click** (or Ctrl-click) the app in Applications and select **Open**.
     - Click **Open** in the dialog that appears.
  4. You only need to do this once — subsequent launches work normally.

---

### Auto-updater artifacts (not for manual install)

- `Gladiator_<version>_x64_portable.zip` — portable app archive for Windows auto-updater
- `Gladiator_<version>_amd64.AppImage.tar.gz` — compressed AppImage for Linux auto-updater
- `latest.json` — update manifest consumed by the built-in auto-updater on all platforms
