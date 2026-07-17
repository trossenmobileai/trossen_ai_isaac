# Workstation Configuration

Covers **one-time** install of Quest / ALVR / Steam / SteamVR / OpenXR, and the **every-session** dual-operator startup used before teleop or recording.

- Stack rationale: [Background and stack](02-background-and-stack.md)
- Commands after VR is live: [IL cheat sheet](../IL_WORKFLOW_CHEATSHEET.md#1-collect-demos--vr-production)
- Design: [VR teleoperation](04-vr-teleoperation.md) · [VR recording](05-vr-recording.md)

---

## Roles

| Role | Who | Responsibility |
|------|-----|----------------|
| **Headset operator** | Person wearing the Quest 3 | Wi-Fi, ALVR app, guardian, hand visibility, watching the sim |
| **Workstation operator** | Person at the PC | ALVR Launcher, SteamVR from ALVR, Toggle Dashboard, Isaac scripts, **Start AR**, keyboard keys (**N** / **U** / …) |

Both roles are required for a smooth session.

---

## Part A — One-time setup

Do this once per machine (or after re-imaging). Skip if the host is already validated.

### A.1 Network / Wi-Fi

Quest 3 and the workstation must be on the **same** local network so ALVR can discover the headset.

- Prefer **5 GHz** Wi-Fi; a dedicated access point near the play area is more reliable than busy institutional Wi-Fi.
- Institutional networks often block peer-to-peer traffic — see [Findings](06-findings-troubleshooting.md).

> **Screenshot placeholder:** `docs/assets/epic4/quest-wifi-same-network.png` — Quest Wi-Fi settings showing the same SSID as the PC.
>
> ![Quest Wi-Fi same network (placeholder)](../assets/epic4/quest-wifi-same-network.png)

### A.2 Meta Quest 3

1. Power on the headset and complete Meta first-run / accounts as needed.
2. Enable **Hand Tracking** and **Auto Switch between Hands and Controllers** (Quest Settings → Movement / Hands — exact labels vary by OS version).
3. Install the **ALVR** client on the headset (Meta Store listing or the sideload path your team uses for the matching ALVR server version).

> **Screenshot placeholder:** `docs/assets/epic4/quest-hand-tracking-settings.png` — Quest hand-tracking toggles.
>
> ![Quest hand tracking settings (placeholder)](../assets/epic4/quest-hand-tracking-settings.png)

> **Screenshot placeholder:** `docs/assets/epic4/quest-alvr-app-installed.png` — ALVR app icon on the Quest home.
>
> ![Quest ALVR app installed (placeholder)](../assets/epic4/quest-alvr-app-installed.png)

### A.3 Workstation: Steam and SteamVR

1. Install **Steam** for Linux from [store.steampowered.com](https://store.steampowered.com/) (or your distro’s supported method).
2. In Steam → Library, install **SteamVR**.
3. Apply the **Linux capability** fix so SteamVR can set scheduling priority:

```bash
sudo setcap CAP_SYS_NICE+eip \
  ~/.steam/debian-installation/steamapps/common/SteamVR/bin/linux64/vrcompositor-launcher
```

Verify with `getcap` on the same path. If the path differs:

```bash
find ~ -name "vrcompositor-launcher" 2>/dev/null
```

4. Set SteamVR **Launch Options** (Steam → Library → right-click **SteamVR** → Properties → Launch Options):

```
~/.steam/debian-installation/steamapps/common/SteamVR/bin/vrmonitor.sh %command%
```

> **Screenshot placeholder:** `docs/assets/epic4/steamvr-launch-options.png` — SteamVR Properties → Launch Options field filled in.
>
> ![SteamVR launch options (placeholder)](../assets/epic4/steamvr-launch-options.png)

### A.4 Workstation: ALVR server

1. Install **ALVR** (Launcher / server) for Linux from the [ALVR project releases](https://github.com/alvr-org/ALVR/releases) matching the Quest client version.
2. Register the ALVR SteamVR driver by creating `~/.local/share/Steam/config/steamvr.vrsettings` if it does not exist:

```json
{
   "Driver_alvr_server" : {
      "enable" : true,
      "loadPriority" : 0
   },
   "steamvr" : {
      "activateMultipleDrivers" : true
   }
}
```

3. In the ALVR dashboard: set **Hand Tracking** interaction to **SteamVR Input 2.0**.

> **Screenshot placeholder:** `docs/assets/epic4/alvr-dashboard-hand-tracking.png` — ALVR dashboard Hand Tracking → SteamVR Input 2.0.
>
> ![ALVR dashboard hand tracking (placeholder)](../assets/epic4/alvr-dashboard-hand-tracking.png)

### A.5 SteamVR as OpenXR runtime

Isaac Sim talks to **OpenXR**, not to ALVR’s API. Point the system OpenXR runtime at SteamVR:

1. Start SteamVR (for this one-time step you may use ALVR’s Launch SteamVR once the chain works).
2. In SteamVR: **☰ Menu → Settings → Developer → Set SteamVR as OpenXR Runtime**.
3. Verify:

```bash
cat ~/.config/openxr/1/active_runtime.json
# Must show "name": "SteamVR"
```

> **Screenshot placeholder:** `docs/assets/epic4/steamvr-set-openxr-runtime.png` — SteamVR Developer setting “Set SteamVR as OpenXR Runtime”.
>
> ![SteamVR set OpenXR runtime (placeholder)](../assets/epic4/steamvr-set-openxr-runtime.png)

### A.6 One-time smoke (before Isaac)

1. Launch **ALVR** on the PC.
2. Open **ALVR** on the Quest; **Trust** the device on the PC Devices list if prompted.
3. From ALVR on the PC, click **Launch SteamVR** (do not start SteamVR only from Steam).
4. Confirm the headset shows the SteamVR home / dashboard environment.

If that fails, fix network / trust / setcap / OpenXR before installing Isaac Lab VR scripts. Troubleshooting: [Findings](06-findings-troubleshooting.md).

---

## Part B — Per-session startup

Run this every time before VR teleoperation or VR recording. Order matters.

### B.1 Same Wi-Fi

Confirm Quest and workstation are on the **same** network (see [A.1](#a1-network--wi-fi)).

### B.2 Open ALVR on the headset; trust on the PC

1. Workstation operator: start **ALVR Launcher** → **Launch** so the ALVR server UI is up.
2. Headset operator: open the **ALVR** app on the Quest.
3. If the headset prompts you to open ALVR on the PC, the workstation operator should already have it running.
4. On the PC ALVR **Devices** list, click **Trust** next to the Quest entry (required on first pairing; later sessions may auto-connect).

> **Screenshot placeholder:** `docs/assets/epic4/alvr-trust-device.png` — ALVR PC Devices tab with Trust next to the Quest.
>
> ![ALVR Trust device (placeholder)](../assets/epic4/alvr-trust-device.png)

### B.3 Room boundary (guardian)

If the Quest detects obstacles or has no play area, it prompts for a **room boundary / guardian**. Follow the on-screen instructions until tracking space is accepted.

> **Screenshot placeholder:** `docs/assets/epic4/quest-room-boundary.png` — Quest guardian / boundary setup screens.
>
> ![Quest room boundary (placeholder)](../assets/epic4/quest-room-boundary.png)

### B.4 Launch SteamVR from ALVR

1. On the PC ALVR UI, click **Launch SteamVR** (do **not** launch SteamVR only from the Steam library).
2. **First run:** after the device is listed, press **Trust** if still required; then SteamVR should bring the headset into the SteamVR environment.
3. **Later runs:** if the headset was trusted before, SteamVR often starts and drops you on the SteamVR dashboard automatically.

Only after SteamVR is up via ALVR is the headset reliably recognized by the PC VR stack.

> **Screenshot placeholder:** `docs/assets/epic4/alvr-launch-steamvr.png` — ALVR UI “Launch SteamVR” control.
>
> ![ALVR Launch SteamVR (placeholder)](../assets/epic4/alvr-launch-steamvr.png)

### B.5 Confirm both hands are tracked

Headset operator: move both arms so the Quest cameras see them. In the SteamVR world you should see **two cursors / hand representations**.

**Tip:** Hold both hands in view of the headset **before** starting SteamVR.

If hands are missing:

1. Ask the workstation operator to **restart SteamVR through ALVR**.
2. If still broken: close SteamVR (Steam apps), close ALVR, then restart from [B.2](#b2-open-alvr-on-the-headset-trust-on-the-pc) (ALVR → Launch SteamVR from ALVR → check hands again).

> **Screenshot placeholder:** `docs/assets/epic4/steamvr-both-hands-tracked.png` — SteamVR view with two hand cursors visible.
>
> ![SteamVR both hands tracked (placeholder)](../assets/epic4/steamvr-both-hands-tracked.png)

### B.6 Toggle SteamVR dashboard off

With hands tracked, the workstation operator clears the SteamVR dashboard overlay so it does not block the view:

1. Focus the **SteamVR** window on the PC.
2. Open the menu via the **stacked lines** (☰) control at the top of the window.
3. Click **Toggle Dashboard** so the in-world dashboard is **off**.

Hands should remain tracked after the dashboard is hidden.

> **Screenshot placeholder:** `docs/assets/epic4/steamvr-toggle-dashboard.png` — SteamVR window menu → Toggle Dashboard.
>
> ![SteamVR Toggle Dashboard (placeholder)](../assets/epic4/steamvr-toggle-dashboard.png)

### B.7 Launch teleop or recording on the PC

Workstation operator starts the Isaac Lab entrypoint (examples: paths are workstation-specific):

- Practice teleop: `teleop_dual_arm_vr.py` — see [VR teleoperation](04-vr-teleoperation.md)
- Production collect: `run_collect_dataset.sh` / `record_dual_arm_vr.py` — see [VR recording](05-vr-recording.md) and [cheat sheet](../IL_WORKFLOW_CHEATSHEET.md#1-collect-demos--vr-production)

### B.8 Isaac Sim: OpenXR + Start AR

In the **Isaac Sim** window (once the app is up from the script):

1. Set **Output Plugin** = **OpenXR** (viewport / XR output controls — see screenshots below).
2. Click **Start AR**.

The headset should leave the SteamVR home and show the simulation stereo view.

> **Screenshot placeholder:** `docs/assets/epic4/isaac-sim-main-window.png` — Full Isaac Sim main window during VR teleop/recording (orientation for where XR controls live).
>
> ![Isaac Sim main window (placeholder)](../assets/epic4/isaac-sim-main-window.png)

> **Screenshot placeholder:** `docs/assets/epic4/isaac-output-plugin-openxr-start-ar.png` — Close-up: Output Plugin = OpenXR and **Start AR** control in the Isaac Sim UI.
>
> ![Isaac Start AR (placeholder)](../assets/epic4/isaac-output-plugin-openxr-start-ar.png)

### B.9 POV reset if the first spawn looks wrong

On the first entry into the sim, the viewpoint is often wrong relative to the robot. **Remove the headset for a couple of seconds, then put it back on** — that usually resets the POV / XR anchor alignment.

### B.10 Engage teleop / recording with the workstation operator

After warm-up (hands live), the workstation operator engages:

| Mode | Typical keys |
|------|----------------|
| Teleop only | **N** engage · **M** pause · **B** re-anchor · **J** reset · **TAB** switch arm (single-arm) ([details](04-vr-teleoperation.md)) |
| Recording | **U** engage · **I** pause · **N** episode · **M** discard · **B** re-anchor · **J** reset ([details](05-vr-recording.md)) |

You are now ready to practice or collect demos.

---

**Hub:** [Epic 4](../EPIC4_VR_INTEGRATION.md) · **Troubleshoot:** [Findings](06-findings-troubleshooting.md)
