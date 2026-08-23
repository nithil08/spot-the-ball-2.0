# Setting up the SFO PC for remote access

Hand this to whoever has physical access to the machine. It only covers **getting remote
access working** — no project setup, no builds. That all happens remotely afterwards.

Budget ~20 minutes, most of it waiting on downloads. You need an **administrator** account.

Anything in `PowerShell (Admin)` means: right-click Start → "Terminal (Admin)" or
"Windows PowerShell (Admin)".

---

## 1. Install Tailscale

This is what makes the machine reachable from Phoenix. It's a private network — the PC is
**not** exposed to the public internet, and no router or firewall changes are needed.

1. Download from <https://tailscale.com/download/windows> and run the installer.
2. When it asks you to log in, **don't** — instead open `PowerShell (Admin)` and run the
   auth key you were sent (it looks like `tskey-auth-...`):

   ```powershell
   tailscale up --authkey tskey-auth-PASTE-THE-KEY-HERE
   ```

   The key is single-use and expires. It exists so you never need anyone's password.

3. **Turn on unattended mode.** In the Tailscale tray icon menu there is a setting for
   running/staying connected when nobody is signed in (wording varies by version — it's
   the one mentioning *unattended* or *when I'm not logged in*). This matters: without it,
   the connection drops when the machine is locked or restarted, and the whole thing
   becomes unreachable.

4. Confirm it worked:

   ```powershell
   tailscale status
   ```

   Note down the **machine name** shown for this PC and send it back.

## 2. Enable the SSH server

1. Settings → System → Optional features → "Add an optional feature" → search
   **OpenSSH Server** → install.
2. Then in `PowerShell (Admin)`, start it and make it survive reboots:

   ```powershell
   Start-Service sshd
   Set-Service -Name sshd -StartupType Automatic
   ```

3. Confirm:

   ```powershell
   Get-Service sshd
   ```

   Should say `Running`.

## 3. Confirm the account details

Remote login needs a Windows account **with a password set** — a blank password will be
refused, and a PIN-only Microsoft account login does not count.

```powershell
whoami
```

Send back what that prints. If the account has no password, set one, and send it over a
different channel than the machine name.

## 4. Stop the machine from ever sleeping

**The most important step.** If the PC sleeps, it disappears from the network and someone
has to physically walk over and wake it. Long render jobs will be running unattended for
hours.

In `PowerShell (Admin)`:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change disk-timeout-ac 0
powercfg /h off
```

The screen is allowed to sleep — that's fine and saves power:

```powershell
powercfg /change monitor-timeout-ac 10
```

If it's a **laptop**, also set lid-close to do nothing: Control Panel → Power Options →
"Choose what closing the lid does" → *When plugged in: Do nothing*. And leave it plugged
in.

Finally, stop Windows Update from silently rebooting mid-job: Settings → Windows Update →
Advanced options → set **Active hours** as wide as possible.

## 5. Recommended: install WSL2 before you leave

Strictly optional, but it removes the one step that could lock everyone out. WSL2 install
requires admin rights *and* a reboot — awkward to do remotely, and if it goes wrong the
machine may not come back cleanly.

```powershell
wsl --install
```

Then **reboot**, sign back in, and let it finish (it may prompt for a Linux username and
password — anything is fine, just write it down and send it back).

Verify after the reboot:

```powershell
wsl --status
tailscale status
Get-Service sshd
```

All three should look healthy. That last check is worth doing regardless — it confirms
remote access survives a restart, which is the thing that actually matters.

---

## Send back

- Tailscale **machine name** (from step 1)
- Windows **username** (from step 3)
- The **password**, over a different channel
- Whether you did step 5, and the WSL username/password if so

That's everything. Remote access takes over from here — no further physical access should
be needed.

## If it goes wrong

- **`tailscale` not recognised** — reopen PowerShell; the installer changes PATH and the
  old window won't have it.
- **Auth key rejected** — they expire and are single-use. Ask for a fresh one.
- **`Start-Service sshd` fails** — the optional feature in step 2 didn't finish
  installing. Recheck Optional features and look for OpenSSH Server in the installed list.
