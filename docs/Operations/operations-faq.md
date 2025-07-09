# Operations FAQ for RemoteRL

Use the checklist below to quickly probe how RemoteRL Cloud is set up and what operational limits apply, grouped by topic for easy scanning.


## 1. Session Duration and Persistence

* **1a. How long can a training job run remote continuously once it starts?**
   *(i.e., the longest uninterrupted training session allowed)*


* **1b. If the simulator (trainer session) sits idle with no user input, how long will the session stay alive before it is automatically shut down?**
   *(i.e., the maximum “cold-idle” waiting time)*


---

### Answers

**1a — Maximum Conntection Duration**
* There’s no fixed wall-clock limit. A training job can run for hours or days so long as your account still has time / data credits.


**1b — Idle timeout before shutdown** 
* The server sends its own heartbeats and use reconnection loop to keep the connection alive, so an idle simulator isn’t shut down just because you aren’t typing. Credits keep accruing while the session is active. The session is closed only if the connection goes completely silent—e.g., the client vanishes and no heartbeats are received— mostly immediate or \~mostly far less than 5 minutes—does the system close the session.

## 2. Disconnection and Reconnection

* **2a. If I deliberately close the simulator connection, does the simulation keep running in the background or stop immediately?**
  *(background-run vs. immediate halt)*

* **2b. If I deliberately close the trainer connection, does learning continue in my absence, and what happens to the simulators that were attached?**
  *(trainer headless-run and simulator behaviour)*

  
* **2c. When a session finally ends, is the live simulation or training state saved anywhere, or must I checkpoint it myself?**
  *(state persistence & resume support)*

### Answers

---
**2a — Closing the simulator connection:**
* Once the simulator socket closes, every environment inside it stops stepping—no “headless” progress is made. The backend parks the simulator container for roughly two minutes while it waits for a new socket with the same session token. During this parking period credits keep ticking; if you manage to reconnect, the simulation resumes exactly where it left off. If no reconnection arrives within the ~2 minute grace window, the container is torn down, resources are freed, and billing stops.
* Because the trainer relies on periodic “fan-out” messages from its simulators, it also watches that same two-minute timer; if no simulator comes back before the window expires, the trainer automatically ends the training session as well.

---
**2b — Closing the trainer connection:**
* Closing the trainer halts its learning loop immediately—no gradients, no logging, no new commands. Connected simulators detach the environment runners that belonged to that trainer and hold them for roughly two minutes; if the trainer does not return within that grace window those runners are discarded. 
* The simulator itself does not automatically terminate, because a single simulator can service multiple trainers: it keeps stepping environments for any other trainers that are still online. Even all trainers have disconnected and none of them to reconnect, the simulator container still be alive and wait for a new connection.

---
**2c — Session state persistence:**
* RemoteRL Cloud does **not** snapshot environment RAM or trainer variables for you. When the session is finally cleaned up, all in-memory state disappears. Persist anything you care about (model weights, replay buffers, metrics) from inside your own code—e.g., call `model.save()` or write to S3—so you can resume from a checkpoint in a fresh session later.


## 3. Supported Regions and Performance

* **3a. Which cloud regions does RemoteRL Cloud currently run in?**
  *(geographic coverage)*

* **3b. Can I steer my session to a specific region, and which choice gives the lowest latency?**
  *(region selection & best performance)*

* **3c. When many users are online, how does the platform stop workloads from slowing each other down?**
  *(resource isolation under heavy load)*

---

### Answers

**3a — Geographic footprint**
* RemoteRL Cloud currently operates production clusters in the **United States** (East & West coasts), **South Korea**, **Japan**, **Germany**, **Singapore**, and **Brazil**. This provides coverage across **North America, Asia-Pacific, Europe, and South America**, so most users can reach a nearby point of presence with minimal latency.
If demand arises elsewhere, our **auto-deploy and maintenance system** can spin up a new regional cluster in under an hour, with no hard limit on how many locations we can add.

---
**3b — Picking the fastest region**
* Yes. By default the SDK points at a canonical URL that routes you to the region bound to your account, but you can override this and dial a specific endpoint (e.g. point Europe-based laptops at the US-West relay or the Seoul relay). Latency is always lowest when you choose the region **closest to your physical location**; the handshake will even auto-redirect if it detects a mismatch, adding only a few hundred milliseconds at connect-time.

---

**3c — Performance under heavy load**
* The Cloud Service’s role is for a **network relay**—all heavy computation runs on the trainer and simulator processes you start on hardware you own or rent. The relay layer is stateless and scales out horizontally: when traffic grows, more relay nodes come online and existing sessions are transparently rebalanced, so latency and throughput stay steady. 

---


## 4. Geographic Mobility (Travel & VPN)
* **4a. Can I sign in and run RemoteRL Cloud from any country, or are there location blocks?**
  *(travelling with the same account & API key)*

* **4b. Will everything still work if I connect through a VPN, and does that change anything important?**
  *(VPN compatibility & side-effects)*

* **4c. If I connect from a region that is far from the nearest relay (or route traffic through a distant VPN exit), what impact will that have on latency and overall experience?**
  *(cross-region performance)*

---

### Answers

**4a — Access while travelling**
* RemoteRL Cloud imposes **no geo-locks** on accounts or API keys. You can log in from any country and the session will automatically attach to the closest relay node, so nothing special is required when you move locations. Our system will seamlessly immigrate your session to the nearest server region and carry over your data credits without issue. This means your experience should remain smooth while traveling. 

**4b — Using a VPN**
* Your session will open on the nearest server region, and there are no inherent restrictions or failures when using a VPN. The only thing that changes is round-trip time: if you connect from farther away, you may notice a few extra milliseconds of latency, but your credits and session behavior remain the same. However, switching regions frequently can introduce a delay of 30 minutes to 1 hour before you are routed to the new nearest relay.

**4c — Cross-region performance**
* Latency scales with physical distance: within the same region you’ll typically see 10-50 ms RTT, whereas trans-ocean links can add 100 ms + of delay. RemoteRL’s relay layer is stateless and horizontally scaled, so throughput (frames per second) and data quality stay constant—the only variable is the extra time each `env.step()` round-trip takes. 


## 5. Interface Indicators (Data Usage Banner)

* **5a. What does the sky-blue “MB” banner actually show — data used or data left?**
* **5b. How often is that number refreshed on-screen?**
* **5c. Is the figure a running total for the session, or an instantaneous bandwidth meter?**
* **5d. Besides time, are there data or other resource caps tied to that banner?**

---

#### Answers

**5a — Meaning of the “MB” value**
* The banner shows the **megabytes you still have available to spend** in the current session. Each time the relay forwards another block of traffic, that counter ticks **downward**, letting you see at a glance how much data credit remains.

**5b — Update frequency**
* The banner refreshes on a **five-minute interval**. Traffic is still metered continuously in the background, but the on-screen number is only recomputed and pushed to the UI once every 5~10 minutes, so you’ll see it step down in chunks rather than change in real time.

**5c — Cumulative vs. instantaneous**
* It is a **live countdown of remaining credit**, not a bandwidth-per-second gauge. The value begins at your session’s data allotment, drops throughout the session, and resets to the full allotment only when you open a brand-new session (or purchase more credit). It never shows instantaneous throughput.

**5d — Related limits**
* When the counter reaches zero, the server slows your relay traffic dramatically and may eventually pause it altogether until you add more data credit. Other account-level limits (such as the maximum number of concurrent simulators or environment runners) safeguard overall relay capacity, but once you’re connected your session is never throttled for capacity reasons—only data exhaustion or an explicit disconnect can interrupt it.

## 6. Connection Technology & Architecture

1. **6a. What transport / protocol does RemoteRL use under the hood?**
3. **6b. Does the cloud do any heavy computation, or is that all on my own machines?**
5. **6e. How is the link secured against eavesdropping or session hijack?**

---

### Answers

**6a — Underlying protocol**
* RemoteRL traffic runs over **TLS-encrypted WebSockets (wss\://)**. The relay layer is stateless: every message is either a small JSON control frame (handshake, ping, etc.) or a binary blob carrying observations, rewards, or actions. This custom framing keeps latency low and avoids the overhead of generic remote-desktop stacks (RDP, VNC, WebRTC).

---
**6b — Where the compute happens**
* The relay does **no simulation or training work**. All heavy lifting—physics engines, neural-network inference/updates, logging—happens on the **trainer and simulator processes you start on hardware you own or rent** (laptop, on-prem cluster, cloud VM, robot SBC, …). RemoteRL only forwards bytes. That means other users can’t steal your GPU time, and your session never slows down because someone else is using the service.

---
**6e — Security**

* **Encryption:** All sockets are **TLS-protected**, so payloads are unreadable on the wire.
* **Auth:** Each connection presents an **API key** plus a session-specific token; without both, the relay refuses the handshake.
* **Isolation:** Simulators and trainers run in separate OS processes (or containers) on your own machines; the relay never stores payloads, and sessions are keyed so that only your authorised peers can attach or reconnect. In short, the link is end-to-end encrypted and gate-kept.

---
