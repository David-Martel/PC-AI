# Lenovo service request draft — not submitted

**Device:** ThinkPad P1 Gen 7, machine type/model 21KV0014US. Supply serial number
privately when submitting.

For several months the internal keyboard has intermittently lost keys despite
Windows reboots. Both internal Shift keys have failed together while ordinary A
works and USB Shift works. Shift later recovered in local applications and RDP.
Internal Right Arrow then failed locally and in RDP while USB Right Arrow worked;
later Left, Down and Right all failed. Down recovered after a hard press. The laptop
travels frequently in horizontal and vertical orientations; no precipitating
physical event is confirmed.

Device-aware Windows traces observed ordinary internal input and USB Shift during
reported internal Shift failure. They do not prove the failing component. Later
arrow captures lack matched physical trials and are inconclusive. PnP enumerates
the internal keyboard on the i8042 path. BIOS1.22/EC1.15 was observed. Individual
Lenovo accessory, UltraslimOSD and Logitech isolation trials reported no improvement;
all utilities were restored. Software causation remains unresolved.

Please inspect the keyboard assembly, both keyboard cables/connectors and locking
mechanisms, cable routing, contact damage/corrosion and mechanical fit. The current
HMM identifies keyboard connectors4/5 on printed page51. If inspection/reseating
does not resolve reproducible failures, test a known-good compatible keyboard
assembly before concluding the system board is faulty. Please check for applicable
internal service bulletins and use serial-specific parts compatibility.

An out-of-Windows failed-key result has not yet been obtained. A short pass should
not close this intermittent fault; retain a before/after key-test record and the
conditions under which the problem returns. No raw typing captures or identifiers
are attached to this draft.
