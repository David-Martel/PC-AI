# Exact-model hardware review — September 12, 2026

**Machine verified locally:** ThinkPad P1 Gen 7, 21KV0014US. Latest reported symptom:
native Left, Down and Right fail; previously both Shift keys failed and recovered.
USB Right worked in the preceding matched local/RDP comparison. A physical defect
or obstruction has not been observed remotely.

**Subsequent user observation:** Down started working again after a hard press.
The laptop travels frequently in vertical and horizontal orientations; the user is
unsure of a precipitating event. Pressure-associated recovery increases the priority
of a contact/assembly fault, but one uncontrolled recovery cannot establish causation.
Do not repeat forceful presses or flex the chassis as a diagnostic method.

## What Lenovo's hardware documentation establishes

The current [P1 Gen 7 Hardware Maintenance Manual](https://download.lenovo.com/pccbbs/mobiles_pdf/p1_gen7_hmm_en.pdf)
was downloaded and its diagrams visually inspected. Printed page51 (PDF56), table7,
identifies **keyboard connectors4 and5 and two keyboard-assembly cables**. The
separate touchpad connector is8. This is a connector-location diagram, not a
keyboard matrix schematic; it supplies no per-key row/column mapping or cable pinout.
Do not assume one cable serves Shift or that all affected keys share a row.

Printed page49 (PDF54) classifies the keyboard bezel assembly as neither a self-service
nor optional-service CRU. Printed page90 (PDF95) lists major access removals,
including storage, CAMM2 components, battery, the applicable system-board assembly,
bracket and touchpad. Full keyboard replacement is substantial service work.
The cable diagram provides specific inspection targets; it does not prove either
connector is loose or establish their condition on this machine.

No applicable public Lenovo keyboard-pressure bulletin, electrical matrix map or
verified mechanical defect notice was located in the targeted search. This is a
search limit, not a claim that no internal service bulletin exists.

## Physical candidates — hypotheses to test

| Candidate | Discriminating evidence |
| --- | --- |
| Debris, damaged scissor mechanism or sticking keycap | Abnormal travel/return or visible obstruction at the affected key. Normal travel makes a simple travel blockage less compelling but does not validate electrical contacts. |
| Keyboard membrane/contact/trace fault | Several intermittent native keys with normal travel; confirmation requires physical testing or substitution of a known-good keyboard assembly. Physical adjacency alone does not establish electrical grouping. |
| Cable contact, latch, routing or assembly damage | Service inspection finds incomplete seating, damaged contacts/latch, corrosion or pinching; controlled reseating/replacement changes reproducible behavior. |
| Chassis loading or deformation | Failure consistently depends on ordinary supported placement or externally visible deck distortion. A flat rigid surface test is observational; do not flex the chassis or press harder to manufacture a result. |
| EC/system-board/native-driver path | Remains possible if the assembly is sound. Working USB and healthy PnP status do not isolate the failing component. |

These rankings are engineering inferences from the symptom, not Lenovo diagnoses.
Older SSD/touchpad EMI anecdotes do not establish a cause for the keyboard.

## Practical next actions

1. Compare normal key travel and spring-back, inspect for obstruction, and record
   whether the laptop is on a flat desk or a stand/clamp. Capture any connection to
   a spill, drop, cleaning or previous internal repair. Lenovo's
   [keyboard troubleshooting](https://support.lenovo.com/id/en/solutions/ht103985-keyboard-keys-may-not-work)
   includes checking lodged objects and cleaning with power off. Avoid keycap removal
   or chassis deformation as an exploratory test.
2. Obtain an OS-independent reproduction when convenient: the exact-model
   [user guide](https://download.lenovo.com/manual/p1_g7/user_guide/en/help_troubleshoot_or_diagnose_problems.html)
   specifies AC power and **F10 immediately at power-on** for UEFI Diagnostics.
   Select the keyboard test if present. A failure there excludes Windows/RDP/profile
   handling for that occurrence; a pass after reboot does not exclude an intermittent
   fault. Esc exits and restarts the computer. No reboot is being prescribed as a cure.
3. If failures persist, give Lenovo service the affected-key chronology, USB contrast,
   capture limits and this exact connector reference. Request inspection of both
   keyboard cables/latches and the assembly, followed by a known-good assembly test
   if inspection/reseating is inconclusive. Use serial-specific parts lookup privately
   before selecting a replacement; no FRU number has been guessed or part ordered.

The agent performed no disassembly, pressure manipulation, connector reseating,
firmware flash or hardware replacement. The user reported the hard press described
above. The physical condition still requires inspection.
