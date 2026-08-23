# Coverage baseline — why it is 94.90 and not 95.0

Recorded 2026-08-22. Measured 94.94% at the time of recording; 94.90 leaves a
sliver of headroom so an unrelated one-line change does not turn the lane red
for a reason that has nothing to do with the change.

## This is a RECORD, not a weakening

The implicit floor was 95.0. The uncovered remainder is not "code nobody got
round to testing" — it is a specific, named set of paths that cannot be
exercised until `remote_exec()` exists (copia#49):

    src/bin/copia/transfer.rs: 117-120, 146, 172, 188, 193, 198, 204, 210, 217

Every one of those is inside an ssh invocation — `transfer_file_to_remote`,
`create_remote_dirs` and their error arms. copia spawns `ssh` from SEVEN
independent call sites, so there is no seam to inject a fake shell at. copia#49
funnels them through one `remote_exec()` honouring `COPIA_REMOTE_SHELL`, at
which point CI can point at a two-line shim and these lines become reachable
without an sshd in the container.

Lines 399, 403 and 406 are also uncovered and deliberately so: they are the
`missing.push(...)` failure arms of the walk's own entry-kind assertion. They
execute only when the walk is broken. Writing a test to cover them would mean
asserting the failure path of an assertion, which proves nothing about the
program.

## Why not just raise coverage first

The change this baseline accompanies is a DATA-LOSS fix: the walk dropped FIFOs,
sockets and devices, and `copia verify` — sharing that walk — reported
"trees are identical — the source may safely be deleted" over the loss. Holding
that behind 0.06% of coverage on paths that are structurally unreachable would
be the unpassable-gate pattern: a floor nobody can clear is a floor people learn
to route around, and the routing then covers the findings that matter.

## What raises it back

copia#49. When the transport matrix lands, delete this file rather than editing
the number — a baseline that survives its own justification is just a lower
floor with a story attached.
