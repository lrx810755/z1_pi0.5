# Bottle preplanning Implementation Plan

**Goal:** Plan both loaded legs before lift without long waits.
**Architecture:** Private preplanned_transport.py extends ForwardTransport. Reuse existing MoveIt request timeouts and audit with explicit future start. Wire only private assistant.
**Spec:** ../specs/2026-11-09-preplanned-bottle.md

- [x] Tests: rejected second leg sends no arm action; planning uses first endpoint and closed gripper; execution adds no planning; stale second start rejected.
- [x] Implement preplanning, future-state trajectory audit, preplanned carry and stale guards.
- [x] Run whole private suite; review change and original hashes; reload at idle boundary, inspect first attempt without monitoring completion.

Verification:55 private tests pass; compile and launch preview pass; independent review found no blocking issues. Reviewer confirms separate cached legs retain physical checks (not a blended continuous trajectory). Reload readiness confirmed21:00:08. Original interface hashes unchanged. No claim of physical success.
