# Cup frustum collision implementation plan

Goal: replace the cup solid cylindrical collider with a solid convex frustum, as requested after discussing upper-wide/lower-narrow geometry. Keep the scanned visual and existing calibrated grasp settings.

Measured visual: height 60.970 mm, maximum radius 25.145 mm; bottom outer radius approximately 18 mm; grasp-height outer radius approximately 22 mm. Use bottom radius 18.0 mm, top radius 25.2 mm, height 61.0 mm, 64 sides. At z=34 mm radius is 22.013 mm. Preserve root origin at the base and the existing physics material.

- [x] Add a failing regression for tapered, closed, outward-facing mesh geometry and correct base origin.
- [x] Add a shared cup collider definition/authoring helper; wire scene builder and scanned asset metadata to it.
- [x] Replace only the cup collider in the currently used generated scene; synchronize current metadata. Save originals outside the repository.
- [x] Validate saved USD geometry, material, visibility, convex approximation and unchanged other scene content; run relevant and project tests; smoke-test physics.

Ruling: edit the active asset with exact backups because the user requested a live model change, and the checkout contains substantial unrelated existing work. No robot reimport or full-scene rebuild; those would overwrite unrelated changes. MoveIt retains its enclosing-cylinder approximation for conservative planning. Existing datasets/checkpoints are not rewritten.
