# Veritas journal audit

Beliefs: 14 active, 3 retracted, 17 total. Consistent: True.

No live contradictions. Memory is consistent.

## Retractions this run
- Use SQLite for the journal index: single-file, queryable via SQL, Python stdlib only, no n
  ↳ contradicted by 'Avoid SQLite for the journal index: flat files keep the repo human-inspectable and diffable; 
- Do not cache the shelf index in SQLite; regenerate the shelf markdown on every snapshot in
  ↳ contradicted by 'Build the shelf index cache in SQLite for fast queries over deliverables' (new entrenchment 0
- Build the shelf index cache in SQLite for fast queries over deliverables
  ↳ contradicted by 'RESOLVED (shelf index cache): chose no-cache over SQLite cache because the shelf regenerates 
