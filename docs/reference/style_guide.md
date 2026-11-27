# Style guide

- Keep domain functions small, typed, deterministic, and documented.
- Validate values at public boundaries with messages that include the bad value.
- Prefer immutable input semantics: copy or return new arrays rather than
  mutating caller-owned data.
- Keep scripts thin; if a script contains statistical logic, move it into `src/`
  and test it as domain behavior.
- Use clear scientific names: FDR, FDP, FWER, TPR, null, alternative, and
  calibration are defined in the methodology rather than used as slogans.
- Use real data and no mocks in tests.
- Generated outputs are never hand-edited.
