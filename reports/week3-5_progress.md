# Weeks 3–5 progress log

Decisions (user, 2026-09-25): D-a … D-h all defaults (see `prompts/week-3-5.md` §0).
Branch `week-3-5`, not pushed.

Format: `iteration N | deliverable | checks run | result | commit`

iteration 1 | D1 ECE + reliability | gate 1-6, backtest x2 reproduce (old keys), score x2 | green (1-6); step 7 NOT RUN, see note | b5b68f3

Note (iteration 1): step 7 `npm ci` fails with EPERM unlinking
`web/node_modules/@astrojs/compiler-binding-win32-x64-msvc/astro.win32-x64-msvc.node`: an
`astro preview --port 4321` server (PIDs 8200/10892, started 14:17 local, before this session)
holds it. Stopping it was denied by the auto-mode classifier, so the user has to stop it.
Web steps (7, 17, D4) wait for that.
iteration 2 | D2 CRPS + sigma + totals baseline | gate 1-6, backtest x2 reproduce, score x2, leakage | green (1-6); step 7 not run (preview server) | eba3e11
iteration 3 | D3 rolling-50 + warnings | gate 1-6, score x2 | green (1-6); step 7 not run (preview server) | 43ddaca

BLOCKED (check 12, 2026-09-25 ~12:50 UTC): The Odds API rejects the key in `.env`.
`GET /v4/sports/basketball_euroleague/events` (free endpoint, 1 call) -> HTTP 401
`{"message":"API key is not valid. ...","error_code":"INVALID_KEY"}`, no x-requests-* headers.
The `.env` value is 30 hex chars, no quotes/whitespace/CR; Odds API keys are normally 32, so it
looks truncated. Needs the user to re-copy the key. D5 is built and tested on a fake transport.
