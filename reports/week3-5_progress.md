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
iteration 4 | D5 odds recorder + market column + daily step | gate 1-6, actionlint 1.7.12 (both workflows), score x2 | green (1-6, 16); check 12 live call BLOCKED (401); step 7 not run | 59aaec8
iteration 5 | D6 OddsPortal spike | 2 requests (robots.txt 307 -> greece.html notice); doc ends in NO-GO | done (no code; gate unchanged since iteration 4) | 07a7d7f
iteration 6 | D7 stint validation | gate 1-6, eurohoops stints x2 byte-identical (13) | green (1-6, 13); sample 50/50 pass | a177086
iteration 7 | D4 site sections + D8 CI web job | full gate 1-7 (web built from fixture), actionlint, 8 screenshots (1440/390 x light/dark x EL/GBL), 0 px horizontal overflow | green | 37b7618

Note (iteration 7): the user authorised stopping the stale astro preview (PIDs 8200/10892); done.
Screenshots via Playwright + installed Edge against a local copy served under the /EuroHoops-Analytics base.
