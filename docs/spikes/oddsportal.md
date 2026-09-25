# Spike: OddsPortal for historical EuroLeague closing odds (PLAN §1.3, §12 item 5)

Date: 2026-09-25. Timebox: one working day. Time used: under an hour. Requests: **2**.

## What was tried
The brief says to read robots.txt and the ToS **first**, so the first request was for robots.txt.

| # | Request (identifying User-Agent, 3 s apart) | Response |
|---|---|---|
| 1 | `GET https://www.oddsportal.com/robots.txt` | `307 Temporary Redirect` → `location: https://www.oddsportal.com/greece.html` (nginx, Varnish, `Date: Fri, 25 Sep 2026 12:45:15 GMT`) |
| 2 | the same, following the redirect | `200`: the page *"OddsPortal - Greece Availability Notice"* |

The notice, in Greek, with translation:

> Δυστυχώς, το OddsPortal δεν είναι διαθέσιμο στην Ελλάδα. [...] το OddsPortal δεν είναι
> πλέον διαθέσιμο στη χώρα σας, λόγω περιορισμών της νομοθεσίας. Θα θέλαμε να σας προτείνουμε
> να χρησιμοποιήσετε τον ιστότοπο του συνεργάτη μας www.flashscore.gr.
>
> *"Unfortunately, OddsPortal is not available in Greece. [...] OddsPortal is no longer
> available in your country because of legislative restrictions. We suggest our partner
> site www.flashscore.gr."*

## What blocked
- **OddsPortal geo-blocks Greece on purpose, for legal reasons.** Even `robots.txt` is
  redirected, so from the project's location there is no site to read, robots.txt included.
- Getting around this (VPN, proxy, running the scraper from a non-Greek CI runner) would
  deliberately evade a restriction the operator says the law requires. That's out of bounds
  for this project, whatever the site's robots.txt or ToS say elsewhere. **I didn't read
  the ToS through another vantage point**: the answer is already no, and fetching it that
  way would itself be a workaround.
- A scheduled GitHub Actions runner (US IP) would probably get the full site. I didn't try
  it, for the same reason.

## Volume / cost estimate
Not reached. For reference, the plan assumed a JavaScript-rendered archive with one page per
match: about 380 EuroLeague regular-season games per season, so thousands of browser-rendered
requests for a 2015-2026 backfill, at ≥ 3 s each. That's hours of headless-browser time
and fragile parsing.

## Recommendation: **NO-GO**
- Don't build an OddsPortal scraper, now or later, as long as the site is unavailable in Greece.
- The market benchmark stays **forward-only**: The Odds API recorder (`docs/data/odds.md`),
  from the first valid call onward.
- For historical closing odds, the only clean option is The Odds API's `/historical`
  endpoint. Its pricing page lists historical odds even on the free plan, which contradicts
  PLAN §1.3's "(paid)". This is unverified: check it with one call once the key works, and
  weigh the credit cost before any backfill.
- flashscore.gr (the suggested partner) wasn't assessed. It's a separate spike, with the
  same robots.txt/ToS-first rule.
