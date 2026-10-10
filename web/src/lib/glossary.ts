// The Glossary's terms: only what the live pages show (column heads, tooltips, labels). The page
// renders from this list, and `shownOn` names the pages where each term appears so a later check
// can confirm none has been left behind when a page changes.

export type GlossaryGroup =
  | "Box score"
  | "Shooting"
  | "Rates and samples"
  | "Team ratings"
  | "Forecasts"
  | "Player projections";

export interface GlossaryTerm {
  /** Stable anchor: /glossary/#ts-pct. Never rename one; other pages may link to it. */
  id: string;
  /** The label as the pages print it. */
  term: string;
  /** The long name, where the label is an abbreviation. */
  name?: string;
  group: GlossaryGroup;
  /** One or two plain sentences. */
  def: string;
  formula?: string;
  /** Pages (route patterns under the site root) where the term is shown. */
  shownOn: string[];
}

export const GROUPS: GlossaryGroup[] = [
  "Box score",
  "Shooting",
  "Rates and samples",
  "Team ratings",
  "Forecasts",
  "Player projections",
];

const PL = "players";
const PP = "players/[slug]";
const TI = "teams";
const TP = "teams/[code]";
const LD = "leaders";
const CM = "compare";
const FC = "forecasts";
const HM = "home";
const ST = "standings";

export const TERMS: GlossaryTerm[] = [
  // Box score
  { id: "gp", term: "GP", name: "Games played", group: "Box score", def: "The games a player appeared in.", shownOn: [PL, PP, CM] },
  { id: "gs", term: "GS", name: "Games started", group: "Box score", def: "The games a player was in the starting five.", shownOn: [PP, CM] },
  {
    id: "min",
    term: "MIN",
    name: "Minutes",
    group: "Box score",
    def: "Minutes played per game. It stays a per-game figure even when other columns are scaled to 36 minutes or 100 possessions.",
    shownOn: [PL, PP, TP],
  },
  { id: "pts", term: "PTS", name: "Points", group: "Box score", def: "Points scored.", shownOn: [PL, PP, TP, LD, CM] },
  { id: "reb", term: "REB", name: "Rebounds", group: "Box score", def: "Rebounds, offensive and defensive together.", shownOn: [PL, PP, TP, LD, CM] },
  {
    id: "ast",
    term: "AST",
    name: "Assists",
    group: "Box score",
    def: "Passes that lead straight to a made basket, as the official scorer records them.",
    shownOn: [PL, PP, TP, LD, CM],
  },
  { id: "stl", term: "STL", name: "Steals", group: "Box score", def: "Times a player takes the ball from an opponent.", shownOn: [PL, PP, TP] },
  { id: "blk", term: "BLK", name: "Blocks", group: "Box score", def: "Opponent shots a player blocks.", shownOn: [PL, PP, TP] },
  { id: "tov", term: "TOV", name: "Turnovers", group: "Box score", def: "Times a player loses the ball to the other side. Fewer is better.", shownOn: [PL, PP, TP] },
  {
    id: "pir",
    term: "PIR",
    name: "Performance Index Rating",
    group: "Box score",
    def: "The EuroLeague's own one-number summary of a player's game. Good plays add to it and bad ones take away.",
    shownOn: [PL, PP, TP, LD],
  },
  {
    id: "plus-minus",
    term: "+/-",
    name: "Plus-minus",
    group: "Box score",
    def: "Points the player's team scored minus points it allowed while he was on court.",
    shownOn: [PL, LD],
  },

  // Shooting
  { id: "fga", term: "FGA", name: "Field-goal attempts", group: "Shooting", def: "Shots from the floor, twos and threes. Free throws are not counted.", shownOn: [PL] },
  { id: "fg-pct", term: "FG%", name: "Field-goal percentage", group: "Shooting", def: "Field goals made per attempt.", shownOn: [PL, PP] },
  { id: "2p-pct", term: "2P%", name: "Two-point percentage", group: "Shooting", def: "Two-pointers made per attempt.", shownOn: [PL] },
  { id: "3pa", term: "3PA", name: "Three-point attempts", group: "Shooting", def: "Shots tried from behind the three-point line.", shownOn: [PL] },
  { id: "3p-pct", term: "3P%", name: "Three-point percentage", group: "Shooting", def: "Three-pointers made per attempt.", shownOn: [PL, PP, TP, LD] },
  { id: "fta", term: "FTA", name: "Free-throw attempts", group: "Shooting", def: "Free throws taken.", shownOn: [PL] },
  { id: "ft-pct", term: "FT%", name: "Free-throw percentage", group: "Shooting", def: "Free throws made per attempt.", shownOn: [PL, PP, TP, LD] },
  {
    id: "efg-pct",
    term: "eFG%",
    name: "Effective field-goal percentage",
    group: "Shooting",
    def: "Field-goal percentage that credits a made three at one and a half times a made two, because it is worth that much more.",
    formula: "(FGM + 0.5 × 3PM) / FGA",
    shownOn: [PL, LD, TP],
  },
  {
    id: "ts-pct",
    term: "TS%",
    name: "True shooting percentage",
    group: "Shooting",
    def: "Points per shooting chance, with threes and free throws counted. It is the fairest single measure of how efficiently someone scores.",
    formula: "PTS / (2 × (FGA + 0.44 × FTA))",
    shownOn: [PL, PP, TP, LD, CM],
  },
  {
    id: "usg-pct",
    term: "USG%",
    name: "Usage percentage",
    group: "Shooting",
    def: "The share of his team's shots, free-throw trips and turnovers that a player ends while he is on court.",
    formula: "(FGA + 0.44 × FTA + TOV) as a share of the team's, while on court",
    shownOn: [PL],
  },
  {
    id: "3pa-rate",
    term: "3PA rate",
    group: "Shooting",
    def: "Three-point attempts per field-goal attempt: how much of a player's shooting comes from three.",
    shownOn: [PP],
  },
  {
    id: "ft-rate",
    term: "Free-throw rate",
    group: "Shooting",
    def: "How often a side gets to the line compared with how often it shoots from the floor. The team rankings count free throws made per 100 field-goal attempts. The four-factors table and the shot-twin comparison count free-throw attempts per field-goal attempt.",
    shownOn: [TP, PP],
  },
  {
    id: "opp-efg",
    term: "Opponent eFG%",
    group: "Shooting",
    def: "The effective field-goal percentage that opponents shoot against a club. Lower is better.",
    shownOn: [TP],
  },
  {
    id: "opp-ft-rate",
    term: "Opponent free-throw rate",
    group: "Shooting",
    def: "The free-throw rate a club allows its opponents. Lower is better.",
    shownOn: [TP],
  },

  // Rates and samples
  { id: "per-game", term: "Per game", group: "Rates and samples", def: "The average over the games played. This is the default view.", shownOn: [PL, LD, CM] },
  {
    id: "per-36",
    term: "Per 36",
    group: "Rates and samples",
    def: "Each figure scaled to 36 minutes on court, so a starter and a bench player can be compared at the same playing time.",
    shownOn: [PL, CM],
  },
  {
    id: "per-100",
    term: "Per 100",
    group: "Rates and samples",
    def: "Each figure scaled to 100 possessions, which removes the effect of playing for a fast or a slow team.",
    shownOn: [PL, PP],
  },
  {
    id: "possession",
    term: "Possession",
    group: "Rates and samples",
    def: "One turn with the ball for a team. It ends with a shot, a turnover or a trip to the line.",
    shownOn: [TP, PL, PP],
  },
  { id: "totals", term: "Totals", group: "Rates and samples", def: "The season or career sum rather than an average.", shownOn: [LD, CM] },
  {
    id: "last-n",
    term: "Last 5, Last 10, Last 20",
    group: "Rates and samples",
    def: "Only a player's most recent games, to show current form.",
    shownOn: [PL, PP],
  },

  // Team ratings
  { id: "ortg", term: "ORtg", name: "Offensive rating", group: "Team ratings", def: "Points a team scores per 100 possessions.", shownOn: [TI, TP] },
  {
    id: "drtg",
    term: "DRtg",
    name: "Defensive rating",
    group: "Team ratings",
    def: "Points a team allows per 100 opponent possessions. Lower is better.",
    shownOn: [TI, TP],
  },
  {
    id: "net-rating",
    term: "Net rating",
    group: "Team ratings",
    def: "Offensive rating minus defensive rating. The Teams page ranks clubs by it.",
    shownOn: [TI, TP],
  },
  { id: "pace", term: "Pace", group: "Team ratings", def: "Possessions per 40 minutes. A high number means a fast game.", shownOn: [TI, TP] },
  {
    id: "four-factors",
    term: "Four factors",
    group: "Team ratings",
    def: "The four things that decide most games: shooting (eFG%), turnovers, rebounding and free throws. Each is shown for a club's offence and for its defence.",
    shownOn: [TP],
  },
  {
    id: "tov-rate",
    term: "Turnover rate",
    group: "Team ratings",
    def: "Turnovers per 100 plays. Lower is better on offence.",
    formula: "TOV / (FGA + 0.44 × FTA + TOV)",
    shownOn: [TP],
  },
  {
    id: "turnovers-forced",
    term: "Turnovers forced",
    group: "Team ratings",
    def: "Opponent turnovers per 100 opponent plays. Higher is better.",
    shownOn: [TP],
  },
  {
    id: "orb-pct",
    term: "Offensive rebound %",
    group: "Team ratings",
    def: "The share of a club's own misses that it rebounds.",
    formula: "own offensive rebounds / (own offensive + opponent defensive rebounds)",
    shownOn: [TP],
  },
  {
    id: "drb-pct",
    term: "Defensive rebound %",
    group: "Team ratings",
    def: "The share of opponents' misses that the club rebounds.",
    shownOn: [TP],
  },
  {
    id: "schedule-strength",
    term: "Schedule strength",
    name: "Mean opponent rating",
    group: "Team ratings",
    def: "The average power rating of the opponents on a club's schedule. A higher number means a harder schedule.",
    shownOn: [TP],
  },

  // Forecasts
  {
    id: "power-rating",
    term: "Power rating",
    name: "Elo rating",
    group: "Forecasts",
    def: "One number for team strength, an Elo rating. It rises after a win and falls after a loss, by more for an upset or a big margin. 1500 is an average team.",
    shownOn: [HM],
  },
  {
    id: "win-probability",
    term: "Win probability",
    group: "Forecasts",
    def: "The chance, set before tip-off, that the home side wins. It comes mainly from the gap between the two power ratings.",
    shownOn: [HM, FC],
  },
  {
    id: "expected-margin",
    term: "Expected margin",
    group: "Forecasts",
    def: "The points by which the home side is forecast to win, or to lose if the number is negative.",
    shownOn: [FC],
  },
  {
    id: "picks-right",
    term: "Picks right",
    group: "Forecasts",
    def: "How often the favourite won. A forecast is right when the side given more than a 50% chance wins the game.",
    shownOn: [FC],
  },
  {
    id: "projected-wins",
    term: "Wins",
    name: "Projected wins",
    group: "Forecasts",
    def: "On the Standings page, the wins a club is projected to add over the rest of the season, averaged across many simulated seasons.",
    shownOn: [ST],
  },
  {
    id: "season-chances",
    term: "Top 6, Play-in, Playoffs, Final Four, Final, Title",
    group: "Forecasts",
    def: "The share of simulated seasons in which a club finishes in that place or reaches that stage. Sixth place or better goes straight to the playoffs; seventh to tenth play in.",
    shownOn: [ST],
  },

  // Player projections
  {
    id: "projection",
    term: "Projected",
    group: "Player projections",
    def: "The central estimate for a player's next season, built from his past seasons.",
    shownOn: [PP],
  },
  {
    id: "interval-80",
    term: "80% interval",
    group: "Player projections",
    def: "The range a figure should land in four times out of five. A wide range means the player has little history to go on.",
    shownOn: [PP],
  },
  {
    id: "impact",
    term: "Impact",
    group: "Player projections",
    def: "The net points per 100 possessions a player adds against an average player, offence and defence together. Zero is average.",
    shownOn: [PP],
  },
  {
    id: "shot-twin",
    term: "Shot twin",
    name: "Match",
    group: "Player projections",
    def: "The EuroLeague player-season since 2007-08 whose shooting looks most like a player's: where the shots come from, how well they go in against the league, and how often he shoots threes and free throws. A match of 100 is identical; two random seasons score about 37.",
    shownOn: [PP],
  },
];
