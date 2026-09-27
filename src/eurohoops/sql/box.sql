-- GBL box scores (only built after `eurohoops ingest --competition gbl --details` or `--pbp`).
CREATE OR REPLACE TABLE player_box AS SELECT * FROM read_parquet({gbl_player_box});
CREATE OR REPLACE TABLE team_box AS SELECT * FROM read_parquet({gbl_team_box});
CREATE OR REPLACE TABLE box_fill AS SELECT * FROM read_parquet({gbl_box_fill});

-- One row per team in every played, non-forfeit GBL game.
CREATE OR REPLACE VIEW gbl_sides AS
SELECT game_id, season, home AS team, home_score AS score
FROM games WHERE competition = 'gbl' AND played AND NOT forfeit
UNION ALL
SELECT game_id, season, away, away_score
FROM games WHERE competition = 'gbl' AND played AND NOT forfeit;

-- What each team's official (ESAKE) box score says; PBP-filled rows never count here.
CREATE OR REPLACE VIEW box_checks AS
WITH players AS (
    SELECT
        game_id,
        team,
        sum(points) AS player_points,
        sum(seconds) / 60.0 AS minutes,
        count_if(fg2m > fg2a OR fg3m > fg3a OR ftm > fta) AS bad_shot_lines,
        count_if(points <> 2 * fg2m + 3 * fg3m + ftm) AS bad_point_lines
    FROM player_box WHERE source = 'esake' GROUP BY game_id, team
),
teams AS (SELECT game_id, team, total_points FROM team_box WHERE source = 'esake')
SELECT
    s.season, s.game_id, s.team, s.score, t.total_points,
    p.player_points, p.minutes, p.bad_shot_lines, p.bad_point_lines
FROM gbl_sides AS s
LEFT JOIN teams AS t USING (game_id, team)
LEFT JOIN players AS p USING (game_id, team);

-- Every team of a short game: how it was filled from PBP and the points it now adds up to.
CREATE OR REPLACE VIEW box_fill_checks AS
WITH pbp_teams AS (SELECT game_id, team, total_points FROM team_box WHERE source = 'pbp'),
all_players AS (SELECT game_id, team, sum(points) AS points FROM player_box GROUP BY game_id, team)
SELECT
    f.season, f.game_id, f.team, f.fill, f.detail, s.score,
    CASE f.fill
        WHEN 'team_totals_from_pbp' THEN t.total_points
        WHEN 'missing_player_from_pbp' THEN p.points
    END AS filled_points
FROM box_fill AS f
JOIN gbl_sides AS s USING (game_id, team)
LEFT JOIN pbp_teams AS t USING (game_id, team)
LEFT JOIN all_players AS p USING (game_id, team);
