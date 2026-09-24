-- GBL box scores (only built after `eurohoops ingest --competition gbl --details`).
CREATE OR REPLACE TABLE player_box AS SELECT * FROM read_parquet({gbl_player_box});
CREATE OR REPLACE TABLE team_box AS SELECT * FROM read_parquet({gbl_team_box});

-- One row per team in every played, non-forfeit GBL game, with what its box score says.
CREATE OR REPLACE VIEW box_checks AS
WITH sides AS (
    SELECT game_id, season, home AS team, home_score AS score
    FROM games WHERE competition = 'gbl' AND played AND NOT forfeit
    UNION ALL
    SELECT game_id, season, away, away_score
    FROM games WHERE competition = 'gbl' AND played AND NOT forfeit
),
players AS (
    SELECT
        game_id,
        team,
        sum(points) AS player_points,
        sum(seconds) / 60.0 AS minutes,
        count_if(fg2m > fg2a OR fg3m > fg3a OR ftm > fta) AS bad_shot_lines,
        count_if(points <> 2 * fg2m + 3 * fg3m + ftm) AS bad_point_lines
    FROM player_box GROUP BY game_id, team
)
SELECT
    s.season, s.game_id, s.team, s.score, t.total_points,
    p.player_points, p.minutes, p.bad_shot_lines, p.bad_point_lines
FROM sides AS s
LEFT JOIN team_box AS t USING (game_id, team)
LEFT JOIN players AS p USING (game_id, team);
