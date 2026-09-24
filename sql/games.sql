-- Staging Parquet -> marts. Brace fields are quoted file paths filled in by eurohoops.marts.
CREATE OR REPLACE TABLE games AS
SELECT 'euroleague' AS competition, * FROM read_parquet({euroleague_games})
UNION ALL BY NAME
SELECT 'gbl' AS competition, * FROM read_parquet({gbl_games});

CREATE OR REPLACE TABLE teams AS
SELECT 'euroleague' AS competition, * FROM read_parquet({euroleague_teams})
UNION ALL BY NAME
SELECT 'gbl' AS competition, * FROM read_parquet({gbl_teams});
