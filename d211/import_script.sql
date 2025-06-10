CREATE TABLE IF NOT EXISTS public.populations
(
    id serial NOT NULL,
    zip_code int NOT NULL,
    population_count int,
	location_id int,
    PRIMARY KEY (id),
	FOREIGN KEY (location_id) REFERENCES public.location (location_id)
);
TRUNCATE populations;

CREATE TABLE IF NOT EXISTS public.incomes
(
    id serial NOT NULL,
    zip_code int NOT NULL,
    median_income int,
    margin_error int,
    location_id int,
	PRIMARY KEY (id),
	FOREIGN KEY (location_id) REFERENCES public.location (location_id)
);
TRUNCATE incomes;


COPY populations(zip_code, population_count)
from 'C:\Users\Public\Downloads\d211-main\populationPerZip.csv'
DELIMITER ','
CSV HEADER;

UPDATE populations
SET location_id=subquery.location_id
FROM (SELECT location_id, zip FROM location) AS subquery
WHERE populations.zip_code=subquery.zip;

Copy incomes(zip_code, median_income, margin_error) 
from 'C:\Users\Public\Downloads\d211-main\MI_ZIP.csv'
DELIMITER ','
CSV HEADER;

UPDATE incomes
SET location_id=subquery.location_id
FROM (SELECT location_id, zip FROM location) AS subquery
WHERE incomes.zip_code=subquery.zip;
