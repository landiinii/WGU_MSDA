select
    shopifycreatedat::date as date,
    count(distinct shopify_id) as count
from 
    staging.mongodb.stg_orders
where 
    team = '668d945b2b2b13d33b7ed1cf' --'657357d83b0676000728b06c'
    and shopifycreatedat::date >= '2014-10-14'
    and shopifycreatedat::date <= '2024-12-31'
group by all
order by date
;

with days_with_counts as (
    select
        team,
        shopifycreatedat::date as unique_dates,
        count(*) as count_on_date
    from staging.mongodb.stg_orders
    group by all
)
select
    team,
    count(distinct unique_dates) as count_unique_dates,
    avg(count_on_date) as avg_count_on_date
from days_with_counts
group by team
order by count_unique_dates desc, avg_count_on_date desc



