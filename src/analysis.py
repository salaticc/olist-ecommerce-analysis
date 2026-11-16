"""Order-grain Olist analysis. Associations and scenarios are not causal effects."""
import numpy as np
import pandas as pd


def unique_key(frame, key):
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f'{key} must be nonmissing and unique')


def prepare_orders(orders, customers, items, reviews):
    orders, customers, items, reviews = [x.copy() for x in (orders, customers, items, reviews)]
    unique_key(orders, 'order_id'); unique_key(customers, 'customer_id')
    if items[['order_id','order_item_id']].isna().any().any() or items.duplicated(['order_id','order_item_id']).any():
        raise ValueError('Duplicate or missing order item key')
    if not set(orders.customer_id).issubset(set(customers.customer_id)):
        raise ValueError('Unknown customer key')
    for frame in (items,reviews):
        if frame.order_id.isna().any() or not set(frame.order_id).issubset(set(orders.order_id)):
            raise ValueError('Unknown or missing order_id in dependent table')
    for column in ('price','freight_value'):
        items[column] = pd.to_numeric(items[column],errors='raise')
        if not np.isfinite(items[column]).all() or items[column].lt(0).any():
            raise ValueError('Invalid item money; inspect source')
    if reviews.review_score.isna().any() or not reviews.review_score.isin([1,2,3,4,5]).all():
        raise ValueError('Review score outside 1..5')
    for col in ['order_purchase_timestamp','order_delivered_customer_date','order_delivered_carrier_date']:
        orders[col] = pd.to_datetime(orders[col],errors='raise')
    if orders.order_purchase_timestamp.isna().any():
        raise ValueError('Missing purchase timestamp')
    reviews['review_creation_date'] = pd.to_datetime(reviews.review_creation_date,errors='raise')
    if 'review_answer_timestamp' in reviews:
        reviews['review_answer_timestamp'] = pd.to_datetime(reviews.review_answer_timestamp,errors='raise')
    # Stable explicit tie-break: creation -> answer -> review id -> source order.
    sort_cols = [x for x in ['review_creation_date','review_answer_timestamp','review_id'] if x in reviews]
    reviews = reviews.sort_values(sort_cols,kind='stable',na_position='first')
    latest = reviews.drop_duplicates('order_id',keep='last')
    item_agg = items.groupby('order_id').agg(total_price=('price','sum'),total_freight=('freight_value','sum'),total_items=('order_item_id','count'),unique_sellers=('seller_id','nunique')).reset_index()
    result = orders.merge(item_agg,on='order_id',how='left',validate='one_to_one').merge(customers[['customer_id','customer_unique_id','customer_state']],on='customer_id',how='left',validate='many_to_one').merge(latest,on='order_id',how='left',validate='one_to_one')
    if len(result)!=len(orders) or not result.order_id.is_unique:
        raise AssertionError('Order grain changed')
    result['delivery_days'] = (result.order_delivered_customer_date-result.order_purchase_timestamp).dt.total_seconds()/86400
    chronology = result.order_delivered_carrier_date.isna() | result.order_delivered_customer_date.ge(result.order_delivered_carrier_date)
    result['is_valid_delivery'] = result.order_status.eq('delivered') & result.delivery_days.notna() & result.delivery_days.ge(0) & chronology
    result['order_month'] = result.order_purchase_timestamp.dt.to_period('M')
    audit = {'orders':len(orders),'review_rows':len(reviews),'latest_review_rows':len(latest),'removed_review_rows':len(reviews)-len(latest),'orders_without_items':int(result.total_items.isna().sum()),'invalid_delivered_dates':int((result.order_status.eq('delivered') & ~result.is_valid_delivery).sum()),'currency':'BRL','grain':'one row per order'}
    return result,audit


def cohort_retention(orders, cutoff='2018-08', min_cohort_size=10):
    cutoff = pd.Period(cutoff,freq='M')
    a = orders.loc[orders.order_status.eq('delivered') & orders.customer_unique_id.notna() & orders.order_month.le(cutoff), ['customer_unique_id','order_month']].drop_duplicates().copy()
    if a.empty:
        raise ValueError('No eligible delivered orders')
    first = a.groupby('customer_unique_id').order_month.min()
    a['cohort'] = a.customer_unique_id.map(first)
    a['age'] = a.order_month.astype('int64')-a.cohort.astype('int64')
    counts = a.groupby(['cohort','age']).customer_unique_id.nunique().unstack()
    sizes = first.value_counts().sort_index()
    sizes = sizes.loc[sizes.ge(min_cohort_size)]
    if sizes.empty:
        return pd.DataFrame(),sizes
    counts = counts.reindex(index=sizes.index,columns=range(cutoff.ordinal-sizes.index.min().ordinal+1))
    retention = counts.fillna(0).div(sizes,axis=0)
    for cohort in retention.index:
        for age in retention.columns:
            if cohort+age>cutoff:
                retention.loc[cohort,age]=np.nan
    return retention,sizes


def retention_scenario(retention,sizes,aov,target=.02,orders_per_returner=1):
    if not 0<=target<=1 or not np.isfinite(aov) or aov<0 or orders_per_returner<0:
        raise ValueError('Invalid scenario assumptions')
    if 1 not in retention:
        raise ValueError('No observed month 1')
    observed = retention[1].notna()
    eligible = sizes.loc[observed]
    if eligible.empty:
        raise ValueError('No mature cohort for month 1')
    base = float((retention.loc[observed,1]*eligible).sum()/eligible.sum())
    n = float(eligible.mean())
    return {'baseline_m1':base,'target_m1':target,'mean_eligible_new_customers_per_cohort':n,'assumed_repeat_orders_per_returner':orders_per_returner,'assumed_repeat_aov_brl':float(aov),'illustrative_incremental_gmv_per_cohort_brl':n*(target-base)*orders_per_returner*aov,'interpretation':'Hypothetical incremental merchandise GMV, not profit, guaranteed revenue or measured intervention effect.'}


def analyse(orders,customers,items,reviews,cutoff='2018-08'):
    df,audit=prepare_orders(orders,customers,items,reviews)
    valid=df.loc[df.is_valid_delivery & df.review_score.notna()].copy()
    valid['delivery_band']=pd.cut(valid.delivery_days,[-np.inf,7,14,19,30,np.inf],labels=['0–7','7–14','14–19','19–30','30+'],right=True)
    delivery=valid.groupby('delivery_band',observed=False).agg(orders=('order_id','count'),mean_score=('review_score','mean'),one_star_share=('review_score',lambda x:x.eq(1).mean())).reset_index()
    regional=valid.groupby('customer_state').agg(orders=('order_id','count'),mean_delivery_days=('delivery_days','mean'),mean_score=('review_score','mean')).reset_index()
    single=df.loc[df.unique_sellers.eq(1)].copy()
    mapping=items.groupby('order_id').seller_id.first()
    single['seller_id']=single.order_id.map(mapping)
    single['valid_delivery_days']=single.delivery_days.where(single.is_valid_delivery)
    sellers=single.groupby('seller_id').agg(orders=('order_id','count'),reviews=('review_score','count'),mean_score=('review_score','mean'),mean_valid_delivery_days=('valid_delivery_days','mean'),merchandise_gmv_brl=('total_price','sum')).reset_index()
    sellers=sellers.loc[sellers.reviews.ge(10)].sort_values(['mean_score','reviews'],ascending=[True,False])
    retention,sizes=cohort_retention(df,cutoff)
    eligible=df.loc[df.order_status.eq('delivered') & df.order_month.le(pd.Period(cutoff))]
    monthly=eligible.groupby('order_month').agg(orders=('order_id','count'),merchandise_gmv_brl=('total_price','sum')).reset_index()
    scenario=retention_scenario(retention,sizes,eligible.total_price.mean())
    summary={**audit,'delivered_orders':int(df.order_status.eq('delivered').sum()),'valid_delivery_reviews':len(valid),'retention_cutoff':cutoff,'cohort_population':'delivered orders; unique customer; observed purchase month; no proof of full history before extract', 'weighted_m1_retention_pct':scenario['baseline_m1']*100,'average_delivered_order_goods_brl':float(eligible.total_price.mean()),'scenario':scenario,'limits':['Delivery/ratings are observational associations, not causal effects.','Legacy 19-day threshold is exploratory; no confirmatory p-value is claimed.','August 2018 cutoff is an explicit analytical assumption, not proof of source completeness.','GMV excludes freight; BRL is not USD; no margin, cost or profit is available.']}
    tables={'delivery':delivery,'regions':regional,'sellers':sellers,'retention_pct':retention*100,'cohort_sizes':sizes.rename('customers').to_frame(),'monthly':monthly}
    return summary,tables
