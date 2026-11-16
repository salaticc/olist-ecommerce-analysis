import unittest
import pandas as pd
from src.analysis import prepare_orders,cohort_retention,retention_scenario


def inputs():
    orders=pd.DataFrame([
        ['a','c1','delivered','2018-01-01','2018-01-02','2018-01-05'],
        ['b','c2','delivered','2018-02-01','2018-02-02','2018-02-06'],
        ['c','c3','canceled','2018-01-01',None,None]],columns=['order_id','customer_id','order_status','order_purchase_timestamp','order_delivered_carrier_date','order_delivered_customer_date'])
    customers=pd.DataFrame([['c1','u1','SP'],['c2','u1','SP'],['c3','u2','RJ']],columns=['customer_id','customer_unique_id','customer_state'])
    items=pd.DataFrame([['a',1,'s1',10,2],['a',2,'s2',20,3],['b',1,'s1',40,4]],columns=['order_id','order_item_id','seller_id','price','freight_value'])
    reviews=pd.DataFrame([['r1','a',1,'2018-01-06'],['r2','a',5,'2018-01-07'],['r3','b',4,'2018-02-07']],columns=['review_id','order_id','review_score','review_creation_date'])
    return orders,customers,items,reviews


class OrderGrainTests(unittest.TestCase):
    def test_items_and_reviews_do_not_multiply_orders(self):
        d,a=prepare_orders(*inputs());self.assertEqual(len(d),3)
        self.assertEqual(d.set_index('order_id').loc['a','total_price'],30)
        self.assertEqual(d.set_index('order_id').loc['a','review_score'],5)
        self.assertEqual(a['removed_review_rows'],1)

    def test_duplicate_customer_key_rejected(self):
        o,c,i,r=inputs();c=pd.concat([c,c.iloc[:1]])
        with self.assertRaises(ValueError):prepare_orders(o,c,i,r)

    def test_duplicate_order_key_rejected(self):
        o,c,i,r=inputs();o=pd.concat([o,o.iloc[:1]])
        with self.assertRaises(ValueError):prepare_orders(o,c,i,r)

    def test_duplicate_item_key_rejected(self):
        o,c,i,r=inputs();i=pd.concat([i,i.iloc[:1]])
        with self.assertRaises(ValueError):prepare_orders(o,c,i,r)

    def test_delivery_before_purchase_is_invalid(self):
        o,c,i,r=inputs();o.loc[0,'order_delivered_customer_date']='2017-12-31'
        d,_=prepare_orders(o,c,i,r);self.assertFalse(d.iloc[0].is_valid_delivery)

    def test_cancelled_customer_not_a_completed_purchase(self):
        d,_=prepare_orders(*inputs());r,s=cohort_retention(d,'2018-02',1)
        self.assertEqual(s.sum(),1);self.assertEqual(r.loc[pd.Period('2018-01'),1],1)

    def test_cutoff_applies_before_cohort_sizes(self):
        d,_=prepare_orders(*inputs());r,s=cohort_retention(d,'2018-01',1)
        self.assertEqual(s.sum(),1);self.assertEqual(r.shape[1],1)

    def test_future_cells_not_filled_with_zero(self):
        o,c,i,r=inputs();c.loc[1,'customer_unique_id']='u3'
        d,_=prepare_orders(o,c,i,r);ret,size=cohort_retention(d,'2018-02',1)
        self.assertTrue(pd.isna(ret.loc[pd.Period('2018-02'),1]))
        self.assertEqual(ret.loc[pd.Period('2018-01'),1],0)

    def test_scenario_weights_customers_not_cohorts_or_orders(self):
        r=pd.DataFrame({1:[.1,.5]},index=['a','b']);s=pd.Series([100,10],index=r.index)
        value=retention_scenario(r,s,20,target=.2)
        self.assertAlmostEqual(value['baseline_m1'],15/110)
        self.assertAlmostEqual(value['illustrative_incremental_gmv_per_cohort_brl'],70)


if __name__=='__main__':unittest.main()
