"""Build the public Olist overview; raw Kaggle CSV files remain untracked."""
from pathlib import Path
import hashlib,json,platform,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.analysis import analyse
from scripts.portfolio_ui import write_overview


def main():
    names=['olist_orders_dataset.csv','olist_customers_dataset.csv','olist_order_items_dataset.csv','olist_order_reviews_dataset.csv']
    paths=[ROOT/'data'/n for n in names]
    missing=[p.name for p in paths if not p.is_file()]
    if missing:raise SystemExit('Download the four Kaggle CSV files into data/: '+', '.join(missing))
    summary,tables=analyse(*[pd.read_csv(p) for p in paths])
    summary['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    summary['verification_environment']={'python':platform.python_version(),'pandas':pd.__version__}
    report=ROOT/'reports';report.mkdir(exist_ok=True)
    (report/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    for n,frame in tables.items():frame.to_csv(report/(n+'.csv'),index=n in ['retention_pct','cohort_sizes'])
    delivery=tables['delivery'];regional=tables['regions'].query('orders >= 100').sort_values('mean_delivery_days',ascending=False).head(8)
    data={'title':'Olist — доставка, качество данных и повторные покупки','subtitle':'2016–2018 · открытый датасет · наблюдательный анализ, не коммерческий заказ',
    'kpis':[{'label':'Заказов в основе анализа','value':f"{summary['orders']:,}".replace(',',' ')},{'label':'Лишних строк отзывов исключено','value':str(summary['removed_review_rows'])},{'label':'Взвешенный retention M1','value':f"{summary['weighted_m1_retention_pct']:.3f}%"},{'label':'Заказов в сегменте 30+ дней','value':str(int(delivery.iloc[-1].orders))}],
    'charts':[{'title':'Оценка по сроку доставки','unit':'баллы из 5','rows':[{'label':str(r.delivery_band)+' дней','value':round(r.mean_score,3)} for r in delivery.itertuples()]},{'title':'Регионы с долгой доставкой','unit':'дни · не менее 100 отзывов','rows':[{'label':r.customer_state,'value':round(r.mean_delivery_days,2)} for r in regional.itertuples()]}],
    'insights':[{'fact':'При доставке до 7 дней оценка 4,42; более 30 дней — 2,25.','next':'Исследовать задержки в сегменте 30+ дней. Связь не доказывает, что сокращение срока само по себе изменит оценку.'},{'fact':'551 дополнительная строка отзывов могла размножить заказы при JOIN.','next':'Собирать витрину на уровне заказа; валидировать ключи и кардинальность, отдельно учитывать заказы без товаров и отзывов.'},{'fact':f"Взвешенный retention M1 — {summary['weighted_m1_retention_pct']:.3f}% для зрелых когорт.",'next':'Отделять наблюдаемый ноль от будущего периода. Сценарий роста до 2% — допущение о GMV, не результат внедрения.'}],
    'scope':'Деньги в BRL, не USD. Retention: delivered, уникальный покупатель, cutoff 2018-08, когорты от 10 покупателей. Текущий сценарий не учитывает затраты и маржу. Таблицы и SHA256 источников доступны в reports/.'}
    write_overview(ROOT,data)
    lines=['# Выводы и решения','']
    for i in data['insights']:lines.extend(['## '+i['fact'],'',i['next'],''])
    lines.extend(['## Сценарий повторных покупок','',f"При target M1=2%, одном повторном заказе на возвращённого покупателя и среднем товарном чеке {summary['average_delivered_order_goods_brl']:.2f} BRL: иллюстративный дополнительный GMV на среднюю зрелую когорту {summary['scenario']['illustrative_incremental_gmv_per_cohort_brl']:.2f} BRL.",'','Это сценарий, а не измеренная прибыль, предотвращённый ущерб или гарантированный эффект.','', '## Ограничения','',*['- '+x for x in summary['limits']]])
    (report/'EXECUTIVE_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['orders','removed_review_rows','weighted_m1_retention_pct']},ensure_ascii=False))


if __name__=='__main__':main()
