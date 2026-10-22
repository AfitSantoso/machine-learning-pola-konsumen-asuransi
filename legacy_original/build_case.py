import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,confusion_matrix,roc_curve
from sklearn.inspection import permutation_importance
from pathlib import Path
p=Path('/mnt/data/insurance_ml_workshop');rng=np.random.default_rng(26)
months=pd.date_range('2025-01-01',periods=22,freq='MS'); n=7000
customer=pd.DataFrame({'customer_id':np.arange(1,n+1),'age':rng.integers(22,65,n),'region':rng.choice(['Jabodetabek','Jawa lain','Sumatera','Kalimantan','Indonesia timur'],n,p=[.36,.28,.18,.1,.08]),'product':rng.choice(['Kesehatan','Jiwa','Kendaraan'],n,p=[.46,.34,.20]),'income_band':rng.choice(['Low','Mid','High'],n,p=[.34,.46,.20]),'tenure_init':rng.integers(1,48,n),'premium_base':rng.integers(180,850,n)*1000,'claim_propensity':rng.beta(1.5,7,n),'sensitivity':rng.normal(0,1,n),'risk':rng.normal(0,1,n)})
rows=[]; active=np.ones(n,dtype=bool);prev_claim=np.zeros(n);complaints=np.zeros(n);late=np.zeros(n)
for i,month in enumerate(months):
    t=month.strftime('%Y-%m'); hike=int(month>=pd.Timestamp('2026-01-01'))
    macro_stress=.15 if month<pd.Timestamp('2026-03-01') else (.75 if month<pd.Timestamp('2026-07-01') else .45)
    renewal=((customer.tenure_init.to_numpy()+i)%12<=1).astype(int)
    claim=(rng.random(n)<customer.claim_propensity.to_numpy()*(1.1 if macro_stress>.5 else 1)).astype(int)
    complaints=(rng.random(n)<(.025+.055*macro_stress+.045*claim)).astype(int)
    late=(rng.random(n)<(.06+.065*macro_stress+ .05*(customer.income_band.to_numpy()=='Low'))).astype(int)
    touch=(rng.random(n)<(.26-.07*macro_stress)).astype(int)
    price=(customer.premium_base.to_numpy()*(1+.1*hike)).astype(int)
    logit=-4.25 + .43*(customer.income_band.to_numpy()=='Low')+.58*late+.76*complaints+.47*renewal+.34*(touch==0)+.32*macro_stress+.29*customer.risk.to_numpy()+.10*hike+.46*((month>=pd.Timestamp('2026-04-01')) & (month<=pd.Timestamp('2026-06-01')))*renewal
    probs=1/(1+np.exp(-logit));leave=(rng.random(n)<probs)&active
    for j in np.where(active)[0]:
        rows.append([int(j+1),t,int(customer.age.iloc[j]),str(customer.region.iloc[j]),str(customer["product"].iloc[j]),str(customer.income_band.iloc[j]),int(customer.tenure_init.iloc[j]+i),int(price[j]),hike,float(macro_stress),int(renewal[j]),int(claim[j]),int(complaints[j]),int(late[j]),int(touch[j]),int(leave[j])])
    active=active&(~leave)
df=pd.DataFrame(rows,columns=['customer_id','month','age','region','product','income_band','tenure_months','premium_idr','price_increase_10pct','economic_stress_index','renewal_window','claim_last_month','complaint_last_month','late_payment_last_month','contact_last_month','churn_next_30d'])
df.to_csv(p/'insurance_policy_monthly_synthetic.csv',index=False)
m=df.groupby('month').agg(active_policies=('customer_id','size'),churned=('churn_next_30d','sum')).reset_index();m['churn_rate']=m.churned/m.active_policies
# Acquisition deliberately includes unobserved latent events, shown ONLY as teaching hypothesis, not as observed features.
a=[]
for i,month in enumerate(months):
    campaign=1 if i in [7,13,18] else 0
    latent=1 if month>=pd.Timestamp('2026-08-01') else 0
    leads=int(max(40,rng.normal(170+20*campaign+205*latent,22)))
    converted=int(rng.binomial(leads,.18+(.06 if latent else 0)))
    a.append([month.strftime('%Y-%m'),campaign,leads,converted])
acq=pd.DataFrame(a,columns=['month','campaign_active','inbound_leads','new_policies']);acq.to_csv(p/'insurance_acquisition_monthly_synthetic.csv',index=False)
# strict calendar split, rows of different time for same customers are NOT independent but a realistic operational forecast
train=df[df.month<'2026-02'].copy();test=df[df.month>='2026-02'].copy()
# Model baseline and nonlinear model train; prune leakage/proxies (not using customer_id/month/target)
num=['age','tenure_months','premium_idr','price_increase_10pct','economic_stress_index','renewal_window','claim_last_month','complaint_last_month','late_payment_last_month','contact_last_month'];cat=['region','product','income_band'];features=num+cat
Xtr=train[features];Xte=test[features];ytr=train.churn_next_30d;yte=test.churn_next_30d
pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
base=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=450,class_weight='balanced'))]);base.fit(Xtr,ytr)
model=Pipeline([('pre',pre),('model',RandomForestClassifier(n_estimators=90,max_depth=11,min_samples_leaf=25,n_jobs=-1,random_state=42,class_weight='balanced_subsample'))]);model.fit(Xtr,ytr)
pred=model.predict_proba(Xte)[:,1];pred_base=base.predict_proba(Xte)[:,1]
res={'positives':int(yte.sum()),'test_rows':len(yte),'prevalence':round(yte.mean(),4),'roc_auc':round(roc_auc_score(yte,pred),4),'pr_auc':round(average_precision_score(yte,pred),4),'baseline_roc_auc':round(roc_auc_score(yte,pred_base),4),'baseline_pr_auc':round(average_precision_score(yte,pred_base),4)}
order=np.argsort(-pred);k=int(len(order)*.1);res['precision_at_10pct']=round(float(yte.iloc[order[:k]].mean()),4);res['recall_at_10pct']=round(float(yte.iloc[order[:k]].sum()/yte.sum()),4);res['lift_at_10pct']=round(res['precision_at_10pct']/res['prevalence'],2)
pd.Series(res).to_csv(p/'model_metrics.csv',header=False)
# 4 charts
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,4.4));ax.plot(pd.to_datetime(m.month),m.churn_rate*100,marker='o',linewidth=2.5);ax.axvline(pd.Timestamp('2026-01-01'),ls='--',c='#f59e0b',label='Premium +10%');ax.axvspan(pd.Timestamp('2026-04-01'),pd.Timestamp('2026-06-30'),color='#dc2626',alpha=.1,label='Elevated churn');ax.set_ylabel('Monthly churn rate (%)');ax.set_title('Premium increase and later churn surge | synthetic');ax.legend(loc='upper left');fig.tight_layout();fig.savefig(p/'churn_trend.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.4));ax.plot(pd.to_datetime(acq.month),acq.new_policies,marker='o',linewidth=2.5,label='New policies');ax.scatter(pd.to_datetime(acq.loc[acq.campaign_active==1,'month']),acq.loc[acq.campaign_active==1,'new_policies'],s=125,facecolors='none',edgecolors='#d97706',linewidths=2,label='Campaign month');ax.axvspan(pd.Timestamp('2026-08-01'),pd.Timestamp('2026-10-31'),color='#16a34a',alpha=.08,label='Unexplained lift / hypothesis');ax.set_title('Acquisition spikes are not aligned to campaign | synthetic');ax.set_ylabel('New policies');ax.legend(loc='upper left');fig.tight_layout();fig.savefig(p/'acquisition_trend.png',dpi=170);plt.close(fig)
precision,recall,th=precision_recall_curve(yte,pred);fpr,tpr,_=roc_curve(yte,pred)
fig,ax=plt.subplots(figsize=(7,4.6));ax.plot(recall,precision,linewidth=2.6,label='Random forest');ax.axhline(yte.mean(),linestyle='--',label='Random selection');ax.set(xlabel='Recall',ylabel='Precision',title='Precision–Recall curve | out-of-time test');ax.legend();fig.tight_layout();fig.savefig(p/'pr_curve.png',dpi=170);plt.close(fig)
from sklearn.metrics import precision_score
q=pd.DataFrame({'score':pred,'churn':yte.to_numpy()}).sort_values('score',ascending=False);q['decile']=np.minimum(10,(np.arange(len(q))*10/len(q)).astype(int)+1);lift=q.groupby('decile')['churn'].mean()/yte.mean()
fig,ax=plt.subplots(figsize=(8,4.4));ax.bar(lift.index,lift.values);ax.axhline(1,ls='--',c='gray');ax.set(title='Risk lift by score decile | 1 = highest risk',xlabel='Risk decile',ylabel='Lift vs population');ax.set_xticks(range(1,11));fig.tight_layout();fig.savefig(p/'decile_lift.png',dpi=170);plt.close(fig)
# Built-in impurity importance grouped by original feature
names=list(num)+list(model.named_steps['pre'].named_transformers_['cat'].get_feature_names_out(cat));imp=model.named_steps['model'].feature_importances_;feat=pd.DataFrame({'feature':names,'importance':imp}).sort_values('importance',ascending=False);feat.to_csv(p/'model_feature_importance.csv',index=False)
fig,ax=plt.subplots(figsize=(8.5,5));top=feat.head(9).iloc[::-1];ax.barh(top.feature,top.importance);ax.set(title='Model feature importance (descriptive, not causal)',xlabel='Random forest impurity importance');fig.tight_layout();fig.savefig(p/'feature_importance.png',dpi=170);plt.close(fig)
print(res)
