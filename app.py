import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import warnings
warnings.filterwarnings('ignore')

from sklearn.dummy            import DummyClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import (StratifiedKFold, GridSearchCV,
                                       cross_val_predict, cross_validate)
from sklearn.metrics          import (f1_score, roc_auc_score, recall_score,
                                       roc_curve, confusion_matrix,
                                       classification_report)

st.set_page_config(
    page_title='UK Household Financial Vulnerability Risk Scorecard',
    page_icon='📊', layout='wide', initial_sidebar_state='expanded'
)

PAGE_BG = '#F4F7FA'
SIDEBAR_BG = "#EBFB11"
CARD_BG = '#FFFFFF'
TEXT = '#0F172A'
PRIMARY = '#1F5FA6'
SECONDARY = '#2B6A2F'
ACCENT = '#C0521A'
ACCENT2 = '#E8A838'

@st.cache_data(show_spinner=False)
def load_data():
    FALLBACK = {
        'year': [2018, 2019, 2020, 2021, 2022, 2023],
        'food_inflation_pct': [2.1, 1.8, 0.5, 1.1, 11.0, 19.1],
        'energy_inflation_pct': [5.5, 2.9, -5.3, 7.9, 59.4, 14.5],
    }
    src = 'Embedded ONS MM23'
    try:
        fr = requests.get('https://api.ons.gov.uk/v1/timeseries/D7G8/dataset/mm23/data', timeout=8)
        er = requests.get('https://api.ons.gov.uk/v1/timeseries/L55O/dataset/mm23/data', timeout=8)
        if fr.status_code == 200 and er.status_code == 200:
            def parse(r, col):
                rows = r.json().get('years', [])
                df = pd.DataFrame(rows)[['year', 'value']].copy()
                df.columns = ['year', col]
                df['year'] = df['year'].astype(int)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                return df[df['year'].between(2018, 2023)].dropna().reset_index(drop=True)

            cpi = parse(fr, 'food_inflation_pct').merge(parse(er, 'energy_inflation_pct'), on='year')
            src = 'ONS Live API'
        else:
            cpi = pd.DataFrame(FALLBACK)
    except Exception:
        cpi = pd.DataFrame(FALLBACK)

    YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
    DEC = list(range(1, 11))
    net = [150.2, 245.8, 320.5, 392.1, 468.3, 548.7, 641.2, 762.4, 938.5, 1542.1,
           158.4, 256.2, 334.1, 408.6, 487.9, 571.4, 668.3, 793.7, 976.2, 1601.8,
           162.7, 263.4, 341.8, 418.2, 498.6, 584.1, 683.5, 812.3, 998.4, 1638.2,
           168.3, 271.6, 352.4, 430.8, 513.2, 601.4, 703.2, 836.5, 1028.6, 1689.4,
           178.5, 287.3, 372.6, 455.4, 542.7, 635.8, 742.1, 883.2, 1086.4, 1784.6,
           188.2, 302.8, 392.5, 480.1, 572.4, 670.3, 782.6, 931.4, 1145.2, 1881.3]
    food = [28.4, 33.2, 37.8, 42.1, 46.8, 51.3, 56.7, 63.2, 72.4, 92.6,
            29.1, 34.1, 38.6, 43.2, 48.1, 52.8, 58.3, 64.9, 74.2, 95.1,
            31.2, 36.4, 41.2, 45.8, 51.3, 56.2, 62.1, 69.4, 79.3, 101.2,
            32.8, 38.1, 43.2, 48.2, 53.9, 59.1, 65.3, 72.8, 83.4, 106.3,
            36.4, 42.8, 48.7, 54.3, 60.8, 66.7, 73.9, 82.4, 94.6, 120.3,
            42.1, 49.8, 56.8, 63.4, 71.2, 78.3, 86.7, 96.8, 111.2, 141.4]
    energy = [14.2, 15.8, 17.1, 18.3, 19.4, 20.6, 21.8, 23.2, 25.4, 30.1,
              13.8, 15.3, 16.6, 17.8, 18.9, 20.1, 21.2, 22.6, 24.7, 29.3,
              14.1, 15.6, 16.9, 18.1, 19.3, 20.5, 21.7, 23.1, 25.2, 29.9,
              15.2, 16.9, 18.3, 19.6, 20.8, 22.1, 23.4, 24.9, 27.2, 32.3,
              24.6, 27.3, 29.5, 31.6, 33.6, 35.7, 37.8, 40.3, 44.1, 52.3,
              28.3, 31.4, 33.9, 36.3, 38.6, 41.0, 43.4, 46.4, 50.8, 60.3]
    p = pd.DataFrame({'year': [y for y in YEARS for _ in DEC],
                      'income_decile': DEC * len(YEARS),
                      'net_income_weekly': net,
                      'food_spend_weekly': food,
                      'energy_spend_weekly': energy})
    p = p.merge(cpi, on='year', how='left')
    p['food_energy_spend_weekly'] = p['food_spend_weekly'] + p['energy_spend_weekly']
    p['food_energy_pct_income'] = (p['food_energy_spend_weekly'] / p['net_income_weekly']) * 100
    p['food_pct_income'] = (p['food_spend_weekly'] / p['net_income_weekly']) * 100
    p['energy_pct_income'] = (p['energy_spend_weekly'] / p['net_income_weekly']) * 100
    p['vulnerable_25'] = (p['food_energy_pct_income'] >= 25).astype(int)
    p['vulnerable_30'] = (p['food_energy_pct_income'] >= 30).astype(int)
    p['vulnerable_35'] = (p['food_energy_pct_income'] >= 35).astype(int)
    return p, src

@st.cache_resource(show_spinner=False)
def train_models(panel):
    FEAT = ['income_decile', 'food_inflation_pct', 'energy_inflation_pct', 'food_energy_pct_income']
    X = panel[FEAT].values
    y = panel['vulnerable_30'].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bl = DummyClassifier(strategy='stratified', random_state=42)
    lrp = Pipeline([('sc', StandardScaler()),
                    ('lr', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000,
                                              random_state=42, class_weight='balanced'))])
    lrg = GridSearchCV(lrp, {'lr__C': [0.01, 0.1, 1, 10]}, cv=cv, scoring='recall', n_jobs=-1)
    lrg.fit(X, y)
    lr = lrg.best_estimator_
    dtp = Pipeline([('sc', StandardScaler()),
                    ('dt', DecisionTreeClassifier(criterion='gini', random_state=42,
                                                  class_weight='balanced'))])
    dtg = GridSearchCV(dtp, {'dt__max_depth': [3, 5, 7, 10]}, cv=cv, scoring='recall', n_jobs=-1)
    dtg.fit(X, y)
    dt = dtg.best_estimator_
    lr.fit(X, y)
    dt.fit(X, y)

    def cvm(est):
        prob = cross_val_predict(est, X, y, cv=cv, method='predict_proba')[:, 1]
        pred = (prob >= 0.5).astype(int)
        return {
            'f1': f1_score(y, pred, average='macro'),
            'auc': roc_auc_score(y, prob),
            'recall': recall_score(y, pred),
        }

    bl_prob = cross_val_predict(bl, X, y, cv=cv, method='predict_proba')[:, 1]
    bl_pred = cross_val_predict(bl, X, y, cv=cv)
    metrics = {
        'Baseline': {
            'f1': f1_score(y, bl_pred, average='macro'),
            'auc': roc_auc_score(y, bl_prob),
            'recall': recall_score(y, bl_pred),
        },
        'Logistic Regression': cvm(lr),
        'Decision Tree': cvm(dt),
    }
    coef = lr.named_steps['lr'].coef_[0]
    fi = pd.DataFrame({'Feature': FEAT, 'Coefficient': coef, 'Abs': np.abs(coef)}).sort_values('Abs', ascending=False)
    dfi = pd.DataFrame({'Feature': FEAT, 'Importance': dt.named_steps['dt'].feature_importances_}).sort_values('Importance', ascending=False)
    return lr, dt, metrics, fi, dfi, X, y, lrg.best_params_, dtg.best_params_


def style_plotly(fig, height=None):
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', color=TEXT),
        title=dict(font=dict(size=18, color=TEXT), x=0.01),
        legend=dict(font=dict(size=11), orientation='h', yanchor='bottom', y=-0.18, x=0),
        margin=dict(l=40, r=24, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(15,23,42,0.08)', zerolinecolor='rgba(15,23,42,0.12)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(15,23,42,0.08)', zerolinecolor='rgba(15,23,42,0.12)')
    return fig

page_styles = f'''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif !important;
    color: {TEXT} !important;
    background-color: {PAGE_BG} !important;
}}
[data-testid='stAppViewContainer'] {{
    background-color: {PAGE_BG} !important;
}}
[data-testid='stSidebar'] {{
    background-color: {SIDEBAR_BG} !important;
}}
section.main .block-container {{
    padding: 1.4rem 2rem 2rem 2rem;
    background-color: {PAGE_BG} !important;
}}
[data-testid='stToolbar'] {{
    background-color: {PAGE_BG} !important;
}}
.stButton>button, div.stButton>button {{
    background-color: {PRIMARY} !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.72rem 1rem !important;
}}
.stButton>button:hover, div.stButton>button:hover {{
    background-color: #164c8e !important;
}}
.css-1d391kg, .css-1v3fvcr, .css-1kyxreq, .css-14xtw13, .css-15zrgzn {{
    background-color: rgba(255, 255, 255, 0.94) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08) !important;
}}
[data-testid='metric-container'] {{
    background: white !important;
}}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
    color: {TEXT} !important;
}}
</style>
'''

st.markdown(page_styles, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### 7005SCN Individual Research Project')
    st.markdown('**Predictive Risk Scorecard**')
    st.markdown('---')
    page = st.radio('Navigate', ['Overview & Data', 'Exploratory Analysis', 'Model Performance',
                                'Risk Scorecard', 'Sensitivity Analysis', 'What-If Simulator'])
    st.markdown('---')
    threshold = st.slider('Vulnerability threshold (%)', 20, 45, 30, 1)
    year_sel = st.multiselect('Year filter', [2018, 2019, 2020, 2021, 2022, 2023], default=[2022, 2023])
    st.markdown('---')
    st.caption('Data: ONS LCF Survey + BoE CPI')
    st.caption('Models: LR + DT | Stratified 5-Fold CV')

with st.spinner('Loading data & training models...'):
    panel, cpi_src = load_data()
    panel['vulnerable_custom'] = (panel['food_energy_pct_income'] >= threshold).astype(int)
    best_lr, best_dt, metrics, fi_lr, fi_dt, X, y, lr_p, dt_p = train_models(panel)
    FEAT = ['income_decile', 'food_inflation_pct', 'energy_inflation_pct', 'food_energy_pct_income']

if page == 'Overview & Data':
    st.title('UK Household Financial Vulnerability — Predictive Risk Scorecard')
    st.info(f'Data: {cpi_src} | Threshold: {threshold}% | Models: LR + DT | CV: Stratified 5-Fold')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Observations', len(panel))
    c2.metric('Years', '2018-2023')
    nv = (panel['food_energy_pct_income'] >= threshold).sum()
    c3.metric(f'Vulnerable ({threshold}%)', f'{nv}/{len(panel)}', f'{nv/len(panel)*100:.1f}%')
    c4.metric('Income Deciles', '10')
    st.subheader('Panel Dataset')
    cols = ['year', 'income_decile', 'net_income_weekly', 'food_spend_weekly',
            'energy_spend_weekly', 'food_energy_pct_income', 'vulnerable_30']
    filt = panel[panel.year.isin(year_sel)][cols].round(2)
    st.dataframe(filt, use_container_width=True)
    cpi_plot = panel[['year', 'food_inflation_pct', 'energy_inflation_pct']].drop_duplicates()
    fig = px.line(cpi_plot.melt('year'), x='year', y='value', color='variable', markers=True,
                  title='ONS CPI Annual Rates: Food & Energy (2018-2023)',
                  color_discrete_map={'food_inflation_pct': ACCENT, 'energy_inflation_pct': PRIMARY})
    fig.add_hline(y=0, line_dash='dash', line_color='grey', opacity=0.5)
    style_plotly(fig, height=440)
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Exploratory Analysis':
    st.title('Exploratory Data Analysis')
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(panel, x='income_decile', y='food_energy_pct_income', color='year', markers=True,
                      title='Food + Energy % Income by Decile and Year',
                      color_discrete_sequence=px.colors.diverging.RdYlGn)
        fig.add_hline(y=threshold, line_dash='dash', line_color='red', annotation_text=f'{threshold}% threshold')
        style_plotly(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        heat = panel.pivot_table('vulnerable_30', 'income_decile', 'year', aggfunc='first')
        fig = px.imshow(heat, color_continuous_scale=px.colors.diverging.RdYlGn,
                        title='Vulnerability Heatmap (30% threshold)', text_auto=True, aspect='auto')
        style_plotly(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)
    corr = panel[FEAT + ['vulnerable_30']].corr()
    fig = px.imshow(corr, color_continuous_scale='RdBu', text_auto='.2f',
                    title='Feature Correlation Matrix', aspect='auto', zmin=-1, zmax=1)
    style_plotly(fig, height=430)
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Model Performance':
    st.title('Model Training & Evaluation')
    st.info(f'LR Best C: {lr_p["lr__C"]} | DT Best depth: {dt_p["dt__max_depth"]} | CV: Stratified 5-Fold')
    rows = [{'Model': n, 'F1-macro': round(m['f1'], 4), 'ROC-AUC': round(m['auc'], 4),
             'Recall (Vulnerable)': round(m['recall'], 4)} for n, m in metrics.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    fig = go.Figure()
    for n, c in [('Baseline', '#AAAAAA'), ('Logistic Regression', PRIMARY), ('Decision Tree', ACCENT)]:
        m = metrics[n]
        fig.add_trace(go.Bar(name=n, x=['F1-macro', 'ROC-AUC', 'Recall (Vulnerable)'],
                             y=[m['f1'], m['auc'], m['recall']], marker_color=c, opacity=0.85))
    fig.update_layout(barmode='group', title='Model Comparison - CV Metrics',
                      yaxis_range=[0, 1.15], yaxis_title='Score')
    style_plotly(fig, height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('ROC Curves')
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fig = go.Figure()
    for est, lbl, col in [(best_lr, 'Logistic Regression', PRIMARY), (best_dt, 'Decision Tree', ACCENT)]:
        prob = cross_val_predict(est, X, y, cv=cv5, method='predict_proba')[:, 1]
        fpr, tpr, _ = roc_curve(y, prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{lbl} AUC={roc_auc_score(y, prob):.3f}',
                                 line=dict(color=col, width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance',
                             line=dict(color='grey', dash='dash')))
    fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                      title='ROC Curves - 5-Fold CV')
    style_plotly(fig, height=470)
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Risk Scorecard':
    st.title('Predictive Risk Scorecard')
    yr = st.selectbox('Select year', [2023, 2022, 2021, 2020, 2019, 2018])
    sub = panel[panel.year == yr].copy()
    Xs = sub[FEAT].values
    probs = best_lr.predict_proba(Xs)[:, 1]
    pdt = best_dt.predict(Xs)
    sc = pd.DataFrame({
        'Decile': sub['income_decile'].values,
        'Weekly Inc': sub['net_income_weekly'].round(0).values,
        'FE pct Inc': sub['food_energy_pct_income'].round(1).values,
        'LR Prob': np.round(probs, 3),
        'LR Pred': ['Vulnerable' if p >= 0.5 else 'Secure' for p in probs],
        'DT Pred': ['Vulnerable' if p == 1 else 'Secure' for p in pdt],
        'Consensus': ['Vulnerable' if (p >= 0.5) + q >= 1 else 'Secure' for p, q in zip(probs, pdt)],
        'Risk Level': ['HIGH RISK' if p >= 0.70 else 'MODERATE' if p >= 0.45 else 'LOW RISK' for p in probs],
        'True Label': ['Vulnerable' if v == 1 else 'Secure' for v in sub['vulnerable_30'].values],
    })
    def cr(v):
        if v == 'HIGH RISK':
            return 'background-color:#FDE7E7;font-weight:bold'
        if v == 'MODERATE':
            return 'background-color:#FEF4D6;font-weight:bold'
        return 'background-color:#ECF7EF'
    st.dataframe(sc.style.applymap(cr, subset=['Risk Level']), use_container_width=True, height=440)
    fig = go.Figure(go.Bar(x=probs, y=[f'D{d}' for d in sub['income_decile']], orientation='h',
                          marker_color=['#C0521A' if p >= 0.7 else '#E8A838' if p >= 0.45 else '#2B6A2F' for p in probs],
                          opacity=0.88, text=[f'{p:.2f}' for p in probs], textposition='outside'))
    fig.add_vline(x=0.5, line_dash='dash', line_color='black', annotation_text='Decision boundary')
    fig.update_layout(title=f'Vulnerability Probability by Decile ({yr})',
                      xaxis_range=[0, 1.15], xaxis_title='LR Vulnerability Probability')
    style_plotly(fig, height=430)
    st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=fi_lr['Coefficient'], y=fi_lr['Feature'], orientation='h',
                              marker_color=[ACCENT if c > 0 else PRIMARY for c in fi_lr['Coefficient']], opacity=0.85))
        fig.add_vline(x=0, line_color='black', line_width=0.8)
        fig.update_layout(title='LR Standardised Coefficients')
        style_plotly(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(fi_dt, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues', title='DT Gini Importances')
        style_plotly(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)
    st.download_button('Download Scorecard CSV', sc.to_csv(index=False).encode(),
                       f'scorecard_{yr}.csv', 'text/csv')

elif page == 'Sensitivity Analysis':
    st.title('Sensitivity Analysis')
    col1, col2 = st.columns(2)
    with col1:
        rows = []
        for t, col in [(25, 'vulnerable_25'), (30, 'vulnerable_30'), (35, 'vulnerable_35')]:
            for yr2, grp in panel.groupby('year'):
                rows.append({'Threshold': f'{t}pct', 'Year': yr2, 'Pct Vulnerable': grp[col].mean() * 100})
        fig = px.line(pd.DataFrame(rows), x='Year', y='Pct Vulnerable', color='Threshold', markers=True,
                      title='Pct Vulnerable by Year and Threshold',
                      color_discrete_sequence=['#E67E22', ACCENT, '#922B21'])
        style_plotly(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        yr_s = st.selectbox('Base year for shift test', [2022, 2023, 2021])
        rows2 = []
        pt = panel[panel.year == yr_s].copy()
        for shift in [-10, -5, -2, 0, 2, 5, 10]:
            tmp = pt.copy()
            tmp['food_inflation_pct'] += shift
            tmp['energy_inflation_pct'] += shift
            pct = best_lr.predict(tmp[FEAT].values).mean() * 100
            rows2.append({'Shift (pp)': shift, 'pct Vulnerable': pct})
        fig = px.bar(pd.DataFrame(rows2), x='Shift (pp)', y='pct Vulnerable',
                     color='Shift (pp)', title=f'Inflation Shift Sensitivity ({yr_s})',
                     color_continuous_scale=px.colors.diverging.RdYlGn, text='pct Vulnerable')
        fig.update_traces(texttemplate='%{text:.0f}pct', textposition='outside')
        fig.add_vline(x=0, line_dash='dash', line_color='black')
        style_plotly(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)
    panel['boundary'] = (panel.vulnerable_25 != panel.vulnerable_35).astype(int)
    bdf = panel[panel.boundary == 1][['year', 'income_decile', 'food_energy_pct_income',
                                     'vulnerable_25', 'vulnerable_30', 'vulnerable_35']]
    st.subheader('Boundary-Sensitive Observations')
    if len(bdf):
        st.dataframe(bdf.round(2), use_container_width=True)
    else:
        st.success('No boundary-sensitive observations.')

elif page == 'What-If Simulator':
    st.title('What-If Inflation Simulator')
    st.markdown('Adjust sliders to simulate a custom inflationary scenario.')
    c1, c2, c3 = st.columns(3)
    fs = c1.slider('Food CPI shock (%)', -5.0, 30.0, 11.0, 0.5)
    es = c2.slider('Energy CPI shock (%)', -10.0, 80.0, 59.4, 0.5)
    sy = c3.selectbox('Base year', [2023, 2022, 2021, 2020])
    base = panel[panel.year == sy].copy()
    sim = base.copy()
    sim['food_inflation_pct'] = fs
    sim['energy_inflation_pct'] = es
    probs = best_lr.predict_proba(sim[FEAT].values)[:, 1]
    bprobs = best_lr.predict_proba(base[FEAT].values)[:, 1]
    nv = probs.sum()
    c1.metric('Food CPI', f'{fs}%', f'{fs - 19.1:+.1f}pp vs 2023')
    c2.metric('Energy CPI', f'{es}%', f'{es - 14.5:+.1f}pp vs 2023')
    c3.metric('Deciles Vulnerable', f'{int(nv)}/10', f'{int(nv * 10)}% of households')
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Simulated', x=[f'D{d}' for d in sim['income_decile']], y=probs,
                          marker_color=['#C0521A' if p >= 0.5 else '#2B6A2F' for p in probs],
                          opacity=0.88, text=[f'{p:.2f}' for p in probs], textposition='outside'))
    fig.add_trace(go.Scatter(name='Actual baseline', x=[f'D{d}' for d in base['income_decile']],
                              y=bprobs, mode='markers+lines',
                              marker=dict(color='grey', size=8), line=dict(dash='dot', color='grey')))
    fig.add_hline(y=0.5, line_dash='dash', line_color='black', annotation_text='50% boundary')
    fig.update_layout(title=f'Simulated vs Actual Vulnerability Probability (base {sy})',
                      yaxis_range=[0, 1.2], yaxis_title='Vulnerability Probability')
    style_plotly(fig, height=470)
    st.plotly_chart(fig, use_container_width=True)
    sim_tbl = pd.DataFrame({'Decile': sim['income_decile'].values,
                             'Weekly Income': base['net_income_weekly'].round(0).values,
                             'Sim Prob': np.round(probs, 3),
                             'Actual Prob': np.round(bprobs, 3),
                             'Change': np.round(probs - bprobs, 3),
                             'Prediction': ['Vulnerable' if p >= 0.5 else 'Secure' for p in probs]})
    def cc(v):
        if v > 0.1:
            return 'color:#C0521A;font-weight:bold'
        if v < -0.1:
            return 'color:#2B6A2F;font-weight:bold'
        return ''
    st.dataframe(sim_tbl.style.applymap(cc, subset=['Change']), use_container_width=True)
