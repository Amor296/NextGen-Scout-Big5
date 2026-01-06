import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="City Group - Elite Sniper Scout", layout="wide", page_icon="⚽")
st.title("💎 NextGen Global Scout: Elite Sniper Edition")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv('data/final_cleaned_stats.csv')
    df.columns = df.columns.str.strip()
    
    # تنظيف السن والدقائق (90s)
    df['Age'] = df['Age'].apply(lambda x: int(str(x).split('-')[0]) if '-' in str(x) else 0)
    if '90s' in df.columns:
        df['90s'] = pd.to_numeric(df['90s'], errors='coerce').fillna(0)
    elif 'Min' in df.columns:
        df['90s'] = pd.to_numeric(df['Min'], errors='coerce').fillna(0) / 90
    else:
        df['90s'] = 1.0

    # تنظيف عدد الأهداف
    df['Gls'] = pd.to_numeric(df['Gls'], errors='coerce').fillna(0)
    
    # حساب الكفاءة (Gls/90)
    df['Efficiency'] = df.apply(lambda row: row['Gls'] / row['90s'] if row['90s'] > 0 else 0, axis=1)

    # مؤشر الذكاء التكتيكي (Tactical IQ %)
    top_clubs = ['Manchester City', 'Real Madrid', 'Arsenal', 'Liverpool', 'Bayern Munich', 'Paris S-G', 'Barcelona']
    def estimate_passing(row):
        base = 72.0
        if 'Pos' in row and any(p in str(row['Pos']) for p in ['MF', 'DF']): base += 10.0
        if 'Squad' in row and row['Squad'] in top_clubs: base += 12.0
        return min(base, 98.0)
    df['Tactical_IQ'] = df.apply(estimate_passing, axis=1)

    # خوارزمية الأسعار (Current vs Projected)
    def get_market_logic(row):
        base_val = (row['Gls'] * 2.0) + (row['Efficiency'] * 15)
        age_multiplier = 3.5 if row['Age'] <= 22 else (2.0 if row['Age'] <= 27 else 0.6)
        squad_multiplier = 2.5 if row['Squad'] in top_clubs else 1.0
        current = base_val * age_multiplier * squad_multiplier
        current = max(min(current, 180.0), 1.5)
        projected = current * (1.4 if row['Age'] <= 22 and row['Efficiency'] > 0.4 else 1.1)
        return round(current, 1), round(projected, 1)

    df[['Current_Val', 'Projected_Val']] = df.apply(lambda x: pd.Series(get_market_logic(x)), axis=1)
    
    scaler = MinMaxScaler()
    df[['Age_norm', 'Gls_norm', 'Eff_norm']] = scaler.fit_transform(df[['Age', 'Gls', 'Efficiency']])
    
    return df

try:
    df = load_data()

    # --- 3. SIDEBAR TACTICAL FILTERS ---
    st.sidebar.header("🎯 Recruitment Filters")
    if st.sidebar.button("🔄 Reset System"): st.rerun()
    
    search = st.sidebar.text_input("🔍 Search Player Name", "").lower()
    
    # أ. فلتر الحد الأدنى للأهداف (New!)
    min_goals = st.sidebar.number_input("🥅 Min Goals Scored", min_value=0, max_value=int(df['Gls'].max()), value=0)
    
    # ب. فلتر الجنسية والمركز
    all_nations = sorted(df['Nation'].dropna().unique()) if 'Nation' in df.columns else []
    selected_nation = st.sidebar.multiselect("🌎 Select Nationality", all_nations)
    
    all_positions = sorted(df['Pos'].unique())
    selected_pos = st.sidebar.multiselect("🛡️ Position Role", all_positions)
    
    # ج. فلتر السن والماليات
    age_range = st.sidebar.slider("🎂 Age Bracket", int(df['Age'].min()), int(df['Age'].max()), (15, 45))
    max_budget = st.sidebar.slider("💰 Max Budget (€M)", 1.0, 200.0, 200.0)

    # --- 4. GLOBAL FILTERING LOGIC ---
    mask = (
        (df['Current_Val'] <= max_budget) & 
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Gls'] >= min_goals) # تطبيق فلتر الأهداف
    )
    f_df = df[mask].copy()
    
    if search:
        f_df = f_df[f_df['Player'].str.contains(search, case=False, na=False)]
    if selected_pos:
        f_df = f_df[f_df['Pos'].isin(selected_pos)]
    if selected_nation:
        f_df = f_df[f_df['Nation'].isin(selected_nation)]

    # --- 5. TOP METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shortlisted", len(f_df))
    m2.metric("Total Goals Found", int(f_df['Gls'].sum()) if not f_df.empty else 0)
    m3.metric("Avg Market Value", f"€{f_df['Current_Val'].mean():.1f}M" if not f_df.empty else "0")
    m4.metric("Growth Potential", f"+{((f_df['Projected_Val'].mean()/f_df['Current_Val'].mean()-1)*100):.1f}%" if not f_df.empty else "0%")

    # --- 6. THE SCOUTING REPORT ---
    st.subheader("📋 Comprehensive Global Report")
    if not f_df.empty:
        f_df['Stars'] = f_df.apply(lambda r: "⭐" * int(min(1 + (r['Efficiency']*4) + (2 if r['Age']<=22 else 0), 5)), axis=1)
        
        view_cols = ['Stars', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Gls', 'Efficiency', 'Current_Val', 'Projected_Val']
        
        st.dataframe(
            f_df[view_cols].rename(columns={
                'Gls': 'Goals', 'Efficiency': 'Gls/90', 
                'Current_Val': 'Value (€M)', 'Projected_Val': 'Proj. (€M)'
            }).sort_values('Goals', ascending=False),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("No global targets match these requirements.")

    # --- 7. VISUAL ANALYSIS ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Top Scorers Potential")
        if not f_df.empty:
            top_5 = f_df.sort_values('Gls', ascending=False).head(5)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=top_5['Player'], y=top_5['Current_Val'], name='Now', marker_color='#00d4ff'))
            fig_bar.add_trace(go.Bar(x=top_5['Player'], y=top_5['Projected_Val'], name='EOY', marker_color='#00ff88'))
            fig_bar.update_layout(barmode='group', template="plotly_dark", height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.subheader("🔬 Tactical Radar")
        if len(f_df) >= 2:
            p1_n = st.selectbox("Player A", f_df['Player'].unique(), index=0)
            p2_n = st.selectbox("Player B", f_df['Player'].unique(), index=1)
            p1 = f_df[f_df['Player'] == p1_n].iloc[0]
            p2 = f_df[f_df['Player'] == p2_n].iloc[0]
            categories = ['Goals', 'Efficiency', 'Potential']
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=[p1['Gls_norm'], p1['Eff_norm'], 1-p1['Age_norm']], theta=categories, fill='toself', name=f"{p1_n} ({p1['Gls']} Gls)"))
            fig_radar.add_trace(go.Scatterpolar(r=[p2['Gls_norm'], p2['Eff_norm'], 1-p2['Age_norm']], theta=categories, fill='toself', name=f"{p2_n} ({p2['Gls']} Gls)"))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template="plotly_dark", height=350)
            st.plotly_chart(fig_radar, use_container_width=True)

except Exception as e:
    st.error(f"🚨 Tactical Error: {e}")