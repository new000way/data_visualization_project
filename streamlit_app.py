import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="🎮 온라인 게임 유저 행동 분석",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# 데이터 로드 및 전처리 함수 (캐싱 적용)
@st.cache_data
def load_data():
    data_url = "https://raw.githubusercontent.com/new000way/data_visualization_project/refs/heads/main/online_gaming_behavior_datasets.csv"

    try:
        df = pd.read_csv(data_url)
        
        # UserID가 'PlayerID'로 되어 있으므로 통일
        df = df.rename(columns={'PlayerID': 'UserID'})
        
        # LTV(평생 가치) 프록시 계산: 구매 여부에 높은 가중치 부여
        df['LTV_Proxy'] = df['InGamePurchases'] * 5000 + df['PlayTimeHours'] * 100 + df['PlayerLevel'] * 10
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: GitHub에서 데이터를 불러오는 데 실패했습니다. ({e})")
        return None

# 데이터 로드
df = load_data()

if df is not None:
    # 헤더
    st.markdown('<p class="main-header">🎮 온라인 게임 유저 행동 분석 대시보드</p>', unsafe_allow_html=True)
    
    # 사이드바
    st.sidebar.title("📊 필터 옵션")
    st.sidebar.markdown("---")
    
    # 필터
    selected_gender = st.sidebar.multiselect(
        "성별 선택",
        options=df['Gender'].unique(),
        default=df['Gender'].unique()
    )
    
    selected_genre = st.sidebar.multiselect(
        "게임 장르 선택",
        options=df['GameGenre'].unique(),
        default=df['GameGenre'].unique()
    )
    
    age_range = st.sidebar.slider(
        "나이 범위",
        int(df['Age'].min()),
        int(df['Age'].max()),
        (int(df['Age'].min()), int(df['Age'].max()))
    )
    
    # 데이터 필터링
    filtered_df = df[
        (df['Gender'].isin(selected_gender)) &
        (df['GameGenre'].isin(selected_genre)) &
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1])
    ].copy()
    
    # 탭 구성 (제목 변경)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 개요", 
        "👥 유저 프로필", 
        "🎮 게임 행동", 
        "📊 헤비 vs 라이트 유저", 
        "💎 유저 가치 분석"
    ])
    
    with tab1:
        st.header("📊 데이터셋 개요 및 핵심 지표")
        
        # 주요 지표 (구매율로 변경)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("전체 유저 수", f"{len(filtered_df):,}")
        with col2:
            avg_playtime = filtered_df['PlayTimeHours'].mean()
            st.metric("평균 플레이 시간", f"{avg_playtime:.1f}h")
        with col3:
            high_engagement = (filtered_df['EngagementLevel'] == 'High').sum()
            st.metric("고관여 유저", f"{high_engagement:,}")
        with col4:
            # 구매율로 변경
            purchase_rate = (filtered_df['InGamePurchases'] == 1).mean() * 100
            st.metric("구매 유저 비율", f"{purchase_rate:.1f}%")
        with col5:
            avg_ltv = filtered_df['LTV_Proxy'].mean()
            st.metric("평균 유저 가치", f"₩{int(avg_ltv):,}")
        
        st.markdown("---")
        
        # 인게이지먼트 레벨 분포
        col1, col2 = st.columns(2)
        
        with col1:
            engagement_counts = filtered_df['EngagementLevel'].value_counts()
            fig_pie = px.pie(
                values=engagement_counts.values,
                names=engagement_counts.index,
                title="인게이지먼트 레벨 분포",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            genre_counts = filtered_df['GameGenre'].value_counts()
            fig_bar = px.bar(
                x=genre_counts.index,
                y=genre_counts.values,
                title="게임 장르별 유저 수",
                labels={'x': '게임 장르', 'y': '유저 수'},
                color=genre_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # ----------------------------------------------------
    # 👥 유저 프로필 분석
    # ----------------------------------------------------
    with tab2:
        st.header("👥 유저 프로필 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 나이 분포
            fig_age = px.histogram(
                filtered_df, x='Age', nbins=30, title="나이 분포",
                labels={'Age': '나이', 'count': '유저 수'}, color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
            # 성별 분포
            gender_counts = filtered_df['Gender'].value_counts()
            fig_gender = px.bar(
                x=gender_counts.index, y=gender_counts.values, title="성별 분포",
                labels={'x': '성별', 'y': '유저 수'}, 
                color=gender_counts.index, color_discrete_map={'Male': '#636EFA', 'Female': '#EF553B'}
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            # 위치별 분포 (Top 10)
            location_counts = filtered_df['Location'].value_counts().head(10)
            fig_location = px.bar(
                x=location_counts.values, y=location_counts.index, orientation='h', 
                title="상위 10개 지역별 유저 수", labels={'x': '유저 수', 'y': '지역'}, 
                color=location_counts.values, color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_location, use_container_width=True)
            
            # 나이 vs 인게이지먼트
            fig_age_engagement = px.box(
                filtered_df, x='EngagementLevel', y='Age', title="인게이지먼트 레벨별 나이 분포",
                labels={'EngagementLevel': '인게이지먼트 레벨', 'Age': '나이'}, color='EngagementLevel',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_age_engagement, use_container_width=True)

    # ----------------------------------------------------
    # 🎮 게임 행동 패턴 분석
    # ----------------------------------------------------
    with tab3:
        st.header("🎮 게임 행동 패턴 분석")
        
        # 플레이 시간 vs 인게이지먼트
        col1, col2 = st.columns(2)
        
        with col1:
            fig_playtime = px.box(
                filtered_df, x='EngagementLevel', y='PlayTimeHours', title="인게이지먼트 레벨별 플레이 시간",
                labels={'EngagementLevel': '인게이지먼트 레벨', 'PlayTimeHours': '플레이 시간 (시간)'},
                color='EngagementLevel', color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_playtime, use_container_width=True)
        
        with col2:
            # 구매율로 변경
            purchase_by_engagement = filtered_df.groupby('EngagementLevel')['InGamePurchases'].apply(
                lambda x: (x == 1).mean() * 100
            ).reset_index()
            purchase_by_engagement.columns = ['EngagementLevel', 'PurchaseRate']
            
            fig_purchases = px.bar(
                purchase_by_engagement, 
                x='EngagementLevel', 
                y='PurchaseRate', 
                title="인게이지먼트 레벨별 구매 유저 비율",
                labels={'EngagementLevel': '인게이지먼트 레벨', 'PurchaseRate': '구매율 (%)'},
                color='EngagementLevel',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_purchases.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            st.plotly_chart(fig_purchases, use_container_width=True)
        
        # 게임 난이도 vs 인게이지먼트
        col1, col2 = st.columns(2)
        
        with col1:
            difficulty_engagement = pd.crosstab(
                filtered_df['GameDifficulty'], filtered_df['EngagementLevel'], normalize='index'
            ) * 100
            
            fig_difficulty = px.bar(
                difficulty_engagement, barmode='group', title="게임 난이도별 인게이지먼트 분포 (%)",
                labels={'value': '비율 (%)', 'GameDifficulty': '게임 난이도'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_difficulty, use_container_width=True)
        
        with col2:
            # 주간 세션 수 vs 평균 세션 시간
            fig_scatter = px.scatter(
                filtered_df, x='SessionsPerWeek', y='AvgSessionDurationMinutes', color='EngagementLevel',
                title="주간 세션 수 vs 평균 세션 시간",
                labels={'SessionsPerWeek': '주간 세션 수', 'AvgSessionDurationMinutes': '평균 세션 시간 (분)'},
                opacity=0.6, size='PlayTimeHours', hover_data=['Age', 'Gender', 'GameGenre']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ----------------------------------------------------
    # 📊 헤비 vs 라이트 유저 (Tab 4: 제목 변경)
    # ----------------------------------------------------
    with tab4:
        st.header("📊 헤비 유저 vs 라이트 유저 비교")
        st.markdown("유저들을 **플레이 시간** 중앙값을 기준으로 **헤비 유저**와 **라이트 유저**로 나누어 비교합니다.")
        
        # 플레이 시간 중앙값 기준으로 세그먼트 분리
        playtime_median = filtered_df['PlayTimeHours'].median()
        
        filtered_df.loc[filtered_df['PlayTimeHours'] > playtime_median, 'TimeSegment'] = '🚀 헤비 유저 (많이 플레이)'
        filtered_df.loc[filtered_df['PlayTimeHours'] <= playtime_median, 'TimeSegment'] = '🌱 라이트 유저 (조금 플레이)'
        
        high_segment_df = filtered_df[filtered_df['TimeSegment'] == '🚀 헤비 유저 (많이 플레이)']
        low_segment_df = filtered_df[filtered_df['TimeSegment'] == '🌱 라이트 유저 (조금 플레이)']
        
        if not high_segment_df.empty and not low_segment_df.empty:
            
            # 1. KPI 비교
            st.subheader("⏱️ 주요 지표 비교")
            col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4)
            
            with col_kpi_1:
                st.metric(label="헤비 유저 비율", value=f"{(len(high_segment_df) / len(filtered_df) * 100):.1f}%")
            with col_kpi_2:
                heavy_purchase_rate = (high_segment_df['InGamePurchases'] == 1).mean() * 100
                st.metric(label="헤비 유저 구매율", value=f"{heavy_purchase_rate:.1f}%")
            with col_kpi_3:
                light_purchase_rate = (low_segment_df['InGamePurchases'] == 1).mean() * 100
                st.metric(label="라이트 유저 구매율", value=f"{light_purchase_rate:.1f}%")
            with col_kpi_4:
                st.metric(label="라이트 유저 평균 세션", value=f"{low_segment_df['AvgSessionDurationMinutes'].mean():.0f}분")
            
            st.markdown("---")
            
            # 2. 인게이지먼트 레벨 분포 비교
            st.subheader("인게이지먼트 레벨 비교")
            
            engagement_segment = pd.crosstab(
                filtered_df['TimeSegment'], filtered_df['EngagementLevel'], normalize='index'
            ) * 100
            
            fig_engagement_segment = px.bar(
                engagement_segment,
                barmode='stack',
                title="헤비 vs 라이트 유저의 인게이지먼트 비율",
                labels={'value': '비율 (%)', 'TimeSegment': '유저 유형'},
                color_discrete_sequence=px.colors.sequential.Agsunset
            )
            st.plotly_chart(fig_engagement_segment, use_container_width=True)

            # 3. 구매 비율 비교
            st.subheader("플레이어 레벨별 구매 유저 비율")
            
            # 레벨 구간별로 구매율 계산
            filtered_df['LevelBin'] = pd.cut(filtered_df['PlayerLevel'], bins=10)
            purchase_by_level = filtered_df.groupby(['LevelBin', 'TimeSegment'])['InGamePurchases'].apply(
                lambda x: (x == 1).mean() * 100
            ).reset_index()
            purchase_by_level.columns = ['LevelBin', 'TimeSegment', 'PurchaseRate']
            purchase_by_level['LevelBin'] = purchase_by_level['LevelBin'].astype(str)
            
            fig_purchases_segment = px.bar(
                purchase_by_level,
                x='LevelBin',
                y='PurchaseRate',
                color='TimeSegment',
                barmode='group',
                title="플레이어 레벨 구간별 구매율",
                labels={'LevelBin': '플레이어 레벨 구간', 'PurchaseRate': '구매율 (%)'}
            )
            fig_purchases_segment.update_xaxes(tickangle=45)
            st.plotly_chart(fig_purchases_segment, use_container_width=True)

        else:
            st.warning("필터링된 데이터에 충분한 유저가 없습니다.")

    # ----------------------------------------------------
    # 💎 유저 가치 분석 (Tab 5: 제목 변경, 비즈니스 제언 삭제)
    # ----------------------------------------------------
    with tab5:
        st.header("💎 유저 가치 및 중요 패턴 분석")
        st.markdown("각 변수들이 서로 어떤 관계를 가지는지, 어떤 유저가 가치 있는지 분석합니다.")

        # 1. 상관관계 히트맵
        st.subheader("📊 주요 변수 간 상관관계")
        
        numeric_cols = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                        'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
        
        # EngagementLevel을 숫자로 변환하여 상관관계에 포함
        engagement_map = {'Low': 1, 'Medium': 2, 'High': 3}
        filtered_df.loc[:, 'Engagement_Numeric'] = filtered_df['EngagementLevel'].map(engagement_map)
        
        final_corr_cols = numeric_cols + ['Engagement_Numeric']
        corr_matrix = filtered_df[final_corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            title="변수 간 상관관계 (숫자가 클수록 강한 관계)",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.info("💡 **읽는 법:** 빨간색에 가까울수록 양의 관계(함께 증가), 파란색에 가까울수록 음의 관계(역으로 증가)입니다.")
        
        st.markdown("---")
        
        # 2. 고관여 유저 프로필
        st.subheader("🏆 고관여 유저 (High Engagement) 프로필")
        
        high_engagement_df = filtered_df[filtered_df['EngagementLevel'] == 'High']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**평균 플레이 지표**")
            st.write(f"• 평균 플레이 시간: {high_engagement_df['PlayTimeHours'].mean():.1f}시간")
            st.write(f"• 평균 세션 시간: {high_engagement_df['AvgSessionDurationMinutes'].mean():.0f}분")
            st.write(f"• 주간 세션: {high_engagement_df['SessionsPerWeek'].mean():.1f}회")
        
        with col2:
            st.markdown("**주요 행동**")
            high_purchase_rate = (high_engagement_df['InGamePurchases'] == 1).mean() * 100
            st.write(f"• 구매율: {high_purchase_rate:.1f}%")
            st.write(f"• 평균 업적: {high_engagement_df['AchievementsUnlocked'].mean():.0f}개")
            st.write(f"• 평균 레벨: {high_engagement_df['PlayerLevel'].mean():.0f}")
        
        with col3:
            st.markdown("**선호 스타일**")
            most_difficulty = high_engagement_df['GameDifficulty'].mode()[0] if not high_engagement_df.empty else 'N/A'
            st.write(f"• 선호 난이도: {most_difficulty}")
            top_genre = high_engagement_df['GameGenre'].value_counts().idxmax() if not high_engagement_df.empty else 'N/A'
            st.write(f"• 최다 장르: {top_genre}")
            st.write(f"• 평균 나이: {high_engagement_df['Age'].mean():.1f}세")
        
        st.markdown("---")
        
        # 3. 저관여 유저 프로필
        st.subheader("🚨 저관여 유저 (Low Engagement) 프로필")
        
        low_engagement_df = filtered_df[filtered_df['EngagementLevel'] == 'Low']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**평균 플레이 지표**")
            st.write(f"• 평균 플레이 시간: {low_engagement_df['PlayTimeHours'].mean():.1f}시간")
            st.write(f"• 평균 세션 시간: {low_engagement_df['AvgSessionDurationMinutes'].mean():.0f}분")
            st.write(f"• 주간 세션: {low_engagement_df['SessionsPerWeek'].mean():.1f}회")
        
        with col2:
            st.markdown("**주요 행동**")
            low_purchase_rate = (low_engagement_df['InGamePurchases'] == 1).mean() * 100
            st.write(f"• 구매율: {low_purchase_rate:.1f}%")
            st.write(f"• 평균 업적: {low_engagement_df['AchievementsUnlocked'].mean():.0f}개")
            st.write(f"• 평균 레벨: {low_engagement_df['PlayerLevel'].mean():.0f}")
        
        with col3:
            st.markdown("**선호 스타일**")
            most_difficulty = low_engagement_df['GameDifficulty'].mode()[0] if not low_engagement_df.empty else 'N/A'
            st.write(f"• 선호 난이도: {most_difficulty}")
            top_genre = low_engagement_df['GameGenre'].value_counts().idxmax() if not low_engagement_df.empty else 'N/A'
            st.write(f"• 최다 장르: {top_genre}")
            st.write(f"• 평균 나이: {low_engagement_df['Age'].mean():.1f}세")
    
    # 데이터 다운로드
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 데이터 다운로드")
    
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button(
        label="필터링된 데이터 다운로드 (CSV)",
        data=csv,
        file_name='filtered_gaming_data.csv',
        mime='text/csv',
    )
    
    # 푸터
    st.sidebar.markdown("---")
    st.sidebar.caption("🎮 온라인 게임 유저 행동 분석 대시보드")
    st.sidebar.caption("Data Source: GitHub")

else:
    st.error("❌ 데이터를 불러올 수 없습니다. GitHub URL을 확인해주세요.")
