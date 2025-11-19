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
    /* st.metric 색상 강조를 위한 커스텀 스타일 */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# 데이터 로드 및 전처리 함수 (캐싱 적용)
@st.cache_data
def load_data():
    # ⚠️ 사용자가 요청한 GitHub URL로 데이터를 로드합니다.
    data_url = "https://raw.githubusercontent.com/new000way/data_visualization_project/main/online_gaming_behavior_dataset.csv"

    try:
        df = pd.read_csv(data_url)
        
        # UserID가 'PlayerID'로 되어 있으므로 통일
        df = df.rename(columns={'PlayerID': 'UserID'})
        
        # LTV(평생 가치) 프록시 계산: 구매 횟수에 높은 가중치를 부여
        df['LTV_Proxy'] = df['InGamePurchases'] * 100 + df['PlayTimeHours'] * 5 + df['PlayerLevel']
        
        # Cohort 분석을 대체하기 위한 가상 시뮬레이션: (실제 데이터는 아니지만 구조 유지를 위해 포함)
        start_date = pd.to_datetime('2023-01-01')
        df['SimulatedRegistrationMonth'] = (start_date + pd.to_timedelta(df['UserID'] % 12 * 30, unit='days')).dt.to_period('M')

        return df
    except Exception as e:
        # GitHub URL 로드 실패 시 에러 메시지
        st.error(f"데이터 로드 오류: GitHub에서 데이터를 불러오는 데 실패했습니다. URL을 확인하거나 네트워크 연결을 점검해주세요. ({e})")
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
    ].copy() # SettingWithCopyWarning 방지를 위해 copy() 사용
    
    # 탭 구성 (Tab 4, 5의 내용 및 이름 변경)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 개요", "👥 유저 프로필", "🎮 게임 행동", "🧪 세그먼트 심화 분석", "🔮 LTV 및 이탈 인사이트"])
    
    with tab1:
        st.header("📊 데이터셋 개요 및 핵심 지표")
        
        # 주요 지표 (LTV Proxy 추가)
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
            avg_purchases = filtered_df['InGamePurchases'].mean()
            st.metric("평균 인게임 구매", f"{avg_purchases:.1f}회")
        with col5:
            avg_ltv = filtered_df['LTV_Proxy'].mean()
            st.metric("평균 LTV Proxy", f"₩{int(avg_ltv):,}")
        
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
    # 👥 유저 프로필 분석 (기존 탭 유지)
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
    # 🎮 게임 행동 패턴 분석 (기존 탭 유지)
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
            fig_purchases = px.box(
                filtered_df, x='EngagementLevel', y='InGamePurchases', title="인게이지먼트 레벨별 인게임 구매",
                labels={'EngagementLevel': '인게이지먼트 레벨', 'InGamePurchases': '구매 횟수'},
                color='EngagementLevel', color_discrete_sequence=px.colors.qualitative.Set3
            )
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
    # 🧪 세그먼트 심화 분석 (Tab 4: 플레이 시간 기반 세그먼트)
    # ----------------------------------------------------
    with tab4:
        st.header("🧪 플레이 시간 기반 유저 세그먼트 분석")
        st.markdown("유저들을 **플레이 시간** 중앙값을 기준으로 **헤비 유저**와 **라이트 유저**로 나누어 주요 행동 지표를 비교합니다.")
        
        # 플레이 시간 중앙값 기준으로 세그먼트 분리
        playtime_median = filtered_df['PlayTimeHours'].median()
        
        # .loc를 사용하여 안전하게 새로운 컬럼 생성
        filtered_df.loc[filtered_df['PlayTimeHours'] > playtime_median, 'TimeSegment'] = '🚀 헤비 유저 (Median 이상)'
        filtered_df.loc[filtered_df['PlayTimeHours'] <= playtime_median, 'TimeSegment'] = '🌱 라이트 유저 (Median 이하)'
        
        high_segment_df = filtered_df[filtered_df['TimeSegment'] == '🚀 헤비 유저 (Median 이상)']
        low_segment_df = filtered_df[filtered_df['TimeSegment'] == '🌱 라이트 유저 (Median 이하)']
        
        if not high_segment_df.empty and not low_segment_df.empty:
            
            # 1. KPI 비교
            st.subheader("⏱️ 헤비 유저 vs 라이트 유저 핵심 지표")
            col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4)
            
            with col_kpi_1:
                st.metric(label="헤비 유저 비율", value=f"{(len(high_segment_df) / len(filtered_df) * 100):.1f}%")
            with col_kpi_2:
                st.metric(label="헤비 유저 평균 LTV", value=f"₩{int(high_segment_df['LTV_Proxy'].mean()):,}")
            with col_kpi_3:
                st.metric(label="헤비 유저 평균 구매", value=f"{high_segment_df['InGamePurchases'].mean():.1f}회")
            with col_kpi_4:
                st.metric(label="라이트 유저 평균 세션", value=f"{low_segment_df['AvgSessionDurationMinutes'].mean():.0f}분")
            
            st.markdown("---")
            
            # 2. 인게이지먼트 레벨 분포 비교
            st.subheader("인게이지먼트 레벨별 유저 세그먼트 분포")
            
            engagement_segment = pd.crosstab(
                filtered_df['TimeSegment'], filtered_df['EngagementLevel'], normalize='index'
            ) * 100
            
            fig_engagement_segment = px.bar(
                engagement_segment,
                barmode='stack',
                title="플레이 시간 세그먼트별 인게이지먼트 비율",
                labels={'value': '비율 (%)', 'TimeSegment': '유저 세그먼트'},
                color_discrete_sequence=px.colors.sequential.Agsunset # 대비되는 색상 사용
            )
            st.plotly_chart(fig_engagement_segment, use_container_width=True)

            # 3. 평균 인게임 구매 비교
            st.subheader("평균 인게임 구매 vs 플레이어 레벨")
            
            fig_purchases_segment = px.scatter(
                filtered_df,
                x='PlayerLevel',
                y='InGamePurchases',
                color='TimeSegment',
                title="플레이어 레벨과 인게임 구매의 관계",
                labels={'PlayerLevel': '플레이어 레벨', 'InGamePurchases': '인게임 구매 횟수'},
                hover_data=['PlayTimeHours', 'EngagementLevel']
            )
            st.plotly_chart(fig_purchases_segment, use_container_width=True)

        else:
            st.warning("필터링된 데이터에 충분한 플레이 시간 세그먼트 구분이 어렵습니다.")

    # ----------------------------------------------------
    # 🔮 LTV 및 이탈 인사이트 (Tab 5: 고급 통계 및 인사이트)
    # ----------------------------------------------------
    with tab5:
        st.header("🔮 LTV 및 이탈 예측 인사이트")
        st.markdown("시간 데이터가 없는 단일 시점 데이터셋이므로, 현재 지표들을 통해 **잠재적 이탈 위험**과 **유저 가치**를 분석합니다.")

        # 1. 상관관계 히트맵
        st.subheader("📊 주요 변수 간 상관관계 (LTV 포함)")
        
        numeric_cols = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                        'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked', 'LTV_Proxy']
        
        # EngagementLevel을 숫자로 변환하여 상관관계에 포함 (High: 3, Medium: 2, Low: 1)
        engagement_map = {'Low': 1, 'Medium': 2, 'High': 3}
        filtered_df.loc[:, 'Engagement_Numeric'] = filtered_df['EngagementLevel'].map(engagement_map)
        
        final_corr_cols = numeric_cols + ['Engagement_Numeric']
        corr_matrix = filtered_df[final_corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            title="상관관계 히트맵 (Engagement 포함)",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.info("💡 **핵심 인사이트:** 'Engagement_Numeric' 행/열을 확인하여 플레이 시간, 구매, 레벨 등이 인게이지먼트와 얼마나 강한 양의 상관관계를 갖는지 확인하세요.")
        
        st.markdown("---")
        
        # 2. 이탈 위험 그룹 분석 (Low Engagement)
        st.subheader("🚨 잠재적 이탈 위험 그룹 (Low Engagement) 프로파일링")
        
        low_engagement_df = filtered_df[filtered_df['EngagementLevel'] == 'Low']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**평균 플레이 지표**")
            st.write(f"• 평균 플레이 시간: {low_engagement_df['PlayTimeHours'].mean():.1f}h")
            st.write(f"• 평균 세션 시간: {low_engagement_df['AvgSessionDurationMinutes'].mean():.0f}분")
            st.write(f"• 주간 세션: {low_engagement_df['SessionsPerWeek'].mean():.1f}회")
        
        with col2:
            st.markdown("**주요 행동**")
            st.write(f"• 평균 구매: {low_engagement_df['InGamePurchases'].mean():.1f}회")
            st.write(f"• 평균 업적: {low_engagement_df['AchievementsUnlocked'].mean():.0f}개")
            st.write(f"• 평균 레벨: {low_engagement_df['PlayerLevel'].mean():.0f}")
        
        with col3:
            st.markdown("**선호 난이도/장르**")
            most_difficulty = low_engagement_df['GameDifficulty'].mode()[0] if not low_engagement_df.empty else 'N/A'
            st.write(f"• 선호 난이도: {most_difficulty}")
            top_genre = low_engagement_df['GameGenre'].value_counts().idxmax() if not low_engagement_df.empty else 'N/A'
            st.write(f"• 최다 장르: {top_genre}")
            
        st.markdown("---")

        # 3. 비즈니스 제언
        st.subheader("💼 비즈니스 제언")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 이탈 방지 및 리인게이지먼트 전략**
            1. **플레이 시간 중앙값 이하** 유저에게 맞춤형 튜토리얼 또는 보상 제공 (라이트 유저 그룹).
            2. **낮은 평균 세션 시간** 유저에게 '일일 퀘스트' 등 짧은 시간 내 성취 가능한 콘텐츠 제공.
            3. **선호 난이도**와 **플레이 시간**의 관계를 분석하여, 진입장벽이 높다고 느껴 이탈하는 유저에게는 쉬운 난이도를 추천.
            """)
        
        with col2:
            st.markdown("""
            **💰 수익화 및 LTV 극대화 전략**
            1. **고관여 유저**가 선호하는 장르에 프리미엄 콘텐츠를 집중 출시하여 LTV를 극대화.
            2. **평균 LTV**가 높은 유저 그룹의 특징을 분석하여, 잠재적 고가치 유저에게 인게임 구매 유도.
            3. **PlayerLevel**이 높은 유저 대상의 독점 콘텐츠로 로열티 및 지출 증가 유도.
            """)
    
    # 데이터 다운로드 (기존 유지)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 데이터 다운로드")
    
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button(
        label="필터링된 데이터 다운로드 (CSV)",
        data=csv,
        file_name='filtered_gaming_data.csv',
        mime='text/csv',
    )
    
    # 푸터 (기존 유지)
    st.sidebar.markdown("---")
    st.sidebar.caption("🎮 온라인 게임 유저 행동 분석 대시보드")
    st.sidebar.caption("Data Source: GitHub / Analysis by Gemini")

else:
    st.error("❌ 데이터를 불러올 수 없습니다. GitHub URL을 확인하거나 Streamlit 배포 환경 설정을 점검해주세요.")
