import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import random # 시뮬레이션 데이터 생성을 위해 추가

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
    # GitHub raw URL을 사용하세요
    url = 'https://raw.githubusercontent.com/new000way/data_visualization_project/main/online_gaming_behavior_dataset.csv'

    try:
        df = pd.read_csv(url)
        # UserID가 없을 경우 임시로 생성 (코호트 분석을 위해 필수)
        if 'UserID' not in df.columns:
             df['UserID'] = range(1, len(df) + 1)
             
        # LTV(평생 가치) 프록시 계산 (결제 횟수와 플레이 시간을 가중치로 부여)
        df['LTV_Proxy'] = df['InGamePurchases'] * 50 + df['PlayTimeHours']
        
        # ------------------------------------------------
        # ⏳ 코호트 분석을 위한 시간 데이터 시뮬레이션
        # ------------------------------------------------
        # 실제 데이터셋에 등록 날짜가 없으므로, 플레이 시간을 기준으로 가상의 날짜를 할당합니다.
        # 실제 프로젝트에서는 'RegistrationDate' 또는 'JoinDate' 컬럼을 사용해야 합니다.
        start_date = pd.to_datetime('2023-01-01')
        df['RegistrationDate'] = start_date + pd.to_timedelta(df['PlayTimeHours'] * 3 + df['UserID'] % 30, unit='days')
        df['RegistrationMonth'] = df['RegistrationDate'].dt.to_period('M')
        
        # 현재 분석 기준 월 (가장 최근 데이터가 있는 월로 가정)
        df['CurrentMonth'] = pd.to_datetime('2023-12-01').to_period('M') 
        
        # CohortMonth: 각 유저의 최소 등록 월 (가장 처음 가입한 월)
        df['CohortMonth'] = df.groupby('UserID')['RegistrationDate'].transform('min').dt.to_period('M')

        # Cohort Index: CohortMonth로부터 경과된 월 수 (리텐션 분석의 핵심)
        def get_month_difference(earlier, later):
            return later.year * 12 + later.month - (earlier.year * 12 + earlier.month)
        
        df['CohortIndex'] = df.apply(lambda row: get_month_difference(row['CohortMonth'], row['CurrentMonth']), axis=1)
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        st.info("💡 Kaggle에서 다운로드한 CSV 파일을 GitHub 레포지토리에 업로드하고 URL을 수정하세요.")
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
    ]
    
    # 탭 구성 (4개 -> 5개로 확장)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 개요", "👥 유저 프로필", "🎮 게임 행동", "🧪 세그먼트 분석", "⏳ 코호트/리텐션"])
    
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
    # 🧪 세그먼트 분석 (신규 추가)
    # ----------------------------------------------------
    with tab4:
        st.header("🧪 구매 행동 기반 유저 세그먼트 분석")
        st.markdown("유저들을 인게임 구매 횟수를 기준으로 **고구매 유저**와 **저구매 유저**로 나누어 주요 지표를 비교합니다.")
        
        # 구매 횟수 중앙값 기준으로 세그먼트 분리
        purchase_median = filtered_df['InGamePurchases'].median()
        high_purchaser_df = filtered_df[filtered_df['InGamePurchases'] > purchase_median]
        low_purchaser_df = filtered_df[filtered_df['InGamePurchases'] <= purchase_median]
        
        if not high_purchaser_df.empty and not low_purchaser_df.empty:
            
            # 1. KPI 비교
            st.subheader("💰 고구매 vs 저구매 유저 핵심 지표")
            col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3)
            
            with col_kpi_1:
                st.metric(label="고구매 유저 비율", value=f"{(len(high_purchaser_df) / len(filtered_df) * 100):.1f}%")
            with col_kpi_2:
                st.metric(label="고구매 유저 평균 LTV", value=f"₩{int(high_purchaser_df['LTV_Proxy'].mean()):,}")
            with col_kpi_3:
                st.metric(label="고구매 유저 평균 플레이 시간", value=f"{high_purchaser_df['PlayTimeHours'].mean():.1f}h")
            
            st.markdown("---")
            
            # 2. 플레이 시간 분포 비교
            st.subheader("플레이 시간 및 세션 지표 비교")
            
            # 두 그룹의 플레이 시간 데이터 준비
            playtime_data = pd.DataFrame({
                'PlayTimeHours': high_purchaser_df['PlayTimeHours'].tolist() + low_purchaser_df['PlayTimeHours'].tolist(),
                'Segment': ['고구매 유저'] * len(high_purchaser_df) + ['저구매 유저'] * len(low_purchaser_df)
            })
            
            fig_segment = px.box(
                playtime_data,
                x='Segment',
                y='PlayTimeHours',
                color='Segment',
                title="구매 세그먼트별 플레이 시간 분포",
                labels={'PlayTimeHours': '플레이 시간 (시간)'}
            )
            st.plotly_chart(fig_segment, use_container_width=True)
            
            # 3. 선호 장르 비교 (고구매 유저)
            st.subheader("고구매 유저의 선호 장르")
            genre_high_purchaser = high_purchaser_df['GameGenre'].value_counts(normalize=True).head(5) * 100
            
            fig_genre_segment = px.bar(
                x=genre_high_purchaser.index,
                y=genre_high_purchaser.values,
                title="고구매 유저의 장르 선호도 (%)",
                labels={'x': '게임 장르', 'y': '비율 (%)'}
            )
            st.plotly_chart(fig_genre_segment, use_container_width=True)

        else:
            st.warning("필터링된 데이터에 충분한 구매 세그먼트 구분이 어렵습니다.")

    # ----------------------------------------------------
    # ⏳ 코호트/리텐션 분석 (신규 추가, 가장 전문적인 부분)
    # ----------------------------------------------------
    with tab5:
        st.header("⏳ 코호트 분석 및 리텐션 히트맵")
        st.warning("⚠️ **주의:** 이 분석은 'PlayTimeHours'를 기반으로 **가상의 등록 날짜**를 생성하여 수행되었습니다. 실제 데이터에는 등록일 컬럼이 필요합니다.")
        
        # 코호트 데이터 생성
        cohort_data = filtered_df.groupby(['CohortMonth', 'CohortIndex'])['UserID'].nunique().reset_index()
        cohort_pivot = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='UserID')
        
        # 리텐션 비율 계산 (Cohort Index 0이 초기 유저 수)
        cohort_sizes = cohort_pivot.iloc[:, 0]
        retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)
        
        st.subheader("1. 유저 리텐션 히트맵")
        
        # Plotly 히트맵 생성
        fig_retention = px.imshow(
            retention_matrix * 100, # %로 표시
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels={'x': '경과 월 (Cohort Index)', 'y': '가입 월 (Cohort Month)', 'color': '리텐션 (%)'},
            title="월별 리텐션 비율 히트맵"
        )
        
        fig_retention.update_layout(
            xaxis=dict(tickvals=list(retention_matrix.columns), ticktext=[f"{i}개월" for i in retention_matrix.columns]),
            yaxis=dict(tickvals=list(retention_matrix.index), ticktext=[str(m) for m in retention_matrix.index]),
            height=600
        )
        st.plotly_chart(fig_retention, use_container_width=True)
        

        st.markdown("---")
        st.subheader("2. 코호트별 지표 요약 (가입 월별)")
        
        # 가입 월별 평균 지표 계산
        cohort_summary = filtered_df.groupby('CohortMonth').agg(
            Total_Users=('UserID', 'nunique'),
            Avg_PlayTime=('PlayTimeHours', 'mean'),
            Avg_LTV=('LTV_Proxy', 'mean')
        ).reset_index()
        
        # 데이터 시각화 (Total Users)
        fig_cohort_users = px.bar(
            cohort_summary,
            x='CohortMonth',
            y='Total_Users',
            title="가입 월별 총 유저 수",
            labels={'CohortMonth': '가입 월', 'Total_Users': '유저 수'}
        )
        st.plotly_chart(fig_cohort_users, use_container_width=True)
        
        # 데이터 시각화 (Avg LTV)
        fig_cohort_ltv = px.line(
            cohort_summary,
            x='CohortMonth',
            y='Avg_LTV',
            title="가입 월별 평균 LTV Proxy 변화",
            labels={'CohortMonth': '가입 월', 'Avg_LTV': '평균 LTV Proxy'},
            markers=True
        )
        st.plotly_chart(fig_cohort_ltv, use_container_width=True)


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
    st.sidebar.caption("Data: Kaggle - Predict Online Gaming Behavior Dataset")

else:
    st.error("❌ 데이터를 불러올 수 없습니다. GitHub URL을 확인해주세요.")
