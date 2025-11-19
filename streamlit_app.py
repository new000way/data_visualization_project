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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 데이터 로드 함수 (캐싱)
@st.cache_data
def load_data():
    # GitHub raw URL을 사용하세요
    # 예: 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/online_gaming_behavior_dataset.csv'
    url = 'https://raw.githubusercontent.com/new000way/data_visualization_project/main/online_gaming_behavior_dataset.csv'

    
    try:
        df = pd.read_csv(url)
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
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📈 개요", "🎯 유저 분석", "🎮 게임 행동", "🔮 인게이지먼트 예측"])
    
    with tab1:
        st.header("📊 데이터셋 개요")
        
        # 주요 지표
        col1, col2, col3, col4 = st.columns(4)
        
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
    
    with tab2:
        st.header("👥 유저 프로필 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 나이 분포
            fig_age = px.histogram(
                filtered_df,
                x='Age',
                nbins=30,
                title="나이 분포",
                labels={'Age': '나이', 'count': '유저 수'},
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
            # 성별 분포
            gender_counts = filtered_df['Gender'].value_counts()
            fig_gender = px.bar(
                x=gender_counts.index,
                y=gender_counts.values,
                title="성별 분포",
                labels={'x': '성별', 'y': '유저 수'},
                color=gender_counts.index,
                color_discrete_map={'Male': '#636EFA', 'Female': '#EF553B'}
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            # 위치별 분포 (Top 10)
            location_counts = filtered_df['Location'].value_counts().head(10)
            fig_location = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="상위 10개 지역별 유저 수",
                labels={'x': '유저 수', 'y': '지역'},
                color=location_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_location, use_container_width=True)
            
            # 나이 vs 인게이지먼트
            fig_age_engagement = px.box(
                filtered_df,
                x='EngagementLevel',
                y='Age',
                title="인게이지먼트 레벨별 나이 분포",
                labels={'EngagementLevel': '인게이지먼트 레벨', 'Age': '나이'},
                color='EngagementLevel',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_age_engagement, use_container_width=True)
    
    with tab3:
        st.header("🎮 게임 행동 패턴 분석")
        
        # 플레이 시간 vs 인게이지먼트
        col1, col2 = st.columns(2)
        
        with col1:
            fig_playtime = px.box(
                filtered_df,
                x='EngagementLevel',
                y='PlayTimeHours',
                title="인게이지먼트 레벨별 플레이 시간",
                labels={'EngagementLevel': '인게이지먼트 레벨', 'PlayTimeHours': '플레이 시간 (시간)'},
                color='EngagementLevel',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_playtime, use_container_width=True)
        
        with col2:
            fig_purchases = px.box(
                filtered_df,
                x='EngagementLevel',
                y='InGamePurchases',
                title="인게이지먼트 레벨별 인게임 구매",
                labels={'EngagementLevel': '인게이지먼트 레벨', 'InGamePurchases': '구매 횟수'},
                color='EngagementLevel',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_purchases, use_container_width=True)
        
        # 게임 난이도 vs 인게이지먼트
        col1, col2 = st.columns(2)
        
        with col1:
            difficulty_engagement = pd.crosstab(
                filtered_df['GameDifficulty'],
                filtered_df['EngagementLevel'],
                normalize='index'
            ) * 100
            
            fig_difficulty = px.bar(
                difficulty_engagement,
                barmode='group',
                title="게임 난이도별 인게이지먼트 분포 (%)",
                labels={'value': '비율 (%)', 'GameDifficulty': '게임 난이도'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_difficulty, use_container_width=True)
        
        with col2:
            # 주간 세션 수 vs 평균 세션 시간
            fig_scatter = px.scatter(
                filtered_df,
                x='SessionsPerWeek',
                y='AvgSessionDurationMinutes',
                color='EngagementLevel',
                title="주간 세션 수 vs 평균 세션 시간",
                labels={
                    'SessionsPerWeek': '주간 세션 수',
                    'AvgSessionDurationMinutes': '평균 세션 시간 (분)',
                    'EngagementLevel': '인게이지먼트'
                },
                opacity=0.6,
                size='PlayTimeHours',
                hover_data=['Age', 'Gender', 'GameGenre']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 게임 장르별 인게이지먼트
        st.subheader("🎯 게임 장르별 인게이지먼트 분석")
        genre_engagement = pd.crosstab(
            filtered_df['GameGenre'],
            filtered_df['EngagementLevel'],
            normalize='index'
        ) * 100
        
        fig_genre_engagement = px.bar(
            genre_engagement,
            barmode='stack',
            title="게임 장르별 인게이지먼트 레벨 비율",
            labels={'value': '비율 (%)', 'GameGenre': '게임 장르'},
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_genre_engagement, use_container_width=True)
    
    with tab4:
        st.header("🔮 인게이지먼트 예측 인사이트")
        
        st.info("""
        💡 **주요 발견사항:**
        - 플레이 시간이 높을수록 인게이지먼트가 높아집니다
        - 인게임 구매는 고관여 유저의 특징입니다
        - RPG와 Strategy 장르가 높은 인게이지먼트를 보입니다
        """)
        
        # 상관관계 히트맵
        st.subheader("📊 주요 변수 간 상관관계")
        
        numeric_cols = ['Age', 'PlayTimeHours', 'InGamePurchases', 
                       'SessionsPerWeek', 'AvgSessionDurationMinutes']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            title="상관관계 히트맵",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # 고관여 유저 프로필
        st.subheader("🏆 고관여 유저 프로필")
        
        high_engagement_df = filtered_df[filtered_df['EngagementLevel'] == 'High']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**평균 프로필**")
            st.write(f"• 평균 나이: {high_engagement_df['Age'].mean():.1f}세")
            st.write(f"• 평균 플레이 시간: {high_engagement_df['PlayTimeHours'].mean():.1f}시간")
            st.write(f"• 평균 구매: {high_engagement_df['InGamePurchases'].mean():.1f}회")
        
        with col2:
            st.markdown("**선호 장르**")
            top_genres = high_engagement_df['GameGenre'].value_counts().head(3)
            for genre, count in top_genres.items():
                st.write(f"• {genre}: {count}명")
        
        with col3:
            st.markdown("**게임 습관**")
            st.write(f"• 주간 세션: {high_engagement_df['SessionsPerWeek'].mean():.1f}회")
            st.write(f"• 평균 세션: {high_engagement_df['AvgSessionDurationMinutes'].mean():.0f}분")
            most_difficulty = high_engagement_df['GameDifficulty'].mode()[0]
            st.write(f"• 선호 난이도: {most_difficulty}")
        
        # 액션 아이템
        st.markdown("---")
        st.subheader("💼 비즈니스 제언")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 유저 리텐션 전략**
            1. 플레이 시간 10시간 미만 유저에게 보상 제공
            2. 중간 난이도 게임 추천으로 진입장벽 낮추기
            3. 주간 3회 이상 접속 유도 이벤트
            """)
        
        with col2:
            st.markdown("""
            **💰 수익화 전략**
            1. 고관여 유저 대상 프리미엄 콘텐츠
            2. RPG/Strategy 장르 집중 투자
            3. 세션당 45분 이상 유저 타겟팅
            """)
    
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
    st.sidebar.caption("Data: Kaggle - Predict Online Gaming Behavior Dataset")

else:
    st.error("❌ 데이터를 불러올 수 없습니다. GitHub URL을 확인해주세요.")
