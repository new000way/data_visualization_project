import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from math import exp # math.exp를 사용하여 Exp(B) 계산

# 머신러닝 (ML) 모델링을 위한 라이브러리 추가
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
    .risk-factor {
        color: #cc0000;
        font-weight: bold;
    }
    .retention-factor {
        color: #008000;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 데이터 로드 및 전처리 함수 (캐싱 적용)
@st.cache_data
def load_data():
    # 실제 데이터셋 URL을 사용하거나, 로컬 파일을 가정합니다.
    # GitHub URL이 작동하지 않을 경우를 대비하여 로컬 파일명을 주석으로 남깁니다.
    data_url = "https://raw.githubusercontent.com/new000way/data_visualization_project/refs/heads/main/online_gaming_behavior_datasets.csv"

    try:
        # 데이터가 이미 로드된 것으로 가정하고, 이 코드를 유지합니다.
        df = pd.read_csv(data_url)
        
        # UserID가 'PlayerID'로 되어 있으므로 통일
        df = df.rename(columns={'PlayerID': 'UserID'})
        
        # LTV(평생 가치) 프록시 계산: 구매 여부에 높은 가중치 부여
        df['LTV_Proxy'] = df['InGamePurchases'] * 5000 + df['PlayTimeHours'] * 100 + df['PlayerLevel'] * 10
        
        # EngagementLevel 순서 정의
        engagement_order = ['Low', 'Medium', 'High']
        df['EngagementLevel'] = pd.Categorical(df['EngagementLevel'], categories=engagement_order, ordered=True)
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: 데이터를 불러오는 데 실패했습니다. ({e})")
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
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 개요", 
        "👥 유저 프로필", 
        "🎮 게임 행동", 
        "🚀 참여율 증진 요인 분석", # Tab 4 제목 변경 유지
        "🚫 사용자 이탈 예측 모델"
    ])
    
    # Tab 1: 개요
    with tab1:
        st.header("📊 데이터셋 개요 및 핵심 지표")
        
        # 주요 지표
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
            purchase_rate = (filtered_df['InGamePurchases'] == 1).mean() * 100
            st.metric("구매 유저 비율", f"{purchase_rate:.1f}%")
        with col5:
            avg_ltv = filtered_df['LTV_Proxy'].mean()
            st.metric("평균 유저 가치", f"₩{int(avg_ltv):,}")
        
        st.markdown("---")
        
        # 인게이지먼트 레벨 분포
        col1, col2 = st.columns(2)
        
        with col1:
            engagement_counts = filtered_df['EngagementLevel'].value_counts().sort_index()
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
    
    # Tab 2: 유저 프로필 분석 (변화 없음)
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
                category_orders={"EngagementLevel": ['Low', 'Medium', 'High']},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_age_engagement, use_container_width=True)

    # Tab 3: 게임 행동 패턴 분석 (난이도 차트 제거)
    with tab3:
        st.header("🎮 게임 행동 패턴 분석")
        
        # 플레이 시간 vs 인게이지먼트
        col1, col2 = st.columns(2)
        
        with col1:
            fig_playtime = px.box(
                filtered_df, x='EngagementLevel', y='PlayTimeHours', title="인게이지먼트 레벨별 플레이 시간",
                labels={'EngagementLevel': '인게이지먼트 레벨', 'PlayTimeHours': '플레이 시간 (시간)'},
                color='EngagementLevel',
                category_orders={"EngagementLevel": ['Low', 'Medium', 'High']},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_playtime, use_container_width=True)
        
        with col2:
            # 구매율
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
                category_orders={"EngagementLevel": ['Low', 'Medium', 'High']},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_purchases.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            st.plotly_chart(fig_purchases, use_container_width=True)
        
        # 주간 세션 수 vs 평균 세션 시간 (Tab 3에 유지)
        st.subheader("주간 세션 수 vs 평균 세션 시간")
        fig_scatter = px.scatter(
            filtered_df, x='SessionsPerWeek', y='AvgSessionDurationMinutes', color='EngagementLevel',
            title="주간 세션 수 vs 평균 세션 시간",
            labels={'SessionsPerWeek': '주간 세션 수', 'AvgSessionDurationMinutes': '평균 세션 시간 (분)'},
            opacity=0.6, size='PlayTimeHours', hover_data=['Age', 'Gender', 'GameGenre'],
            category_orders={"EngagementLevel": ['Low', 'Medium', 'High']},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown(
            "**인사이트:** 플레이 시간이나 구매력 등 개별 행동 지표가 인게이지먼트 레벨과 밀접한 관계가 있음을 보여주며, 이는 전반적인 유저 참여 전략 수립에 중요합니다."
        )

    # Tab 4: 참여율 증진 요인 분석 (난이도, 장르 추가)
    with tab4:
        st.header("🚀 참여율 증진 요인 분석: 무엇이 유저 참여를 높이는가?")
        st.markdown("사용자의 참여 수준('Low' -> 'High')에 영향을 미치는 주요 요인들을 분석하여, 리텐션 및 몰입 증진 전략의 기반을 마련합니다.")
        
        # 1. 플레이어 레벨 vs 참여율 (유지)
        st.subheader("1. 플레이어 레벨 (PlayerLevel)별 참여 수준 분포")
        
        fig_level = px.box(
            filtered_df, 
            x='EngagementLevel', 
            y='PlayerLevel', 
            color='EngagementLevel',
            category_orders={"EngagementLevel": ['Low', 'Medium', 'High']},
            title="참여 수준별 플레이어 레벨 분포",
            color_discrete_map={'Low': '#EF553B', 'Medium': '#FFC400', 'High': '#636EFA'},
            labels={'PlayerLevel': '플레이어 레벨'}
        )
        st.plotly_chart(fig_level, use_container_width=True)
        st.markdown(
            "**인사이트:** 'High' 유저의 레벨 중앙값이 'Low' 유저보다 현저히 높다면, **레벨업 인센티브**와 **초기 성장 구간** 관리를 통해 신규 유저의 이탈을 방지하고 참여를 유도해야 합니다."
        )

        # 2. 업적 달성 vs 참여율 (유지)
        st.subheader("2. 잠금 해제된 업적 수 (AchievementsUnlocked) vs 참여율")
        fig_achievements = px.violin(
            filtered_df, 
            x='EngagementLevel', 
            y='AchievementsUnlocked', 
            color='EngagementLevel',
            category_orders={"EngagementLevel": ['Low', 'Medium', 'High']},
            title="참여 수준별 업적 달성 분포",
            color_discrete_map={'Low': '#EF553B', 'Medium': '#FFC400', 'High': '#636EFA'},
            box=True,
            points="all",
            labels={'AchievementsUnlocked': '잠금 해제된 업적 수'}
        )
        st.plotly_chart(fig_achievements, use_container_width=True)
        st.markdown(
            "**인사이트:** 업적 달성 수가 참여 수준과 강한 상관관계를 보인다면, **참여 유도형 업적 시스템**을 신규/복귀 유저에게 집중적으로 노출하여 지속적인 목표를 제공해야 합니다."
        )

        # 3. 게임 난이도 vs 인게이지먼트 (Tab 3에서 이동)
        st.subheader("3. 게임 난이도 (GameDifficulty)별 참여 수준 분포 (%)")
        difficulty_engagement = pd.crosstab(
            filtered_df['GameDifficulty'], filtered_df['EngagementLevel'], normalize='index'
        ) * 100
        
        fig_difficulty = px.bar(
            difficulty_engagement, barmode='group', title="게임 난이도별 인게이지먼트 분포 (%)",
            labels={'value': '비율 (%)', 'GameDifficulty': '게임 난이도'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_difficulty, use_container_width=True)
        st.markdown(
            "**인사이트:** 특정 난이도('Hard' 또는 'Easy')에서 'High' 인게이지먼트 유저의 비율이 높다면, **해당 난이도 선호 그룹**에 맞는 맞춤형 콘텐츠 업데이트가 효과적일 수 있습니다. 'Medium'이 가장 분산되어 있다면, 난이도 세분화가 필요합니다."
        )

        # 4. 게임 장르 vs 인게이지먼트 (추가 요청)
        st.subheader("4. 게임 장르 (GameGenre)별 참여 수준 분포 (%)")
        genre_engagement = pd.crosstab(
            filtered_df['GameGenre'], filtered_df['EngagementLevel'], normalize='index'
        ) * 100

        fig_genre_engagement = px.bar(
            genre_engagement, barmode='group', title="게임 장르별 인게이지먼트 분포 (%)",
            labels={'value': '비율 (%)', 'GameGenre': '게임 장르'},
            color_discrete_sequence=px.colors.qualitative.G10
        )
        st.plotly_chart(fig_genre_engagement, use_container_width=True)
        st.markdown(
            "**인사이트:** 'High' 인게이지먼트 유저가 특정 장르(예: 'RPG')에 집중되어 있다면, **해당 장르의 핵심 매커니즘**을 다른 장르에 부분적으로 차용하거나, 해당 장르 유저를 위한 **크로스 프로모션**을 기획하는 것이 참여율 증진에 도움이 될 수 있습니다."
        )


    # Tab 5: 사용자 이탈 예측 모델 (로지스틱 회귀 보고서 추가, 특성 중요도 제거)
    with tab5:
        st.header("🚫 사용자 이탈 예측 모델 (User Churn Prediction)")
        st.markdown("저관여 유저(Engagement Level = 'Low')를 **이탈 위험 사용자(Churn=1)**로 정의하고, 로지스틱 회귀 모델을 통해 이탈 가능성을 예측합니다. 이를 통해 선제적인 리텐션 대상자를 파악할 수 있습니다.")
        st.image("https://images.unsplash.com/photo-1542838749-43486162c938?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w0NTIyMjh8MHwxfHNlYXJjaHwxfHxDaHVybiUyMFByZWRpY3Rpb24lMjBtb2RlbCUyMHdvcmtmbG93fGVufDB8fHx8MTcwOTk2MTIwMHww&ixlib=rb-4.0.3&q=80&w=1080", 
                 caption="이탈 예측 모델 워크플로우 예시", 
                 use_column_width=True)

        # 1. 데이터 준비: 'Low'를 이탈(1), 나머지를 활동(0)으로 정의
        if filtered_df.empty:
            st.warning("필터링된 데이터가 없어 모델 학습을 진행할 수 없습니다.")
            st.stop()
            
        # .loc를 사용하여 SettingWithCopyWarning 방지
        filtered_df.loc[:, 'Churn'] = filtered_df['EngagementLevel'].apply(lambda x: 1 if x == 'Low' else 0)
        
        features = [
            'Age', 'Gender', 'Location', 'GameGenre', 'PlayTimeHours', 
            'InGamePurchases', 'GameDifficulty', 'SessionsPerWeek', 
            'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked'
        ]
        target = 'Churn'
        
        df_model = filtered_df[features + [target]].copy()
        
        # 결측치 처리
        nan_count_before = df_model.isnull().sum().sum()
        if nan_count_before > 0:
            st.info(f"데이터에서 총 {nan_count_before}개의 결측치(NaN)가 발견되어 모델 학습 전에 해당 행을 제거합니다.")
            df_model.dropna(inplace=True)
        
        if df_model.empty:
            st.warning("데이터 클리닝 후 남은 데이터가 없어 모델 학습을 진행할 수 없습니다.")
            st.stop()
            
        X = df_model[features]
        y = df_model[target]
        
        # 2. 전처리 파이프라인 구축
        numeric_features = ['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
        categorical_features = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'InGamePurchases'] # InGamePurchases는 0/1이지만 OHE로 처리하여 Exp(B) 해석을 단순화
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='drop' # 나머지 컬럼은 버림
        )

        # 3. 모델 정의 및 학습
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(solver='liblinear', random_state=42))])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        st.subheader("모델 학습 및 성능 평가 (Logistic Regression)")
        
        try:
            with st.spinner('모델 학습 및 평가 중...'):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 성능 지표
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, target_names=['Active (0)', 'Churn (1)'], output_dict=True)
                conf_mat = confusion_matrix(y_test, y_pred)

            st.success("✅ 모델 학습 완료!")

            col_acc, col_rep = st.columns([1, 2])
            
            with col_acc:
                st.subheader("예측 정확도")
                st.metric(label="모델 정확도 (Accuracy)", value=f"{accuracy:.4f}")
                
                st.subheader("이탈 비율 (Test Set)")
                st.info(f"실제 이탈 비율: {y_test.sum() / len(y_test) * 100:.2f}%")

                st.subheader("혼동 행렬")
                conf_df = pd.DataFrame(conf_mat, 
                                        index=['실제 Active (0)', '실제 Churn (1)'], 
                                        columns=['예측 Active (0)', '예측 Churn (1)'])
                st.dataframe(conf_df)

            with col_rep:
                st.subheader("분류 보고서")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.iloc[:-1, :].style.format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}))
                st.markdown(f"""
                - **정밀도 (Churn=1):** 모델이 이탈이라고 예측한 사용자 중 실제로 이탈한 비율
                - **재현율 (Churn=1):** 실제 이탈 사용자 중 모델이 정확히 이탈이라고 예측한 비율 (이탈 사용자 선별 능력)
                """)

            st.markdown("---")
            # --- 로지스틱 회귀분석 해석 보고서 (Exp(B)) ---
            st.subheader("📊 이탈 위험 요인 상세 분석 보고서 (로지스틱 회귀 계수 Exp(B) 기반)")
            
            # 특성 중요도 추출 (로지스틱 회귀 계수 사용)
            classifier = model.named_steps['classifier']
            
            # 전체 특성 이름 가져오기
            try:
                feature_names = numeric_features + list(model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
            except AttributeError:
                 feature_names = [] # 오류 방지용 임시 처리
            
            if len(feature_names) == len(classifier.coef_[0]):
                coefficients = pd.Series(classifier.coef_[0], index=feature_names)
                
                # Exp(B) (오즈비) 계산
                df_results = pd.DataFrame({
                    '변수': feature_names,
                    'B (회귀 계수)': classifier.coef_[0],
                    'Exp(B) (오즈비)': np.exp(classifier.coef_[0])
                })

                # 해석 필드 생성
                df_results['요인 유형'] = df_results['Exp(B) (오즈비)'].apply(lambda x: '잔존 요인 (보호)' if x < 1.0 else '이탈 위험 요인')
                df_results['오즈 변화율 (%)'] = df_results['Exp(B) (오즈비)'].apply(lambda x: f"{abs(round((x - 1) * 100, 1)):.1f}% {'감소' if x < 1.0 else '증가'}")
                
                # 유의미한 변수 (회귀 계수의 절대값이 큰 상위 20개만 표시)
                # Sig. 값을 알 수 없으므로, 절대값으로 판단하고 중요도 순으로 정렬합니다.
                df_results['abs_B'] = df_results['B (회귀 계수)'].abs()
                df_results = df_results.sort_values(by='abs_B', ascending=False).head(20).drop(columns=['abs_B'])

                # 시각화를 위한 필터링 및 컬럼 순서 조정
                df_interpretation = df_results[['변수', 'B (회귀 계수)', 'Exp(B) (오즈비)', '요인 유형', '오즈 변화율 (%)']].copy()
                
                # 스타일링 함수 정의 (잔존/위험 요인에 따라 색상 부여)
                def highlight_factor(s):
                    is_risk = s['요인 유형'] == '이탈 위험 요인'
                    is_retention = s['요인 유형'] == '잔존 요인 (보호)'
                    color = ''
                    if is_risk:
                        color = 'background-color: #ffe8e8; font-weight: bold; color: #cc0000' # 연한 빨강
                    elif is_retention:
                        color = 'background-color: #e8ffe8; font-weight: bold; color: #008000' # 연한 초록
                    
                    return [color if col in ['변수', '요인 유형', '오즈 변화율 (%)'] else '' for col in df_interpretation.columns]

                st.dataframe(
                    df_interpretation.style.apply(highlight_factor, axis=1).format({'B (회귀 계수)': "{:.4f}", 'Exp(B) (오즈비)': "{:.3f}"}),
                    height=400,
                    use_container_width=True
                )
                
                st.markdown("""
                ### 💡 Exp(B) (오즈비) 해석 원칙
                이탈을 종속변수(1)로 정의했습니다.
                - **Exp(B) < 1.0 (잔존 요인):** 변수 값이 증가할 때, **이탈할 오즈(확률)가 감소**합니다. (고객 잔존에 기여)
                - **Exp(B) > 1.0 (이탈 위험 요인):** 변수 값이 증가할 때, **이탈할 오즈(확률)가 증가**합니다. (고객 이탈 위험 증가)
                - **변수 해석 주의:** 원핫인코딩된 변수(예: GameGenre_Strategy)는 기준 장르 대비 해당 장르를 선택했을 때의 이탈 오즈 변화를 의미합니다.
                """)
            else:
                st.error("특성 이름과 계수 개수가 일치하지 않아 로지스틱 회귀 계수를 분석할 수 없습니다.")
            # --- 로지스틱 회귀분석 해석 보고서 끝 ---


        except Exception as e:
            st.error(f"모델 학습 및 예측 중 오류 발생: {e}")

    # 데이터 다운로드 (나머지 부분은 변경 없음)
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
