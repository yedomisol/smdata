# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ------------------------------------------------------------
# 1. 텍스트 전처리
# ------------------------------------------------------------
def tags_to_text(tag):
    if pd.isna(tag):
        return ""
    s = " ".join(map(str, tag)) if isinstance(tag, list) else str(tag)
    s = "" if str(s).strip() == "없음" else s
    s = s.replace(",", " ")
    s = " ".join(s.lower().split())
    return s

# ------------------------------------------------------------
# 2. TF-IDF 행렬 생성
# ------------------------------------------------------------
def build_tfidf_matrix(program_df, tag_col="tag",
                       min_df=3, ngram_range=(2, 4)):
    """프로그램 단위 TF-IDF 행렬"""
    corpus = program_df[tag_col].apply(tags_to_text)
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=ngram_range, min_df=min_df
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

# ------------------------------------------------------------
# 3. 밑작업
# ------------------------------------------------------------
CORE_NAME_MAP = {
    "core_1": "자기주도성",
    "core_2": "성장지향성",
    "core_3": "창의성",
    "core_4": "문제 해결적 대처",
    "core_5": "공감적이해",
    "core_6": "윤리적 의사소통",
    "core_7": "국제감각",
    "core_8": "공동체 의식"
}

# 가중치 계산
def recency_weight(dates, ref, half_life_days=180):
    dd = (ref - pd.to_datetime(dates, errors="coerce")).dt.days.fillna(365)
    return np.exp(-np.log(2) * (dd / half_life_days))

# ------------------------------------------------------------
# 4. 학생 벡터 생성 (가중 평균)
# ------------------------------------------------------------
def build_student_vector(student_df, program_df, X,
                         program_id_col="program_id", ref_date=None):
    if ref_date is None:
        ref_date = pd.Timestamp.today().normalize()

    # 기준일 이전 이력만 사용
    tmp = student_df.copy()
    tmp["start"] = pd.to_datetime(tmp.get("start"), errors="coerce")
    tmp["end"]   = pd.to_datetime(tmp.get("end"),   errors="coerce")
    end_eff = tmp["end"].fillna(tmp["start"])
    tmp = tmp[end_eff <= ref_date]
    if tmp.empty:
        return np.zeros((1, X.shape[1]))

    # 프로그램 매칭(정렬 맞추기 위해 program_df 서브셋 인덱스와 merge)
    sel = program_df[[program_id_col]].reset_index().rename(columns={"index":"prog_idx"})
    tmp = tmp.merge(sel, on=program_id_col, how="inner")
    if tmp.empty:
        return np.zeros((1, X.shape[1]))

    # 이수여부, 최근성 가중
    w_finish = tmp["completed"].astype(str).str.contains("이수", na=False).astype(float)
    w_rec = recency_weight(tmp.get("end", ref_date), ref=ref_date)
    w = 0.3*w_finish + 0.7*w_rec

    # X에서 해당 프로그램 행만 뽑아 동일 순서로 가중 평균
    rows = tmp["prog_idx"].to_list()
    X_sel = X[rows].toarray()
    vec = np.average(X_sel, axis=0, weights=w.values)
    return vec.reshape(1, -1)

# ------------------------------------------------------------
# 5. 추천 함수
# ------------------------------------------------------------
def recommend_tfidf(
    df, student_id, top_n=10, candidate_k=100,
    upcoming_days=90, ref_date=None,
    duration_filter=None, type_filter=None,
    core_type=None
):
    # 고정 출력 컬럼
    KEEP_COLS = [
    "program_id","program_name","reason","type","core","method","start","duration_cat",
    "score","content_sim","tag"
    ]

    # 기준일
    ref_date = pd.Timestamp.today().normalize() if ref_date is None else pd.to_datetime(ref_date).normalize()

    ## 학생의 학년 추출 (없으면 None) ##
    stu_row = df.loc[df["student_id"].eq(student_id)].head(1)
    stu_grade = stu_row["grade"].iloc[0] if "grade" in df.columns and not stu_row.empty else None

    # 프로그램 메타(중복 제거)
    base_cols = ["program_id","program_name","tag","start","end","type","method",
                 "core_1","core_2","core_3","core_4","core_5","core_6","core_7","core_8"]
    program_df = (df.sort_values("start")
                    .drop_duplicates("program_id")
                    .loc[:, [c for c in base_cols if c in df.columns]]
                    .reset_index(drop=True))
    program_df["start"] = pd.to_datetime(program_df["start"], errors="coerce")
    program_df["end"]   = pd.to_datetime(program_df["end"],   errors="coerce")

    # 기간 계산 + 카테고리
    end_eff_prog = program_df["end"].fillna(program_df["start"])
    program_df["duration_days"] = (end_eff_prog - program_df["start"]).dt.days.add(1).clip(lower=1)
    program_df["duration_cat"] = np.select(
        [
            program_df["duration_days"].eq(1),
            program_df["duration_days"].between(2, 29, inclusive="both"),
            program_df["duration_days"].ge(30)
        ],
        ["일회성","단기","장기"],
        default="단기"
    )

    # TF-IDF
    program_df["tag"] = program_df["tag"].astype(str)
    vectorizer, X = build_tfidf_matrix(program_df, tag_col="tag")

    # 학생 로그(기준일 이전만)
    stu_logs = df.loc[df["student_id"].eq(student_id),
                      ["student_id","program_id","start","end","completed","satisfy"] +
                      (["grade"] if ("grade" in df.columns) else [])
                     ].copy()

    # 로그가 비어있진 않은지 확인
    if stu_logs.empty:
        return pd.DataFrame(columns=KEEP_COLS)

    # 기준일 이전 로그만 남기기
    stu_logs["start"] = pd.to_datetime(stu_logs["start"], errors="coerce")
    stu_logs["end"]   = pd.to_datetime(stu_logs["end"],   errors="coerce")
    end_eff_logs = stu_logs["end"].fillna(stu_logs["start"])
    stu_logs = stu_logs[end_eff_logs <= ref_date]
    if stu_logs.empty:
        return pd.DataFrame(columns=KEEP_COLS)

    # 학생 벡터 정의
    stu_vec = build_student_vector(stu_logs, program_df, X, ref_date=ref_date)

    # 벡터가 비어있진 않은지 확인
    if np.linalg.norm(stu_vec) < 1e-12:
        return pd.DataFrame(columns=KEEP_COLS)

    # 콘텐츠 유사도
    sim = cosine_similarity(normalize(stu_vec), normalize(X)).ravel()
    program_df["content_sim"] = sim

    # 필터1: 기간 + 이미 수강 제외
    horizon = ref_date + pd.Timedelta(days=upcoming_days)
    cand = program_df[(program_df["start"] >= ref_date) &
                  (program_df["start"] <= horizon)].copy()
    taken = set(stu_logs["program_id"])
    cand = cand[~cand["program_id"].isin(taken)]

    # 필터2: 2~4학년은 '신입생'/'새내기' 태그 포함 프로그램 제외
    if stu_grade in [2, 3, 4]:
        tag_col = None
        for c in ["tag", "tags", "태그", "program_tag"]:
            if c in cand.columns:
                tag_col = c
                break
        if tag_col:
            cand = cand[~cand[tag_col].astype(str).str.contains("신입생|새내기", na=False)]

    # 필터3: 기간 필터 적용 (선택 시에만)
    if duration_filter is not None:
        dlist = [duration_filter] if isinstance(duration_filter, str) else list(duration_filter)
        dlist = [d for d in dlist if d in {"일회성","단기","장기"}]
        if dlist:
            cand = cand[cand["duration_cat"].isin(dlist)]

    # 필터4: 타입 필터 적용 (선택 시에만)
    if type_filter is not None:
        allowed_types = ["학습","심리","취업","학생활동","진로","창업"]
        tlist = [type_filter] if isinstance(type_filter, str) else list(type_filter)
        tlist = [t for t in tlist if t in allowed_types]
        if tlist:
            cand = cand[cand["type"].isin(tlist)]

    # 필터5: 핵심역량 포함 필터 (선택 시에만)
    if core_type is not None:
        def _as_core_cols(x):
            xs = [x] if isinstance(x, (int, str)) else list(x)
            out = []
            for v in xs:
                if isinstance(v, int):
                    out.append(f"core_{v}")
                else:
                    s = str(v).strip()
                    if s.startswith("core_"):
                        out.append(s)
                    else:
                        for k, name in CORE_NAME_MAP.items():
                            if name == s:
                                out.append(k)
            return [c for c in out if c in cand.columns]

        want_cols = _as_core_cols(core_type)
        if want_cols:
            cand = cand[cand[want_cols].fillna(0).sum(axis=1) > 0]

    if cand.empty:
        return pd.DataFrame(columns=KEEP_COLS)

    # Top-k
    cand = cand.sort_values("content_sim", ascending=False).head(candidate_k).copy()

    # 추천 사유(reason)
    id_to_idx = {pid: i for i, pid in enumerate(program_df["program_id"])}
    taken_ids = [pid for pid in taken if pid in id_to_idx]
    cand_ids  = [pid for pid in cand["program_id"] if pid in id_to_idx]
    if taken_ids and cand_ids:
        taken_idx = [id_to_idx[pid] for pid in taken_ids]
        cand_idx  = [id_to_idx[pid] for pid in cand_ids]
        X_taken = normalize(X[taken_idx])
        X_cand  = normalize(X[cand_idx])
        sim_mat = cosine_similarity(X_cand, X_taken)
        best_pos = sim_mat.argmax(axis=1)
        taken_names_ordered = program_df.loc[taken_idx, "program_name"].tolist()
        best_names = [taken_names_ordered[j] if 0 <= j < len(taken_names_ordered) else None
                      for j in best_pos]
        reason_series = pd.Series(
            [f"이전에 들었던 '{nm}'과(와) 비슷한 프로그램이에요." if isinstance(nm, str) and len(nm) > 0
             else "이전에 수강한 프로그램들과의 내용 유사도가 높아요."
             for nm in best_names],
            index=program_df.loc[cand_idx, "program_id"].values
        )
        cand = cand.merge(reason_series.rename("reason"),
                          left_on="program_id", right_index=True, how="left")
    else:
        cand["reason"] = "송이에게 추천하는 프로그램이에요."

    # 해당 프로그램이 가진 핵심역량 소개 컬럼 추가
    core_cols_present = [c for c in CORE_NAME_MAP if c in cand.columns]

    if core_cols_present:
        def summarize_core_names(row, eps=1e-12):
            names = [CORE_NAME_MAP[c] for c in core_cols_present
                    if pd.notna(row.get(c)) and float(row.get(c)) > eps]
            return ", ".join(names)
        cand["core"] = cand.apply(summarize_core_names, axis=1)
    else:
        cand["core"] = ""

    return cand.sort_values("content_sim", ascending=False).head(top_n)

import warnings
warnings.filterwarnings("ignore", message="Could not infer format")

"""**실행 예시 : 2025년 1월 1일 기준으로 이후 90일 내 프로그램 추천**"""

df = pd.read_csv('final_data.csv')

# ------------------------------------------------------------
# 5. 실행 예시
# ------------------------------------------------------------
# 핵심역량 + 코호트(같은 단과대학) 적용
rec = recommend_tfidf(
    df, student_id='998742-6376898',
    top_n=10, candidate_k=100,
    upcoming_days=60, ref_date='2025-01-01',
    duration_filter=None, type_filter=None,
    core_type=None
)

rec
