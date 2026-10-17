from __future__ import annotations

"""
Aimers 9기 투구 제구 성공 확률 예측용 CatBoost 실험 파이프라인

핵심 원칙
1) train/valid는 season-game_month 기준 expanding window로 분리한다.
2) game_type 필터는 학습 행에만 적용하고, validation 행은 모든 실험에서 동일하게 유지한다.
3) test.csv 내부 다른 행을 이용한 집계/rolling/count encoding은 만들지 않는다.
4) 투수×카운트, 투수×좌우매치업 등은 문자열 범주형으로 만들고 CatBoost에 직접 전달한다.
5) Bayesian smoothing prior는 각 fold의 학습 데이터로만 적합한다.
6) calibration은 base model의 OOF 예측값으로만 적합한다.

필요 패키지
    pip install pandas numpy scikit-learn catboost matplotlib joblib
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
import gc
import json
import math
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


TARGET = "control_success"
MISSING_CAT = "__MISSING__"
EPS = 1e-6


# -----------------------------------------------------------------------------
# 1. 평가 지표
# -----------------------------------------------------------------------------
def dacon_bss_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> dict[str, float]:
    """DACON 공식 산식과 같은 Brier Skill Score 계열 지표를 계산한다."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)

    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true와 y_prob는 길이가 같은 1차원 배열이어야 합니다.")
    if len(y) == 0:
        raise ValueError("평가 표본이 비어 있습니다.")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("y_true는 0/1이어야 합니다.")

    brier = float(np.mean((p - y) ** 2))
    prevalence = float(np.mean(y))
    reference_brier = prevalence * (1.0 - prevalence)

    if reference_brier <= 0:
        raw_bss = float("nan")
        official_score = float("nan")
    else:
        raw_bss = 1.0 - brier / reference_brier
        official_score = max(0.0, 100000.0 * raw_bss)

    return {
        "n": float(len(y)),
        "positive_rate": prevalence,
        "brier": brier,
        "reference_brier": reference_brier,
        "raw_bss": raw_bss,
        "official_score": official_score,
    }


# -----------------------------------------------------------------------------
# 2. Feature configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    count_state: bool = False
    categorical_interactions: bool = False
    recent_form: bool = False
    n_features: bool = False
    bayesian_smoothing: bool = False
    pitcher_batter_interactions: bool = False
    pitchmix_diversity: bool = False
    game_context: bool = False


BASELINE_CONFIG = FeatureConfig()
COUNT_CONFIG = FeatureConfig(count_state=True)
PITCHER_CONTEXT_CONFIG = FeatureConfig(
    count_state=True,
    categorical_interactions=True,
)
FORM_CONFIG = FeatureConfig(
    count_state=True,
    categorical_interactions=True,
    recent_form=True,
)
SMOOTH_CONFIG = FeatureConfig(
    count_state=True,
    categorical_interactions=True,
    recent_form=True,
    n_features=True,
    bayesian_smoothing=True,
)
INTERACTION_CONFIG = FeatureConfig(
    count_state=True,
    categorical_interactions=True,
    recent_form=True,
    n_features=True,
    bayesian_smoothing=True,
    pitcher_batter_interactions=True,
)
ALL_FEATURE_CONFIG = FeatureConfig(
    count_state=True,
    categorical_interactions=True,
    recent_form=True,
    n_features=True,
    bayesian_smoothing=True,
    pitcher_batter_interactions=True,
    pitchmix_diversity=True,
    game_context=True,
)

FEATURE_EXPERIMENTS: dict[str, FeatureConfig] = {
    "00_native_catboost": BASELINE_CONFIG,
    "01_add_count_state": COUNT_CONFIG,
    "02_add_pitcher_context": PITCHER_CONTEXT_CONFIG,
    "03_add_recent_form": FORM_CONFIG,
    "04_add_n_and_smoothing": SMOOTH_CONFIG,
    "05_add_pitcher_batter": INTERACTION_CONFIG,
    "06_add_pitchmix_and_context": ALL_FEATURE_CONFIG,
}


# CatBoost에서 범주형으로 처리할 원본/파생 변수 후보
BASE_CATEGORICAL_COLUMNS = {
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "game_dayofweek",
    "season_cat",
    "game_month_cat",
    "inning_cat",
    "outs_before_cat",
    "count_state",
    "count_advantage",
    "count_pressure_state",
    "hand_matchup",
    "pitcher_count",
    "pitcher_hand_matchup",
    "pitcher_count_hand_matchup",
    "pitcher_batter",
    "pitcher_batter_hand_matchup",
    "pressure_state",
}


# -----------------------------------------------------------------------------
# 3. 안전한 row-wise feature engineering
# -----------------------------------------------------------------------------
def _require_columns(frame: pd.DataFrame, columns: Iterable[str], block_name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{block_name} 생성에 필요한 컬럼이 없습니다: {missing}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _cat_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_CAT).astype(str)


def _combine_categories(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    _require_columns(frame, columns, "범주형 interaction")
    parts = [_cat_string(frame[c]) for c in columns]
    out = parts[0]
    for part in parts[1:]:
        out = out.str.cat(part, sep="__")
    return out


def add_rowwise_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """
    한 행 내부 정보만 사용하는 파생변수 생성.

    이 함수는 다른 행의 평균/빈도/순위/rolling을 계산하지 않으므로 test 행 독립성 원칙에 맞는다.
    """
    out = frame.copy()

    # 시간/상황 변수의 범주형 복사본: 원래 숫자형 변수도 동시에 유지한다.
    for source, target in [
        ("season", "season_cat"),
        ("game_month", "game_month_cat"),
        ("inning", "inning_cat"),
        ("outs_before", "outs_before_cat"),
    ]:
        if source in out.columns:
            out[target] = _cat_string(out[source])

    if config.count_state or config.categorical_interactions:
        _require_columns(out, ["balls_before", "strikes_before"], "카운트 변수")
        balls = _numeric(out["balls_before"])
        strikes = _numeric(out["strikes_before"])

        out["count_state"] = _cat_string(out["balls_before"]).str.cat(
            _cat_string(out["strikes_before"]), sep="_"
        )
        out["two_strike"] = (strikes == 2).astype("int8")
        out["three_ball"] = (balls == 3).astype("int8")
        out["full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
        out["pitcher_ahead"] = (strikes > balls).astype("int8")
        out["batter_ahead"] = (balls > strikes).astype("int8")
        out["even_count"] = (balls == strikes).astype("int8")

        advantage = np.select(
            [strikes > balls, balls > strikes],
            ["PITCHER_AHEAD", "BATTER_AHEAD"],
            default="EVEN",
        )
        out["count_advantage"] = pd.Series(advantage, index=out.index, dtype="object")
        out["count_pressure_state"] = (
            out["two_strike"].astype(str)
            + "__"
            + out["three_ball"].astype(str)
            + "__"
            + out["full_count"].astype(str)
        )

    if config.categorical_interactions:
        _require_columns(
            out,
            ["pitcher_id", "pitcher_hand", "batter_hand", "count_state"],
            "투수-카운트-좌우 interaction",
        )
        out["hand_matchup"] = _combine_categories(out, ["pitcher_hand", "batter_hand"])
        out["pitcher_count"] = _combine_categories(out, ["pitcher_id", "count_state"])
        out["pitcher_hand_matchup"] = _combine_categories(
            out, ["pitcher_id", "hand_matchup"]
        )
        out["pitcher_count_hand_matchup"] = _combine_categories(
            out, ["pitcher_id", "count_state", "hand_matchup"]
        )

    if config.recent_form:
        for metric in ["success_rate", "middle_rate"]:
            c1 = f"asof_pitcher_prev1_game_{metric}"
            c3 = f"asof_pitcher_prev3_game_{metric}"
            c5 = f"asof_pitcher_prev5_game_{metric}"
            career = f"asof_pitcher_{metric}"
            _require_columns(out, [c1, c3, c5, career], f"최근 {metric} form")

            v1 = _numeric(out[c1])
            v3 = _numeric(out[c3])
            v5 = _numeric(out[c5])
            vc = _numeric(out[career])

            out[f"pitcher_{metric}_form_1m5"] = v1 - v5
            out[f"pitcher_{metric}_form_3m5"] = v3 - v5
            out[f"pitcher_{metric}_form_1m3"] = v1 - v3
            out[f"pitcher_{metric}_form_1mcareer"] = v1 - vc
            out[f"pitcher_{metric}_form_3mcareer"] = v3 - vc

    if config.n_features or config.bayesian_smoothing:
        n_columns = [
            "asof_pitcher_n",
            "asof_batter_n",
            "asof_pitcher_pitchmix_n",
            "p_n_tot",
            "b_n_tot",
        ]
        for col in n_columns:
            if col in out.columns:
                n = _numeric(out[col]).clip(lower=0)
                out[f"log1p_{col}"] = np.log1p(n)
                out[f"sqrt_{col}"] = np.sqrt(n)

    if config.pitcher_batter_interactions:
        _require_columns(out, ["pitcher_id", "batter_id"], "투수-타자 interaction")
        out["pitcher_batter"] = _combine_categories(out, ["pitcher_id", "batter_id"])
        if "hand_matchup" not in out.columns:
            _require_columns(out, ["pitcher_hand", "batter_hand"], "좌우매치업")
            out["hand_matchup"] = _combine_categories(out, ["pitcher_hand", "batter_hand"])
        out["pitcher_batter_hand_matchup"] = _combine_categories(
            out, ["pitcher_id", "batter_id", "hand_matchup"]
        )

        numeric_pairs = [
            ("asof_pitcher_success_rate", "asof_batter_success_rate", "success"),
            ("asof_pitcher_middle_rate", "asof_batter_middle_rate", "middle"),
        ]
        for p_col, b_col, name in numeric_pairs:
            if p_col in out.columns and b_col in out.columns:
                p = _numeric(out[p_col])
                b = _numeric(out[b_col])
                out[f"pitcher_minus_batter_{name}"] = p - b
                out[f"pitcher_x_batter_{name}"] = p * b
                out[f"pitcher_div_batter_{name}"] = p / b.clip(lower=EPS)

    if config.pitchmix_diversity:
        pitch_cols = [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
        _require_columns(out, pitch_cols, "pitchmix 다양성")
        mix = out[pitch_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0)
        mix_sum = mix.sum(axis=1, min_count=1).replace(0.0, np.nan)
        prob = mix.div(mix_sum, axis=0)
        positive_prob = prob.where(prob > 0)

        entropy = -(positive_prob * np.log(positive_prob)).sum(axis=1, min_count=1)
        out["pitchmix_entropy"] = entropy / math.log(len(pitch_cols))
        out["pitchmix_hhi"] = (prob**2).sum(axis=1, min_count=1)
        out["pitchmix_dominant_rate"] = prob.max(axis=1)
        out["pitchmix_spread"] = prob.max(axis=1) - prob.min(axis=1)
        out["fastball_minus_breaking"] = prob[pitch_cols[0]] - prob[pitch_cols[1]]

    if config.game_context:
        needed = [
            "runner_on_2b",
            "runner_on_3b",
            "runner_on_1b",
            "inning",
            "score_diff_pitcher_team",
            "li",
        ]
        _require_columns(out, needed, "경기 압박 상황")
        r1 = _numeric(out["runner_on_1b"])
        r2 = _numeric(out["runner_on_2b"])
        r3 = _numeric(out["runner_on_3b"])
        inning = _numeric(out["inning"])
        score_diff = _numeric(out["score_diff_pitcher_team"])
        li = _numeric(out["li"])

        out["risp"] = ((r2 == 1) | (r3 == 1)).astype("int8")
        out["bases_loaded"] = ((r1 == 1) & (r2 == 1) & (r3 == 1)).astype("int8")
        out["late_inning"] = (inning >= 7).astype("int8")
        out["close_game"] = (score_diff.abs() <= 1).astype("int8")
        out["high_leverage"] = (li >= 1.5).astype("int8")
        out["pressure_state"] = (
            out["late_inning"].astype(str)
            + "__"
            + out["risp"].astype(str)
            + "__"
            + out["close_game"].astype(str)
            + "__"
            + out["high_leverage"].astype(str)
        )

    return out


# -----------------------------------------------------------------------------
# 4. Bayesian smoothing: fold train으로 prior를 fit하고 valid/test는 transform만
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RateSpec:
    rate_col: str
    n_col: str
    prior_kind: Literal["target", "weighted_feature"]
    alpha: float


DEFAULT_RATE_SPECS: tuple[RateSpec, ...] = (
    RateSpec("asof_pitcher_success_rate", "asof_pitcher_n", "target", 100.0),
    RateSpec("asof_batter_success_rate", "asof_batter_n", "target", 100.0),
    RateSpec("asof_pitcher_reverse_rate", "asof_pitcher_n", "weighted_feature", 100.0),
    RateSpec("asof_pitcher_middle_rate", "asof_pitcher_n", "weighted_feature", 100.0),
    RateSpec("asof_pitcher_ball_rate", "asof_pitcher_n", "weighted_feature", 100.0),
    RateSpec("asof_pitcher_strike_rate", "asof_pitcher_n", "weighted_feature", 100.0),
    RateSpec("asof_batter_middle_rate", "asof_batter_n", "weighted_feature", 100.0),
    RateSpec(
        "asof_pitcher_fastball_rate",
        "asof_pitcher_pitchmix_n",
        "weighted_feature",
        50.0,
    ),
    RateSpec(
        "asof_pitcher_breaking_rate",
        "asof_pitcher_pitchmix_n",
        "weighted_feature",
        50.0,
    ),
    RateSpec(
        "asof_pitcher_offspeed_rate",
        "asof_pitcher_pitchmix_n",
        "weighted_feature",
        50.0,
    ),
)


class BayesianRateSmoother:
    """훈련 fold에서만 prior를 학습하는 empirical-Bayes형 rate smoother."""

    def __init__(self, specs: Sequence[RateSpec] = DEFAULT_RATE_SPECS, alpha_scale: float = 1.0):
        if alpha_scale <= 0:
            raise ValueError("alpha_scale은 양수여야 합니다.")
        self.specs = tuple(specs)
        self.alpha_scale = float(alpha_scale)
        self.priors_: dict[str, float] = {}
        self.fitted_specs_: list[RateSpec] = []

    def fit(self, frame: pd.DataFrame, y: Sequence[int]) -> "BayesianRateSmoother":
        y_arr = np.asarray(y, dtype=float)
        if len(frame) != len(y_arr):
            raise ValueError("BayesianRateSmoother.fit: X와 y 길이가 다릅니다.")
        if len(y_arr) == 0:
            raise ValueError("BayesianRateSmoother.fit: 학습 데이터가 비어 있습니다.")

        target_prior = float(np.mean(y_arr))
        self.priors_.clear()
        self.fitted_specs_.clear()

        for spec in self.specs:
            if spec.rate_col not in frame.columns or spec.n_col not in frame.columns:
                continue

            rate = _numeric(frame[spec.rate_col]).clip(0.0, 1.0)
            n = _numeric(frame[spec.n_col]).clip(lower=0.0)

            if spec.prior_kind == "target":
                prior = target_prior
            else:
                mask = rate.notna() & n.notna() & (n > 0)
                if mask.any() and float(n.loc[mask].sum()) > 0:
                    prior = float(np.average(rate.loc[mask], weights=n.loc[mask]))
                else:
                    fallback = rate.mean(skipna=True)
                    prior = target_prior if pd.isna(fallback) else float(fallback)

            prior = float(np.clip(prior, EPS, 1.0 - EPS))
            self.priors_[spec.rate_col] = prior
            self.fitted_specs_.append(spec)

        if not self.fitted_specs_:
            warnings.warn("Bayesian smoothing에 사용할 rate/n 컬럼 조합이 없습니다.")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "priors_"):
            raise RuntimeError("fit 후 transform을 호출해야 합니다.")

        out = frame.copy()
        for spec in self.fitted_specs_:
            prior = self.priors_[spec.rate_col]
            alpha = spec.alpha * self.alpha_scale

            rate = _numeric(out[spec.rate_col]).clip(0.0, 1.0).fillna(prior)
            n = _numeric(out[spec.n_col]).clip(lower=0.0).fillna(0.0)
            posterior = (n * rate + alpha * prior) / (n + alpha)

            out[f"{spec.rate_col}__eb"] = posterior
            out[f"{spec.rate_col}__prior_delta"] = rate - prior
            out[f"{spec.rate_col}__reliability"] = n / (n + alpha)
            out[f"{spec.rate_col}__posterior_sd"] = np.sqrt(
                posterior * (1.0 - posterior) / (n + alpha + 1.0)
            )

        return out

    def fit_transform(self, frame: pd.DataFrame, y: Sequence[int]) -> pd.DataFrame:
        return self.fit(frame, y).transform(frame)


# -----------------------------------------------------------------------------
# 5. CatBoost 입력 정리: one-hot 금지, 범주형을 문자열 그대로 전달
# -----------------------------------------------------------------------------
def prepare_catboost_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.drop(columns=[TARGET, "row_id"], errors="ignore").copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    cat_cols: list[str] = []
    for col in out.columns:
        dtype = out[col].dtype
        if (
            col in BASE_CATEGORICAL_COLUMNS
            or pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            cat_cols.append(col)

    for col in cat_cols:
        out[col] = _cat_string(out[col])

    numeric_cols = [c for c in out.columns if c not in cat_cols]
    for col in numeric_cols:
        if pd.api.types.is_bool_dtype(out[col].dtype):
            out[col] = out[col].astype("int8")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out, cat_cols


# -----------------------------------------------------------------------------
# 6. Expanding time CV
# -----------------------------------------------------------------------------
def build_month_period(frame: pd.DataFrame) -> pd.Series:
    """정확한 날짜가 없으므로 season × game_month를 시간 단위로 사용한다."""
    _require_columns(frame, ["season", "game_month"], "시간순 CV")
    season = _numeric(frame["season"])
    month = _numeric(frame["game_month"])

    if season.isna().any() or month.isna().any():
        raise ValueError("season 또는 game_month에 결측이 있어 시간순 fold를 만들 수 없습니다.")
    if not month.between(1, 12).all():
        raise ValueError("game_month는 1~12 범위여야 합니다.")

    return (season.astype(int) * 100 + month.astype(int)).rename("_period")


def make_expanding_month_folds(
    frame: pd.DataFrame,
    n_splits: int = 4,
    valid_months: int = 2,
    min_train_months: int = 12,
    gap_months: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits는 2 이상이어야 calibration까지 확인할 수 있습니다.")
    if valid_months < 1 or min_train_months < 1 or gap_months < 0:
        raise ValueError("valid_months/min_train_months는 양수, gap_months는 0 이상이어야 합니다.")

    period = build_month_period(frame)
    unique_periods = np.array(sorted(period.unique()))
    first_valid_pos = len(unique_periods) - n_splits * valid_months

    if first_valid_pos - gap_months < min_train_months:
        raise ValueError(
            "월 수가 부족합니다. "
            f"전체 월={len(unique_periods)}, 요구 최소={min_train_months + gap_months + n_splits * valid_months}. "
            "n_splits 또는 valid_months/min_train_months를 줄이세요."
        )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        valid_start = first_valid_pos + fold * valid_months
        valid_end = valid_start + valid_months
        train_end = valid_start - gap_months

        train_periods = unique_periods[:train_end]
        valid_periods = unique_periods[valid_start:valid_end]
        train_idx = np.flatnonzero(period.isin(train_periods).to_numpy())
        valid_idx = np.flatnonzero(period.isin(valid_periods).to_numpy())

        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise RuntimeError(f"fold={fold}의 train 또는 valid가 비었습니다.")
        if period.iloc[train_idx].max() >= period.iloc[valid_idx].min():
            raise RuntimeError("시간순 fold 생성 오류: train과 valid 시간 순서가 뒤섞였습니다.")
        folds.append((train_idx, valid_idx))

    return folds


# -----------------------------------------------------------------------------
# 7. game_type 학습 필터
# -----------------------------------------------------------------------------
TrainFilterMode = Literal["all", "r_only", "r_or_2023plus", "recent_2023plus"]


def training_filter_mask(frame: pd.DataFrame, mode: TrainFilterMode) -> pd.Series:
    if mode == "all":
        return pd.Series(True, index=frame.index)

    _require_columns(frame, ["game_type", "season"], "game_type 학습 필터")
    game_type = _cat_string(frame["game_type"]).str.upper()
    season = _numeric(frame["season"])

    if mode == "r_only":
        return game_type.eq("R")
    if mode == "r_or_2023plus":
        # 사용자가 쓰던 조건: 2023년 이전은 R만, 2023년 이후는 game_type과 무관하게 유지
        return game_type.eq("R") | season.ge(2023)
    if mode == "recent_2023plus":
        return season.ge(2023)
    raise ValueError(f"지원하지 않는 train filter mode: {mode}")


# -----------------------------------------------------------------------------
# 8. CV 학습
# -----------------------------------------------------------------------------
@dataclass
class CVResult:
    experiment_name: str
    filter_mode: str
    feature_config: FeatureConfig
    oof_prediction: np.ndarray
    fold_id: np.ndarray
    fold_metrics: pd.DataFrame
    overall_metrics: dict[str, float]
    best_iterations: list[int]

    @property
    def median_best_iteration(self) -> int:
        valid = [x for x in self.best_iterations if x > 0]
        return int(np.median(valid)) if valid else 1000


def default_catboost_params(
    task_type: Literal["CPU", "GPU"] = "GPU",
    random_seed: int = 42,
    iterations: int = 2000,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "BrierScore",
        "iterations": iterations,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        # interaction 문자열을 직접 만들었으므로 추가 조합을 제한해 메모리를 아낀다.
        "max_ctr_complexity": 1,
        "one_hot_max_size": 10,
        "random_seed": random_seed,
        "task_type": task_type,
        "allow_writing_files": False,
        "verbose": 200,
    }
    if task_type == "GPU":
        params["devices"] = "0"
    else:
        params["thread_count"] = -1
    return params



def run_season_cv(
    train: pd.DataFrame,
    feature_config: FeatureConfig,
    filter_mode: TrainFilterMode = "all",
    experiment_name: str = "experiment",
    valid_seasons: Sequence[int] = (2022, 2023, 2024),
    alpha_scale: float = 1.0,
    catboost_params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 150,
) -> CVResult:
    """
    시즌 walk-forward CV.

    예:
      valid=2022 -> train season < 2022, valid season == 2022
      valid=2023 -> train season < 2023, valid season == 2023
      valid=2024 -> train season < 2024, valid season == 2024

    validation 행에는 game_type 필터를 걸지 않는다. 각 validation 시즌 전체를 예측한 뒤
    ALL / R / F 성능을 동시에 기록한다. 학습 데이터에만 filter_mode를 적용한다.
    """
    if TARGET not in train.columns:
        raise KeyError(f"학습 파일에 target 컬럼 '{TARGET}'이 없습니다.")
    _require_columns(train, ["season", "game_type"], "시즌/리그 CV")

    y_all = pd.to_numeric(train[TARGET], errors="raise").astype(int).to_numpy()
    if not np.isin(y_all, [0, 1]).all():
        raise ValueError("control_success는 0/1이어야 합니다.")

    season_num = _numeric(train["season"])
    game_type_upper = _cat_string(train["game_type"]).str.upper()

    available_seasons = set(pd.to_numeric(train["season"], errors="coerce").dropna().astype(int).unique())
    requested = [int(s) for s in valid_seasons]
    missing = [s for s in requested if s not in available_seasons]
    if missing:
        raise ValueError(
            f"validation 시즌이 데이터에 없습니다: {missing}. "
            f"사용 가능 시즌={sorted(available_seasons)}"
        )

    raw_x = train.drop(columns=[TARGET])
    row_x = add_rowwise_features(raw_x, feature_config)
    params = default_catboost_params() if catboost_params is None else dict(catboost_params)

    oof = np.full(len(train), np.nan, dtype=float)
    fold_id = np.full(len(train), -1, dtype=int)
    fold_records: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    def prefixed_metrics(prefix: str, idx: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
        if len(idx) == 0:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_positive_rate": np.nan,
                f"{prefix}_prediction_mean": np.nan,
                f"{prefix}_prediction_std": np.nan,
                f"{prefix}_brier": np.nan,
                f"{prefix}_reference_brier": np.nan,
                f"{prefix}_raw_bss": np.nan,
                f"{prefix}_official_score": np.nan,
            }
        m = dacon_bss_metrics(y_all[idx], pred)
        return {f"{prefix}_{k}": v for k, v in m.items()}

    print(
        f"\n===== {experiment_name} | season walk-forward | "
        f"valid={requested} | filter={filter_mode} ====="
    )

    for fold, valid_season in enumerate(requested):
        train_idx_base = np.flatnonzero(season_num.lt(valid_season).to_numpy())
        valid_idx = np.flatnonzero(season_num.eq(valid_season).to_numpy())

        if len(train_idx_base) == 0 or len(valid_idx) == 0:
            raise ValueError(
                f"valid_season={valid_season}: train={len(train_idx_base)}, valid={len(valid_idx)}"
            )

        fold_train_meta = train.iloc[train_idx_base]
        keep_mask = training_filter_mask(fold_train_meta, filter_mode).to_numpy()
        train_idx = train_idx_base[keep_mask]

        if len(train_idx) < 1000:
            raise ValueError(
                f"valid_season={valid_season}: 필터 적용 후 학습 행이 "
                f"{len(train_idx):,}개뿐입니다. filter 설정을 확인하세요."
            )

        x_train_raw = row_x.iloc[train_idx].copy()
        x_valid_raw = row_x.iloc[valid_idx].copy()
        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]

        if feature_config.bayesian_smoothing:
            smoother = BayesianRateSmoother(alpha_scale=alpha_scale)
            x_train_raw = smoother.fit_transform(x_train_raw, y_train)
            x_valid_raw = smoother.transform(x_valid_raw)

        x_train, cat_cols = prepare_catboost_frame(x_train_raw)
        x_valid, valid_cat_cols = prepare_catboost_frame(x_valid_raw)
        x_valid = x_valid.reindex(columns=x_train.columns)

        if set(cat_cols) != set(valid_cat_cols):
            raise RuntimeError("train/valid 범주형 컬럼 구성이 다릅니다.")

        train_pool = Pool(x_train, label=y_train, cat_features=cat_cols)
        valid_pool = Pool(x_valid, label=y_valid, cat_features=cat_cols)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=params.get("verbose", 200),
        )

        pred = np.clip(model.predict_proba(valid_pool)[:, 1], EPS, 1.0 - EPS)
        oof[valid_idx] = pred
        fold_id[valid_idx] = fold

        metrics_all = dacon_bss_metrics(y_valid, pred)
        gt_valid = game_type_upper.iloc[valid_idx].to_numpy()
        local_r = np.flatnonzero(gt_valid == "R")
        local_f = np.flatnonzero(gt_valid == "F")
        idx_r = valid_idx[local_r]
        idx_f = valid_idx[local_f]
        pred_r = pred[local_r]
        pred_f = pred[local_f]

        best_iteration = model.get_best_iteration()
        best_iteration = (
            params.get("iterations", 2000)
            if best_iteration is None or best_iteration < 0
            else best_iteration + 1
        )
        best_iterations.append(int(best_iteration))

        record: dict[str, Any] = {
            "fold": fold,
            "valid_season": int(valid_season),
            "train_start_season": int(season_num.iloc[train_idx_base].min()),
            "train_end_season": int(season_num.iloc[train_idx_base].max()),
            "train_rows_before_filter": len(train_idx_base),
            "train_rows_after_filter": len(train_idx),
            "valid_rows": len(valid_idx),
            "valid_R_rows": len(idx_r),
            "valid_F_rows": len(idx_f),
            "best_iteration": int(best_iteration),
            **metrics_all,
            **prefixed_metrics("R", idx_r, pred_r),
            **prefixed_metrics("F", idx_f, pred_f),
        }
        fold_records.append(record)

        r_score = record["R_official_score"]
        f_score = record["F_official_score"]
        r_txt = "NA" if not np.isfinite(r_score) else f"{r_score:.3f}"
        f_txt = "NA" if not np.isfinite(f_score) else f"{f_score:.3f}"
        print(
            f"valid={valid_season} | train={len(train_idx):,} | valid={len(valid_idx):,} "
            f"| ALL={metrics_all['official_score']:.3f} "
            f"| R={r_txt} ({len(idx_r):,}) | F={f_txt} ({len(idx_f):,}) "
            f"| BS={metrics_all['brier']:.8f} | best_iter={best_iteration}"
        )

        del model, train_pool, valid_pool, x_train, x_valid, x_train_raw, x_valid_raw
        gc.collect()

    valid_oof = np.isfinite(oof)
    overall = dacon_bss_metrics(y_all[valid_oof], oof[valid_oof])

    overall_gt = game_type_upper.to_numpy()
    oof_r = valid_oof & (overall_gt == "R")
    oof_f = valid_oof & (overall_gt == "F")
    if oof_r.any():
        overall.update({f"R_{k}": v for k, v in dacon_bss_metrics(y_all[oof_r], oof[oof_r]).items()})
    if oof_f.any():
        overall.update({f"F_{k}": v for k, v in dacon_bss_metrics(y_all[oof_f], oof[oof_f]).items()})

    print(
        f"OOF seasons overall | n={valid_oof.sum():,} | BS={overall['brier']:.8f} "
        f"| score={overall['official_score']:.3f} "
        f"| R={overall.get('R_official_score', np.nan):.3f} "
        f"| F={overall.get('F_official_score', np.nan):.3f}"
    )

    return CVResult(
        experiment_name=experiment_name,
        filter_mode=filter_mode,
        feature_config=feature_config,
        oof_prediction=oof,
        fold_id=fold_id,
        fold_metrics=pd.DataFrame(fold_records),
        overall_metrics=overall,
        best_iterations=best_iterations,
    )


def run_time_cv(
    train: pd.DataFrame,
    feature_config: FeatureConfig,
    filter_mode: TrainFilterMode = "all",
    experiment_name: str = "experiment",
    n_splits: int = 4,
    valid_months: int = 2,
    min_train_months: int = 12,
    gap_months: int = 0,
    alpha_scale: float = 1.0,
    catboost_params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 150,
) -> CVResult:
    if TARGET not in train.columns:
        raise KeyError(f"학습 파일에 target 컬럼 '{TARGET}'이 없습니다.")

    y_all = pd.to_numeric(train[TARGET], errors="raise").astype(int).to_numpy()
    if not np.isin(y_all, [0, 1]).all():
        raise ValueError("control_success는 0/1이어야 합니다.")

    raw_x = train.drop(columns=[TARGET])
    row_x = add_rowwise_features(raw_x, feature_config)
    folds = make_expanding_month_folds(
        train,
        n_splits=n_splits,
        valid_months=valid_months,
        min_train_months=min_train_months,
        gap_months=gap_months,
    )

    params = default_catboost_params() if catboost_params is None else dict(catboost_params)
    oof = np.full(len(train), np.nan, dtype=float)
    fold_id = np.full(len(train), -1, dtype=int)
    fold_records: list[dict[str, Any]] = []
    best_iterations: list[int] = []
    period = build_month_period(train)

    print(f"\n===== {experiment_name} | filter={filter_mode} =====")
    for fold, (train_idx_base, valid_idx) in enumerate(folds):
        fold_train_meta = train.iloc[train_idx_base]
        keep_mask = training_filter_mask(fold_train_meta, filter_mode).to_numpy()
        train_idx = train_idx_base[keep_mask]

        if len(train_idx) < 1000:
            raise ValueError(
                f"fold={fold}: 필터 적용 후 학습 행이 {len(train_idx):,}개뿐입니다. "
                "필터 또는 fold 설정을 확인하세요."
            )

        x_train_raw = row_x.iloc[train_idx].copy()
        x_valid_raw = row_x.iloc[valid_idx].copy()
        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]

        smoother: BayesianRateSmoother | None = None
        if feature_config.bayesian_smoothing:
            smoother = BayesianRateSmoother(alpha_scale=alpha_scale)
            x_train_raw = smoother.fit_transform(x_train_raw, y_train)
            x_valid_raw = smoother.transform(x_valid_raw)

        x_train, cat_cols = prepare_catboost_frame(x_train_raw)
        x_valid, valid_cat_cols = prepare_catboost_frame(x_valid_raw)
        x_valid = x_valid.reindex(columns=x_train.columns)

        if cat_cols != valid_cat_cols:
            # 컬럼 집합은 동일해야 하며 순서는 x_train 기준으로 통일한다.
            valid_cat_set = set(valid_cat_cols)
            if set(cat_cols) != valid_cat_set:
                raise RuntimeError("train/valid 범주형 컬럼 구성이 다릅니다.")

        train_pool = Pool(x_train, label=y_train, cat_features=cat_cols)
        valid_pool = Pool(x_valid, label=y_valid, cat_features=cat_cols)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=params.get("verbose", 200),
        )

        pred = np.clip(model.predict_proba(valid_pool)[:, 1], EPS, 1.0 - EPS)
        oof[valid_idx] = pred
        fold_id[valid_idx] = fold

        metrics = dacon_bss_metrics(y_valid, pred)
        best_iteration = model.get_best_iteration()
        best_iteration = params.get("iterations", 2000) if best_iteration is None or best_iteration < 0 else best_iteration + 1
        best_iterations.append(int(best_iteration))

        record: dict[str, Any] = {
            "fold": fold,
            "train_rows_before_filter": len(train_idx_base),
            "train_rows_after_filter": len(train_idx),
            "valid_rows": len(valid_idx),
            "train_end_period": int(period.iloc[train_idx_base].max()),
            "valid_start_period": int(period.iloc[valid_idx].min()),
            "valid_end_period": int(period.iloc[valid_idx].max()),
            "best_iteration": int(best_iteration),
            **metrics,
        }
        fold_records.append(record)
        print(
            f"fold={fold} | train={len(train_idx):,} | valid={len(valid_idx):,} "
            f"| BS={metrics['brier']:.8f} | score={metrics['official_score']:.3f} "
            f"| best_iter={best_iteration}"
        )

        del model, train_pool, valid_pool, x_train, x_valid, x_train_raw, x_valid_raw
        gc.collect()

    valid_oof = np.isfinite(oof)
    overall = dacon_bss_metrics(y_all[valid_oof], oof[valid_oof])
    print(
        f"OOF overall | n={valid_oof.sum():,} | BS={overall['brier']:.8f} "
        f"| raw_BSS={overall['raw_bss']:.8f} | score={overall['official_score']:.3f}"
    )

    return CVResult(
        experiment_name=experiment_name,
        filter_mode=filter_mode,
        feature_config=feature_config,
        oof_prediction=oof,
        fold_id=fold_id,
        fold_metrics=pd.DataFrame(fold_records),
        overall_metrics=overall,
        best_iterations=best_iterations,
    )


# -----------------------------------------------------------------------------
# 9. Calibration: OOF 예측만 사용
# -----------------------------------------------------------------------------
def _logit(prob: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


class PlattCalibrator:
    """OOF probability의 logit을 입력으로 사용하는 sigmoid/Platt calibrator."""

    def __init__(self, c: float = 1e6):
        self.c = c
        self.model_: LogisticRegression | None = None

    def fit(self, prob: Sequence[float], y: Sequence[int]) -> "PlattCalibrator":
        p = np.asarray(prob, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise ValueError("calibration 학습 y에 두 클래스가 모두 있어야 합니다.")
        model = LogisticRegression(C=self.c, solver="lbfgs", max_iter=1000)
        model.fit(_logit(p).reshape(-1, 1), y_arr)
        self.model_ = model
        return self
