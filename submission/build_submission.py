from __future__ import annotations

"""
AIMERS 9 - DACON 코드제출 규격용 02_add_pitcher_context 패키지 빌더

실행:
    py aimers9_build_dacon_submit_02.py

입력(이 파일과 같은 폴더):
    map_train.csv
    aimers9_outputs/feature/02_add_pitcher_context.joblib   # 있으면 설정 참고용

출력:
    aimers9_outputs/dacon_02/
      submit_02_mapping.zip      <- DACON 제출 파일
      package_preview/
        model/catboost_model.cbm
        model/metadata.json
        script.py
        requirements.txt

중요:
- map_train에서 사용한 mapping QC 6개(p_n_tot/p_purity/p_hand_ok, b_*)를
  평가 서버 test.csv에도 map_pitcher/map_batter를 row-wise lookup join하여 그대로 재현한다.
- build_mapping.py/LCS는 평가 서버에서 다시 실행하지 않는다. 이미 확정된 mapping CSV만 사용한다.
- 02까지만 사용: count state + pitcher-specific count/hand context.
- calibration / recent form / smoothing / pitcher-batter / pitchmix / game context 없음.
- inference script는 test 행끼리 집계/정렬/rolling을 하지 않는다.
"""

from pathlib import Path
import json
import shutil
import zipfile
import warnings

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "map_train.csv"
MAP_PITCHER_PATH = BASE_DIR / "map_pitcher.csv"
MAP_BATTER_PATH = BASE_DIR / "map_batter.csv"
OUTPUT_ROOT = BASE_DIR / "aimers9_outputs" / "dacon_02"
PACKAGE_DIR = OUTPUT_ROOT / "package_preview"
MODEL_DIR = PACKAGE_DIR / "model"
ZIP_PATH = OUTPUT_ROOT / "submit_02_mapping.zip"

TARGET = "control_success"

# map_train에 join해서 사용했던 매핑 신뢰도/QC 열. hidden test에도 mapping CSV로 재현한다.
MAPPING_QC_COLS = {
    "p_n_tot", "p_purity", "p_hand_ok",
    "b_n_tot", "b_purity", "b_hand_ok",
}

# 기존 CV result가 있으면 filter/iteration 설정만 참고한다.
CV_RESULT_CANDIDATES = [
    BASE_DIR / "aimers9_outputs" / "feature" / "02_add_pitcher_context.joblib",
    BASE_DIR / "aimers9_outputs" / "feature" / "02A_baseline.joblib",
]
FALLBACK_FILTER = "r_or_2023plus"   # all / r_only / r_or_2023plus
FALLBACK_ITERATIONS = 800

TASK_TYPE = "CPU"   # 로컬 최종학습. NVIDIA/CUDA 환경 확실하면 GPU로 변경 가능
CPU_THREADS = 6
RANDOM_SEED = 42

MISSING_CAT = "__MISSING__"

BASE_CATEGORICAL_COLUMNS = {
    "top_bottom", "game_type", "base_state",
    "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "game_dayofweek",
    "season_cat", "game_month_cat", "inning_cat", "outs_before_cat",
    "count_state", "count_advantage", "count_pressure_state",
    "hand_matchup", "pitcher_count", "pitcher_hand_matchup",
    "pitcher_count_hand_matchup",
}


def _cat_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_CAT).astype(str)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _combine_categories(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [_cat_string(frame[c]) for c in columns]
    out = parts[0]
    for part in parts[1:]:
        out = out.str.cat(part, sep="__")
    return out


def add_02_features(frame: pd.DataFrame) -> pd.DataFrame:
    """02 = count state + pitcher context. 각 행 자체 정보만 사용."""
    out = frame.copy()

    for source, target in [
        ("season", "season_cat"),
        ("game_month", "game_month_cat"),
        ("inning", "inning_cat"),
        ("outs_before", "outs_before_cat"),
    ]:
        if source in out.columns:
            out[target] = _cat_string(out[source])

    required = ["balls_before", "strikes_before", "pitcher_id", "pitcher_hand", "batter_hand"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"02 feature 생성에 필요한 컬럼이 없습니다: {missing}")

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
        + "__" + out["three_ball"].astype(str)
        + "__" + out["full_count"].astype(str)
    )

    out["hand_matchup"] = _combine_categories(out, ["pitcher_hand", "batter_hand"])
    out["pitcher_count"] = _combine_categories(out, ["pitcher_id", "count_state"])
    out["pitcher_hand_matchup"] = _combine_categories(out, ["pitcher_id", "hand_matchup"])
    out["pitcher_count_hand_matchup"] = _combine_categories(
        out, ["pitcher_id", "count_state", "hand_matchup"]
    )
    return out


def prepare_catboost_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # mapping QC 6개도 모델 입력으로 유지한다. hidden test에는 mapping CSV를 join해 재현한다.
    drop_cols = [TARGET, "row_id"]
    out = frame.drop(columns=drop_cols, errors="ignore").copy()
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

    for col in [c for c in out.columns if c not in cat_cols]:
        if pd.api.types.is_bool_dtype(out[col].dtype):
            out[col] = out[col].astype("int8")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out, cat_cols


def training_filter_mask(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "all":
        return pd.Series(True, index=frame.index)
    game_type = _cat_string(frame["game_type"]).str.upper()
    season = _numeric(frame["season"])
    if mode == "r_only":
        return game_type.eq("R")
    if mode == "r_or_2023plus":
        return game_type.eq("R") | season.ge(2023)
    if mode == "recent_2023plus":
        return season.ge(2023)
    raise ValueError(f"지원하지 않는 filter_mode: {mode}")


def load_cv_settings() -> tuple[str, int, str]:
    for path in CV_RESULT_CANDIDATES:
        if not path.exists():
            continue
        try:
            obj = joblib.load(path)
            filter_mode = str(getattr(obj, "filter_mode"))
            med = int(getattr(obj, "median_best_iteration"))
            if med <= 0:
                continue
            return filter_mode, med, str(path)
        except Exception as exc:
            warnings.warn(f"CV result 설정을 읽지 못해 건너뜁니다: {path} | {exc}")
    return FALLBACK_FILTER, FALLBACK_ITERATIONS, "fallback"


def train_final_model(train: pd.DataFrame, filter_mode: str, iterations: int) -> tuple[CatBoostClassifier, list[str], list[str]]:
    keep = training_filter_mask(train, filter_mode)
    fit_df = train.loc[keep].copy()
    y = pd.to_numeric(fit_df[TARGET], errors="raise").astype(int).to_numpy()

    x_raw = add_02_features(fit_df.drop(columns=[TARGET]))
    x, cat_cols = prepare_catboost_frame(x_raw)

    params = {
        "loss_function": "Logloss",
        "eval_metric": "BrierScore",
        "iterations": int(iterations),
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "max_ctr_complexity": 1,
        "one_hot_max_size": 10,
        "random_seed": RANDOM_SEED,
        "task_type": TASK_TYPE,
        "allow_writing_files": False,
        "verbose": 200,
    }
    if TASK_TYPE == "GPU":
        params["devices"] = "0"
    else:
        params["thread_count"] = CPU_THREADS

    print(f"최종학습 행={len(fit_df):,} | features={x.shape[1]} | cat={len(cat_cols)}")
    print(f"filter={filter_mode} | iterations={iterations} | task_type={TASK_TYPE}")
    model = CatBoostClassifier(**params)
    model.fit(Pool(x, label=y, cat_features=cat_cols))
    return model, list(x.columns), cat_cols


SCRIPT_TEMPLATE = r'''from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

TARGET = "control_success"
MISSING_CAT = "__MISSING__"
MAPPING_QC_COLS = {
    "p_n_tot", "p_purity", "p_hand_ok",
    "b_n_tot", "b_purity", "b_hand_ok",
}
BASE_CATEGORICAL_COLUMNS = {
    "top_bottom", "game_type", "base_state",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "game_dayofweek",
    "season_cat", "game_month_cat", "inning_cat", "outs_before_cat",
    "count_state", "count_advantage", "count_pressure_state",
    "hand_matchup", "pitcher_count", "pitcher_hand_matchup",
    "pitcher_count_hand_matchup",
}


def _cat_string(series):
    return series.astype("string").fillna(MISSING_CAT).astype(str)


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _combine_categories(frame, columns):
    parts = [_cat_string(frame[c]) for c in columns]
    out = parts[0]
    for part in parts[1:]:
        out = out.str.cat(part, sep="__")
    return out


def add_02_features(frame):
    out = frame.copy()

    for source, target in [
        ("season", "season_cat"),
        ("game_month", "game_month_cat"),
        ("inning", "inning_cat"),
        ("outs_before", "outs_before_cat"),
    ]:
        if source in out.columns:
            out[target] = _cat_string(out[source])

    balls = _numeric(out["balls_before"])
    strikes = _numeric(out["strikes_before"])
    out["count_state"] = _cat_string(out["balls_before"]).str.cat(_cat_string(out["strikes_before"]), sep="_")
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
        out["two_strike"].astype(str) + "__" + out["three_ball"].astype(str) + "__" + out["full_count"].astype(str)
    )
    out["hand_matchup"] = _combine_categories(out, ["pitcher_hand", "batter_hand"])
    out["pitcher_count"] = _combine_categories(out, ["pitcher_id", "count_state"])
    out["pitcher_hand_matchup"] = _combine_categories(out, ["pitcher_id", "hand_matchup"])
    out["pitcher_count_hand_matchup"] = _combine_categories(out, ["pitcher_id", "count_state", "hand_matchup"])
    return out


def prepare(frame, feature_columns, categorical_columns):
    out = frame.drop(columns=[TARGET, "row_id"], errors="ignore").copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    for col in out.columns:
        if col in categorical_columns:
            out[col] = _cat_string(out[col])
        else:
            if pd.api.types.is_bool_dtype(out[col].dtype):
                out[col] = out[col].astype("int8")
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out = out.reindex(columns=feature_columns)
    for col in categorical_columns:
        out[col] = _cat_string(out[col])
    return out


def attach_mapping_qc(test, base):
    mp = pd.read_csv(base / "model" / "map_pitcher.csv").rename(
        columns={"n_tot": "p_n_tot", "purity": "p_purity", "hand_ok": "p_hand_ok"}
    )
    mb = pd.read_csv(base / "model" / "map_batter.csv").rename(
        columns={"n_tot": "b_n_tot", "purity": "b_purity", "hand_ok": "b_hand_ok"}
    )

    # 사용자가 로컬 map_test를 만들 때와 동일한 join.
    out = (test.merge(mp, on="pitcher_id", how="left", validate="many_to_one")
               .merge(mb, on="batter_id", how="left", validate="many_to_one"))
    out = out.drop(columns=["pitcher_trackman_id", "batter_trackman_id"], errors="ignore")
    return out


def resolve_data_file(base, filename):
    # DACON 가이드/평가환경 표기 차이를 모두 허용한다.
    candidates = [
        base / "data" / filename,
        base / "open" / "data" / filename,
        base / "open" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(base.glob(f"**/{filename}"))
    hits = [p for p in hits if "output" not in p.parts and "model" not in p.parts]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"{filename}를 찾을 수 없습니다. candidates={candidates}, hits={hits[:5]}")


def main():
    base = Path(__file__).resolve().parent
    test_path = resolve_data_file(base, "test.csv")
    model_path = base / "model" / "catboost_model.cbm"
    metadata_path = base / "model" / "metadata.json"
    output_path = base / "output" / "submission.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"평가 test.csv를 찾을 수 없습니다: {test_path}")

    test_raw = pd.read_csv(test_path, low_memory=False)
    test = attach_mapping_qc(test_raw, base)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_columns = meta["feature_columns"]
    categorical_columns = meta["categorical_columns"]

    x_raw = add_02_features(test)
    x = prepare(x_raw, feature_columns, categorical_columns)

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    pred = model.predict_proba(Pool(x, cat_features=categorical_columns))[:, 1]
    pred = np.clip(pred, 1e-6, 1.0 - 1e-6)

    if len(pred) != len(test_raw):
        raise RuntimeError("prediction row count mismatch")
    if not np.isfinite(pred).all():
        raise RuntimeError("prediction contains NaN/inf")

    # 가능하면 평가 서버 sample_submission 형식을 그대로 보존한다.
    try:
        sample_path = resolve_data_file(base, "sample_submission.csv")
        submission = pd.read_csv(sample_path)
        if len(submission) != len(pred):
            raise ValueError("sample_submission row count mismatch")
        if TARGET not in submission.columns:
            raise KeyError(f"sample_submission에 {TARGET} 컬럼이 없습니다: {submission.columns.tolist()}")
        submission[TARGET] = pred
        if "row_id" in submission.columns and "row_id" in test_raw.columns:
            if not submission["row_id"].reset_index(drop=True).equals(test_raw["row_id"].reset_index(drop=True)):
                raise ValueError("sample_submission과 test의 row_id가 다릅니다.")
    except FileNotFoundError:
        row_id = test_raw["row_id"] if "row_id" in test_raw.columns else np.arange(len(test_raw))
        submission = pd.DataFrame({"row_id": row_id, TARGET: pred})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"submission saved: {output_path}")
    print(f"rows={len(submission):,}, pred_mean={float(np.mean(pred)):.6f}")


if __name__ == "__main__":
    main()
'''


def write_package(model: CatBoostClassifier, feature_columns: list[str], cat_cols: list[str], filter_mode: str, iterations: int) -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_DIR / "catboost_model.cbm"))
    shutil.copy2(MAP_PITCHER_PATH, MODEL_DIR / "map_pitcher.csv")
    shutil.copy2(MAP_BATTER_PATH, MODEL_DIR / "map_batter.csv")

    metadata = {
        "feature_columns": feature_columns,
        "categorical_columns": cat_cols,
        "model_name": "02_add_pitcher_context_with_mapping_qc",
        "filter_mode": filter_mode,
        "iterations": int(iterations),
        "mapping_qc_included": sorted(MAPPING_QC_COLS),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    (PACKAGE_DIR / "script.py").write_text(SCRIPT_TEMPLATE, encoding="utf-8")

    # pandas/numpy는 대회 서버 기본 설치 버전을 사용. CatBoost만 추가 설치.
    (PACKAGE_DIR / "requirements.txt").write_text("catboost==1.2.8\n", encoding="utf-8")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # ZIP 최상위에 추가 상위폴더 없이 정확히 model/, script.py, requirements.txt 배치
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(PACKAGE_DIR).as_posix())


def inspect_zip() -> None:
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
    required = {
        "script.py", "requirements.txt",
        "model/catboost_model.cbm", "model/metadata.json",
        "model/map_pitcher.csv", "model/map_batter.csv",
    }
    missing = required.difference(names)
    bad_root = [n for n in names if n.startswith("package_preview/") or n.startswith("submit_02/")]
    if missing:
        raise RuntimeError(f"ZIP 필수 파일 누락: {sorted(missing)}")
    if bad_root:
        raise RuntimeError(f"ZIP에 불필요한 최상위 폴더가 있습니다: {bad_root[:5]}")
    print("ZIP 구조 확인 완료:")
    for n in names:
        print("  ", n)


def main() -> None:
    missing_inputs = [p for p in [TRAIN_PATH, MAP_PITCHER_PATH, MAP_BATTER_PATH] if not p.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "필수 입력 파일이 없습니다: " + ", ".join(str(p) for p in missing_inputs) + "\n"
            "이 빌더와 map_train.csv, map_pitcher.csv, map_batter.csv를 같은 폴더에 두세요."
        )

    print("[1/4] map_train + mapping CSV 로드")
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    if TARGET not in train.columns:
        raise KeyError(f"'{TARGET}' 컬럼이 없습니다.")
    print(f"train={train.shape}")

    present_qc = sorted(MAPPING_QC_COLS.intersection(train.columns))
    missing_qc = sorted(MAPPING_QC_COLS.difference(train.columns))
    print(f"모델에 포함할 mapping QC={present_qc}")
    if missing_qc:
        raise KeyError(
            f"map_train.csv에 mapping QC 컬럼이 없습니다: {missing_qc}. "
            "사용자가 보낸 join 코드로 map_train.csv를 다시 만든 뒤 실행하세요."
        )

    print("\n[2/4] 기존 CV 설정 참고")
    filter_mode, iterations, source = load_cv_settings()
    print(f"setting source={source}")
    print(f"filter_mode={filter_mode}, iterations={iterations}")
    print("기존 02와 동일하게 mapping QC 6개를 유지합니다. hidden test에서는 model/ mapping CSV로 재현합니다.")

    print("\n[3/4] mapping join 재현 가능한 02 모델 재학습")
    model, feature_columns, cat_cols = train_final_model(train, filter_mode, iterations)

    print("\n[4/4] DACON submit.zip 생성")
    write_package(model, feature_columns, cat_cols, filter_mode, iterations)
    inspect_zip()

    size_mb = ZIP_PATH.stat().st_size / 1024**2
    print("\n완료")
    print(f"DACON 제출 파일: {ZIP_PATH}")
    print(f"zip size={size_mb:.2f} MB")
    print("이 ZIP 파일 자체를 DACON 제출 탭에 업로드하세요.")


if __name__ == "__main__":
    main()