# LG Aimers 9기 — Pitch Control Success Prediction

> 투구 직전까지 확인 가능한 경기 상황과 과거 기록을 활용해 투구 제구 성공 확률을 예측한 팀 프로젝트

## Overview

- **Task**: 각 투구의 제구 성공 확률 예측
- **Model**: CatBoost 기반 확률 예측
- **My role**: 주요 성능 개선안의 실험·구현·검증, 개인 실험 결과 정리 및 팀 공유
- **Tech**: Python, pandas, NumPy, CatBoost, scikit-learn

## Problem

정규리그(R)와 퓨처스리그(F)는 **타깃 분포와 표본 규모가 달라**, 동일한 학습 기준을 일괄 적용하기 어려웠습니다.

또한 선수 누적 기록만 사용하면 현재 시즌의 상태가 과거 시즌과 섞일 수 있어, 단순 누적값을 그대로 쓰기보다 **현재 시즌을 더 잘 드러내는 변수 구성**이 필요했습니다.

## Key Decisions

### 1. 리그별 모델링 및 검증
R/F 리그의 분포와 표본 규모 차이를 확인하고, 리그별 학습 기준과 모델링 전략을 분리해 비교했습니다.

### 2. 시간 순서 기반 검증
random split보다 실제 예측 상황에 가까운 평가를 위해 시즌 순서를 유지한 walk-forward 검증을 사용했습니다.

- valid=2022 → train < 2022
- valid=2023 → train < 2023
- valid=2024 → train < 2024

검증 시즌 전체를 예측한 뒤 ALL / R / F 성능을 따로 기록했습니다.

### 3. 현재 시즌 정보 재구성
누적 기록만으로는 현재 시즌의 상태가 희석될 수 있어 시즌별 기록을 분리하고, 표본이 적은 선수의 비율은 smoothing을 적용해 보정했습니다.

### 4. 단계적 피처 실험
볼카운트, 투수 상황, 최근 폼, smoothing, 투수-타자 interaction, 구종 다양성, 경기 상황 변수를 단계적으로 추가하며 효과를 비교했습니다.

## Results

| Stage | Score |
|---|---:|
| Initial baseline | 549 |
| Model improved through my main experiments | **1,105** |
| Team final | **1,147** |

- **Final rank**: 94 / 1,087 teams
- **Top percentile**: 약 8.6%

> 1,105점까지의 주요 개선·구현·검증을 주도했으며, 1,147점은 팀 최종 결과입니다.

## Validation Principles

실험 코드에서는 다음 원칙을 유지했습니다.

- 시간 순서가 보존된 시즌별 walk-forward CV
- validation 행은 실험 간 동일하게 유지
- test 내부 다른 행을 이용한 rolling / count encoding 생성 금지
- Bayesian smoothing prior는 각 fold의 train 데이터로만 적합
- calibration은 OOF 예측을 기준으로 적합
- 제출 파일의 행 수, ID 순서, 확률 범위 등을 별도로 확인

## Repository Structure

```text
.
├── README.md
├── src/
│   └── season_league_cv.py      # 시즌·리그별 CatBoost 검증 파이프라인
├── experiments/
│   └── feature_ablation.py      # 피처 추가·제거 실험
├── submission/
│   └── build_submission.py      # 제출 파일 생성 및 점검
├── docs/
│   └── portfolio_notes.md       # 개인 기여와 팀 결과 구분
├── requirements.txt
└── .gitignore
```

## Notes

이 저장소는 **포트폴리오용 정리본**입니다. 대회 원본 데이터와 팀의 전체 제출 산출물은 포함하지 않았습니다.

코드는 실제 실험 과정에서 사용한 파일 중 대표성이 높은 부분을 선별해 파일명을 정리한 것이며, 팀 최종 성과와 개인 기여 범위를 구분해 작성했습니다.
