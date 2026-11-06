"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

BRAIN_COLUMNS = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
MRI_COLUMN = "MRI_Count"

def _pearson_with_mri(subset: pd.DataFrame) -> dict[str, float]:
    r_values = subset[BRAIN_COLUMNS].corrwith(subset[MRI_COLUMN], method="pearson")
    return {str(col): float(r) for col, r in r_values.items()}

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """

    brain = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
    brain["Gender"] = brain["Gender"].str.strip()

    female_part = brain[brain["Gender"] == "Female"]
    male_part = brain[brain["Gender"] == "Male"]

    female_r = _pearson_with_mri(female_part)
    male_r = _pearson_with_mri(male_part)

    best_name = ""
    best_abs_r = -1.0
    for group_r in (female_r, male_r):
        for col, r in group_r.items():
            if pd.notna(r) and abs(r) > best_abs_r:
                best_abs_r = abs(r)
                best_name = col

    return BrainCorrelationSummary(
        men_count=int(len(male_part)),
        women_count=int(len(female_part)),
        women_mri_correlation=female_r,
        men_mri_correlation=male_r,
        strongest_mri_feature=best_name,
    )