"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """

    passengers = pd.read_csv(data.csv_path)

    nan_counts = {
        str(col_name): int(nan_num) for col_name, nan_num in passengers.isna().sum().items()
    }

    older_than_30 = int((passengers["Age"] > 30).sum())

    age_per_class = passengers.groupby("Pclass")["Age"].mean()
    avg_age = {int(cls): float(avg) for cls, avg in age_per_class.items()}

    survived_per_class = passengers.groupby("Pclass")["Survived"].mean()
    survived_share = {
        int(cls): float(share) for cls, share in survived_per_class.items()
    }

    top_fares = [float(price) for price in passengers["Fare"].nlargest(5)]

    return TitanicSummary(
        row_count=int(len(passengers)),
        missing_by_column=nan_counts,
        adults_over_30_count=older_than_30,
        mean_age_by_pclass=avg_age,
        survival_rate_by_pclass=survived_share,
        highest_fares=top_fares,
    )