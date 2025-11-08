# %% [markdown]
# # Задание 2. Бейзлайн и оценка качества
#
# **Цель:** построить и оценить базовые модели для задачи бинарной классификации (`win`).
#
# **Метрики:** AUC и F1. AUC порого-инвариантен и оценивает разделимость классов. F1 балансирует precision/recall.
#
# **Правила воспроизводимости:** фиксируем `RANDOM_STATE`, не используем утечки цели, ноутбук выполняется целиком.
#
# **Замечание об утечках:** признаки `win_prob` и `ev` не используются как фичи, так как они зависят от истинной
# вероятности выигрыша и создадут утечку цели.

# %%
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# воспроизводимость
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# %% [markdown]
# ## 1. Данные
# Генерируем синтетический датасет Leduc Hold’em. Целевая переменная `win` в {0,1}.
# Признаки для обучения: категориальные (`private, public, position, round, action`)
# и числовые (`bet_size, pot`). Исключаем `win_prob` и `ev` из признакового набора.

# %%
ranks = ['J', 'Q', 'K']
positions = ['SB', 'BB']
actions = ['check', 'call', 'raise', 'fold']


def hand_strength(priv: str, pub: str | None) -> float:
    if pub is None:
        return {'J': 0.45, 'Q': 0.50, 'K': 0.55}[priv]
    if pub == priv:
        return 0.80
    base = {'J': 0.40, 'Q': 0.50, 'K': 0.60}[priv]
    adj = {'J': 0.02, 'Q': 0.00, 'K': -0.02}[pub]
    return float(np.clip(base + adj, 0.05, 0.95))


rows = []
n_samples = 25_000
for _ in range(n_samples):
    priv = rng.choice(ranks)
    pub = rng.choice([None] + ranks, p=[0.6, 0.133, 0.133, 0.134])
    pos = rng.choice(positions)
    rnd = 'preflop' if pub is None else 'flop'
    pot = int(rng.integers(2, 10))

    s = hand_strength(priv, pub)
    logits = np.array([1 - s, 0.6, -0.2 + 2 * s, 1.2 - 2 * s])
    p = np.exp(logits - logits.max())
    p /= p.sum()

    act = rng.choice(actions, p=p)
    bet = int(rng.integers(1, 5)) if act == 'raise' else int(rng.integers(1, 3)) if act == 'call' else 0
    win_p = float(np.clip(s * (1.05 if act == 'raise' else 1.0) * (0.97 if act == 'fold' else 1.0), 0.01, 0.99))
    win = int(rng.binomial(1, win_p))
    ev = win_p * (pot + bet) - (1 - win_p) * bet

    rows.append((priv, pub if pub else 'None', pos, rnd, act, bet, pot, win, win_p, ev))

df = pd.DataFrame(
    rows,
    columns=['private', 'public', 'position', 'round', 'action',
             'bet_size', 'pot', 'win', 'win_prob', 'ev']
)

# быстрые проверки целостности
assert {'win'}.issubset(df.columns)
assert set(df['win'].unique()) <= {0, 1}
assert not df[['bet_size', 'pot']].isna().any().any()

# %% [markdown]
# ## 2. Разбиение на выборки
# Стратифицированный train/test с фиксированным `random_state`.

# %%
cat_cols = ['private', 'public', 'position', 'round', 'action']
num_cols = ['bet_size', 'pot']

X = df[cat_cols + num_cols].copy()
y = df['win'].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# проверка стратификации
p_tr, p_te = y_train.mean(), y_test.mean()
assert abs(p_tr - p_te) < 0.02

# %% [markdown]
# ## 3. Константный бейзлайн
# Оцениваем «most frequent» константу. Это нижняя граница качества.

# %%
dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
dummy.fit(X_train, y_train)
proba_d = dummy.predict_proba(X_test)[:, 1]
pred_d = dummy.predict(X_test)

const_auc = roc_auc_score(y_test, proba_d)
const_f1 = f1_score(y_test, pred_d)
const_acc = accuracy_score(y_test, pred_d)

# ручная проверка модального класса
maj_class = int(y_train.mode()[0])
pred_const = np.full_like(y_test, maj_class)
manual_f1 = f1_score(y_test, pred_const)

print(f"Const baseline | AUC={const_auc:.3f} F1={const_f1:.3f} Acc={const_acc:.3f} ManualF1={manual_f1:.3f}")

# %% [markdown]
# ## 4. Предобработка и модели
# Категориальные признаки кодируем `OneHotEncoder(drop='first', handle_unknown='ignore')`. Это устраняет дамми-ловушку
# и снижает мультиколлинеарность в линейной модели; `handle_unknown='ignore'` защищает от редких категорий на тесте.
# Числовые признаки передаются без изменений.
# Модели: `LogisticRegression` и `DecisionTreeClassifier` как простые бейзлайны.

# %%
pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

logreg_pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=200, random_state=RANDOM_STATE))
])

tree_pipe = Pipeline([
    ("pre", pre),
    ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))
])

# %% [markdown]
# ## 5. Обучение и оценка на отложенной выборке
# **Отчёт по метрикам на тестовой выборке (hold-out).**

# %%
@dataclass
class EvalResult:
    model: str
    auc: float
    f1: float


def evaluate(pipe: Pipeline, name: str) -> EvalResult:
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return EvalResult(model=name, auc=roc_auc_score(y_test, proba), f1=f1_score(y_test, pred))


res_logreg = evaluate(logreg_pipe, "LogisticRegression")
res_tree = evaluate(tree_pipe, "DecisionTree(max_depth=5)")

report = pd.DataFrame(
    [
        ["Const(most_frequent)", const_auc, const_f1],
        [res_logreg.model, res_logreg.auc, res_logreg.f1],
        [res_tree.model, res_tree.auc, res_tree.f1],
    ],
    columns=["Model", "AUC", "F1"],
)

print("Метрики на тестовой выборке (hold-out):")
print(report.to_string(index=False))

# %% [markdown]
# ## 6. Настройка порога для F1 (для логистической регрессии)
# AUC порого-инвариантен. F1 зависит от порога. Покажем лучший порог на сетке.

# %%
logreg_pipe.fit(X_train, y_train)
proba_lr = logreg_pipe.predict_proba(X_test)[:, 1]

grid = np.linspace(0.1, 0.9, 17)
pairs = [(t, f1_score(y_test, (proba_lr >= t).astype(int))) for t in grid]
best_t, best_f1 = max(pairs, key=lambda x: x[1])

print(f"Best threshold (LogReg) for F1: t={best_t:.2f}, F1={best_f1:.3f}")

# %% [markdown]
# ## 7. Воспроизводимость и самотесты
# Фиксирован `RANDOM_STATE = 42`. Идемпотентные предсказания. Стратификация сохранена.

# %%
# стабильность вывода пайплайна
proba_lr_r = logreg_pipe.predict_proba(X_test)[:, 1]
assert np.allclose(proba_lr, proba_lr_r), "Non-deterministic pipeline outputs"

# стратификация уже проверена выше (p_tr ~ p_te)
