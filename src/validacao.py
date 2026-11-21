"""Validação temporal e métricas do projeto (definidas no notebook 02)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

FPR_ALVO = 0.05

# (meses de treino, mês de validação) — janela crescente
FOLDS = [([0, 1, 2], 3), ([0, 1, 2, 3], 4), ([0, 1, 2, 3, 4], 5)]


def gerar_folds(df, folds=FOLDS):
    """Para cada fold, devolve os meses e os índices das linhas de treino e de validação."""
    for meses_treino, mes_val in folds:
        idx_tr = df.index[df["month"].isin(meses_treino)]
        idx_val = df.index[df["month"] == mes_val]
        yield meses_treino, mes_val, idx_tr, idx_val


def nome_fold(meses_treino, mes_val):
    return f"{meses_treino[0]}–{meses_treino[-1]} → {mes_val}"


def recall_a_fpr(y, score, fpr_alvo=FPR_ALVO):
    """Maior recall (TPR) alcançável sem passar de `fpr_alvo` de falsos positivos."""
    fpr, tpr, _ = roc_curve(y, score)
    return tpr[fpr <= fpr_alvo].max()


def avaliar(y, score):
    """As três métricas do projeto para um vetor de rótulos e um de scores."""
    return {
        "recall@5%FPR": recall_a_fpr(y, score),
        "PR-AUC": average_precision_score(y, score),
        "ROC AUC": roc_auc_score(y, score),
    }


def validar(modelo, df, alvo="fraud_bool"):
    """Avalia `modelo` nos folds temporais.

    `modelo` recebe (dados_treino, dados_validacao) e devolve um score por linha da validação.
    Retorna as métricas por fold, a média e o desvio padrão.
    """
    linhas = []
    for meses_tr, mes_val, idx_tr, idx_val in gerar_folds(df):
        score = modelo(df.loc[idx_tr], df.loc[idx_val])
        linhas.append({"fold": nome_fold(meses_tr, mes_val), **avaliar(df.loc[idx_val, alvo], score)})
    resultado = pd.DataFrame(linhas).set_index("fold")
    resultado.loc["média"] = resultado.mean()
    resultado.loc["desvio padrão"] = resultado.iloc[:-1].std()
    return resultado


def limiar_para_fpr(y, score, fpr_alvo=FPR_ALVO):
    """Limiar acima do qual ficam `fpr_alvo` dos casos legítimos (negativos)."""
    return np.quantile(score[np.asarray(y) == 0], 1 - fpr_alvo)


def taxas(y, score, limiar):
    """FPR e recall de uma regra 'barrar se score > limiar'."""
    y = np.asarray(y)
    alerta = score > limiar
    return {"FPR": alerta[y == 0].mean(), "recall": alerta[y == 1].mean()}