"""Pontua novos pedidos com o modelo final.

Uso (a partir da raiz do repositório):
    python -m src.prever caminho/entrada.csv caminho/saida.csv

A entrada deve ter as mesmas colunas da base original (a coluna fraud_bool, se existir, é ignorada).
A saída traz, para cada pedido, a probabilidade de fraude e se ele está entre os mais suspeitos do lote
(política de orçamento: revisar a fração `orcamento` de pedidos com maior score).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessamento import limpar

RAIZ = Path(__file__).resolve().parent.parent
ARQ_MODELO = RAIZ / "models" / "lightgbm_final.joblib"
ARQ_METADADOS = RAIZ / "models" / "metadados.json"


def carregar():
    modelo = joblib.load(ARQ_MODELO)
    metadados = json.loads(ARQ_METADADOS.read_text(encoding="utf-8"))
    return modelo, metadados


def prever(df, modelo, orcamento):
    """Probabilidade de fraude e alerta (os `orcamento` × 100% pedidos mais suspeitos do lote)."""
    prob = modelo.predict_proba(limpar(df))[:, 1]
    limiar = np.quantile(prob, 1 - orcamento)
    return pd.DataFrame({"prob_fraude": prob, "revisar": prob > limiar}, index=df.index)


if __name__ == "__main__":
    entrada, saida = sys.argv[1], sys.argv[2]
    modelo, metadados = carregar()
    pedidos = pd.read_csv(entrada)
    resultado = prever(pedidos, modelo, metadados["politica_limiar"]["orcamento"])
    resultado.to_csv(saida, index=False)
    print(f"{len(resultado):,} pedidos pontuados; {resultado['revisar'].sum():,} marcados para revisão -> {saida}")