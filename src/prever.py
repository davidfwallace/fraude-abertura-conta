"""Pontua novos pedidos com o modelo final.

Uso (a partir da raiz do repositório):
    python -m src.prever caminho/entrada.csv caminho/saida.csv          # política de orçamento
    python -m src.prever caminho/entrada.csv caminho/saida.csv 0.12     # com um limiar fixo

A entrada deve ter as mesmas colunas da base original (a coluna fraud_bool, se existir, é ignorada).
A saída traz, para cada pedido, a probabilidade de fraude e se ele deve ser revisado:
- sem limiar: política de orçamento — revisar a fração `orcamento` de pedidos com maior score do lote
  (exige um lote representativo, de pelo menos TAMANHO_MINIMO_LOTE pedidos);
- com limiar: revisar quem tiver probabilidade acima dele (para pedidos individuais ou lotes pequenos,
  com o limiar calculado numa janela recente de pedidos).
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
TAMANHO_MINIMO_LOTE = 1_000


def carregar():
    modelo = joblib.load(ARQ_MODELO)
    metadados = json.loads(ARQ_METADADOS.read_text(encoding="utf-8"))
    return modelo, metadados


def prever(df, modelo, orcamento=None, limiar=None):
    """Probabilidade de fraude e alerta de revisão para cada pedido.

    Com `limiar`, revisa quem estiver acima dele. Sem `limiar`, aplica o orçamento ao próprio lote
    (os `orcamento` × 100% mais suspeitos) — o que só faz sentido num lote grande o bastante.
    """
    prob = modelo.predict_proba(limpar(df))[:, 1]
    if limiar is None:
        if len(df) < TAMANHO_MINIMO_LOTE:
            raise ValueError(
                f"Lote com {len(df)} pedidos: a política de orçamento precisa de pelo menos "
                f"{TAMANHO_MINIMO_LOTE} para estimar o percentil. Informe um `limiar` calculado "
                f"numa janela recente de pedidos."
            )
        limiar = np.quantile(prob, 1 - orcamento)
    return pd.DataFrame({"prob_fraude": prob, "revisar": prob > limiar}, index=df.index)


if __name__ == "__main__":
    entrada, saida = sys.argv[1], sys.argv[2]
    limiar = float(sys.argv[3]) if len(sys.argv) > 3 else None
    modelo, metadados = carregar()
    pedidos = pd.read_csv(entrada)
    resultado = prever(pedidos, modelo, metadados["politica_limiar"]["orcamento"], limiar)
    resultado.to_csv(saida, index=False)
    print(f"{len(resultado):,} pedidos pontuados; {resultado['revisar'].sum():,} marcados para revisão -> {saida}")