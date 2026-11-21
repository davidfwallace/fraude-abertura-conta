"""Execução de buscas de hiperparâmetros: em paralelo, salvando cada resultado assim que termina."""
import time
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from src.preprocessamento import ALVO, limpar
from src.validacao import validar


def pontuar(pipe, X):
    """Score de fraude: probabilidade, quando o modelo tem; senão, a função de decisão (ex.: SVM)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def modelo_de(fabrica):
    """Transforma uma função sem argumentos que cria um pipeline no formato que a `validar` espera."""
    def modelo(df_tr, df_val):
        pipe = fabrica()
        pipe.fit(limpar(df_tr), df_tr[ALVO])
        return pontuar(pipe, limpar(df_val))
    return modelo


def avaliar_candidato(candidato, fabricas, caminho_treino):
    """Avalia um candidato {'modelo': nome, 'params': {...}} nos folds e devolve uma linha de resultados."""
    df = pd.read_parquet(caminho_treino)
    fabrica = fabricas[candidato["modelo"]]
    inicio = time.time()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        res = validar(modelo_de(lambda: fabrica(**candidato["params"])), df)

    linha = {"id": candidato["id"], "modelo": candidato["modelo"], "params": repr(candidato["params"])}
    for metrica in res.columns:
        linha[f"{metrica} média"] = res.loc["média", metrica]
        linha[f"{metrica} desvio"] = res.loc["desvio padrão", metrica]
        for i, valor in enumerate(res[metrica].iloc[:3], start=1):
            linha[f"{metrica} fold {i}"] = valor
    linha["folds sem convergir"] = sum(issubclass(a.category, ConvergenceWarning) for a in avisos)
    linha["segundos"] = round(time.time() - inicio, 1)
    return linha


def rodar_busca(candidatos, fabricas, caminho_treino, arquivo, n_processos=3):
    """Roda os candidatos ainda não presentes em `arquivo`, gravando cada resultado ao terminar."""
    arquivo = Path(arquivo)
    feitos = set(pd.read_csv(arquivo)["id"]) if arquivo.exists() else set()
    pendentes = [c for c in candidatos if c["id"] not in feitos]
    print(f"{len(feitos)} já feitos, {len(pendentes)} pendentes")

    inicio = time.time()
    execucao = Parallel(n_jobs=n_processos, return_as="generator_unordered")(
        delayed(avaliar_candidato)(c, fabricas, caminho_treino) for c in pendentes
    )
    for i, linha in enumerate(execucao, start=1):
        pd.DataFrame([linha]).to_csv(arquivo, mode="a", header=not arquivo.exists(), index=False)
        print(f"[{i}/{len(pendentes)}] {linha['id']} -> recall {linha['recall@5%FPR média']:.4f} "
              f"({linha['segundos']:.0f}s, folds sem convergir: {linha['folds sem convergir']})")
    print(f"Concluído em {(time.time() - inicio) / 60:.1f} min")