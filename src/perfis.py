"""Perfis de pré-processamento por família de modelo (decididos no notebook 03)."""
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.preprocessamento import BINARIAS, CATEGORICAS, CAUDA_LONGA, NUMERICAS

PROPORCAO_UNDERSAMPLING = 0.1  # fraudes = 10% dos legítimos, só no treino
LIMIAR_CATEGORIA_RARA = 2_000
SEMENTE = 42


def preprocessador_escala():
    """Perfil 'sensíveis à escala': mediana + indicador · log1p na cauda longa · Standard · one-hot."""
    numericas = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    cauda_longa = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    categoricas = OneHotEncoder(min_frequency=LIMIAR_CATEGORIA_RARA,
                                handle_unknown="infrequent_if_exist", sparse_output=False)
    return ColumnTransformer(
        [("num", numericas, NUMERICAS), ("cauda", cauda_longa, CAUDA_LONGA),
         ("bin", "passthrough", BINARIAS), ("cat", categoricas, CATEGORICAS)],
        verbose_feature_names_out=False,
    )


def pipeline_escala(modelo, undersampling=True):
    """Pipeline completo do perfil 'sensíveis à escala' para um modelo qualquer."""
    etapas = [("pre", preprocessador_escala())]
    if undersampling:
        etapas.append(("balanceamento", RandomUnderSampler(
            sampling_strategy=PROPORCAO_UNDERSAMPLING, random_state=SEMENTE)))
    etapas.append(("modelo", modelo))
    return Pipeline(etapas).set_output(transform="pandas")

def para_categoria(X):
    """Converte as colunas categóricas (texto) para o tipo `category` do pandas."""
    X = X.copy()
    X[CATEGORICAS] = X[CATEGORICAS].astype("category")
    return X

def pipeline_arvores(modelo, undersampling=False, categorias_nativas=True):
    """Pipeline do perfil 'árvores': ausentes como NaN, sem escala.

    `categorias_nativas=True` (LightGBM): categorias como tipo `category`, tratadas pelo próprio modelo.
    `categorias_nativas=False` (árvore do scikit-learn, que não aceita texto): códigos inteiros (ordinal).
    """
    if categorias_nativas:
        pre = FunctionTransformer(para_categoria, feature_names_out="one-to-one")
    else:
        pre = ColumnTransformer(
            [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAS)],
            remainder="passthrough", verbose_feature_names_out=False,
        )
    etapas = [("pre", pre)]
    if undersampling:
        etapas.append(("balanceamento", RandomUnderSampler(
            sampling_strategy=PROPORCAO_UNDERSAMPLING, random_state=SEMENTE)))
    etapas.append(("modelo", modelo))
    return Pipeline(etapas).set_output(transform="pandas")