"""Grupos de colunas e limpeza sem estado (decisões da EDA, notebook 01)."""

ALVO = "fraud_bool"
DESCARTAR = ["device_fraud_count", "month"]  # constante / só serve para dividir os dados

CATEGORICAS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
BINARIAS = ["email_is_free", "phone_home_valid", "phone_mobile_valid",
            "has_other_cards", "foreign_request", "keep_alive_session"]
CAUDA_LONGA = ["days_since_request", "session_length_in_minutes",
               "bank_branch_count_8w", "prev_address_months_count"]
NUMERICAS = ["income", "name_email_similarity", "current_address_months_count", "customer_age",
             "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w",
             "date_of_birth_distinct_emails_4w", "credit_risk_score", "bank_months_count",
             "proposed_credit_limit", "device_distinct_emails_8w"]
VELOCIDADES = ["velocity_6h", "velocity_24h", "velocity_4w"]  # drift mais forte (EDA, seção 8)

# Códigos de ausência, segundo o dicionário de dados (EDA, seção 2)
AUSENTE_MENOS_1 = ["prev_address_months_count", "current_address_months_count", "bank_months_count",
                   "session_length_in_minutes", "device_distinct_emails_8w"]
AUSENTE_NEGATIVO = ["intended_balcon_amount"]

# Não podem ser negativas pelo significado; exceção: credit_risk_score (escala inclui negativos)
NAO_NEGATIVAS = [c for c in NUMERICAS + CAUDA_LONGA if c != "credit_risk_score"]


def limpar(df):
    """Limpeza sem estado: não aprende nada dos dados, então não há risco de vazamento."""
    out = df.drop(columns=DESCARTAR + [ALVO], errors="ignore").copy()
    for c in AUSENTE_MENOS_1:
        out[c] = out[c].where(out[c] != -1)
    for c in AUSENTE_NEGATIVO:
        out[c] = out[c].where(out[c] >= 0)
    out[NAO_NEGATIVAS] = out[NAO_NEGATIVAS].clip(lower=0)
    return out