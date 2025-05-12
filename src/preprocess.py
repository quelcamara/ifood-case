import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_customers_info(data: pd.DataFrame) -> pd.DataFrame:
    """
    """
    # Ajuste do formato da data
    data["registered_on"] = pd.to_datetime(data["registered_on"], format="%Y%m%d")
    # Removendo placeholder de idade e substituindo por nulos
    # Coluna passando a ser float
    data.loc[data["age"] == 118, "age"] = np.NaN
    
    return data


def preprocess_transactions_info(data: pd.DataFrame) -> pd.DataFrame:
    """
    """
    # Expandindo informações contidas em `value`
    data = pd.concat([data.drop(columns=["value"]), data["value"].apply(pd.Series)], axis=1)

    # Combinando as informações das colunas de oferta e mantendo apenas a completa
    data["offer_id"] = data["offer id"].combine_first(data["offer_id"])
    data.drop(columns="offer id", inplace=True)
    
    return data


def multilabel_onehot_encode(data: pd.DataFrame, column: str) -> pd.DataFrame:
    # One-hot de multilabels
    mlb = MultiLabelBinarizer()
    original_index = data.index

    # Substitui nulos temporariamente
    values = data[column].apply(lambda x: x if isinstance(x, list) else [])

    encoded = mlb.fit_transform(values)
    df = pd.DataFrame(encoded, columns=[f"{column}_{c}" for c in mlb.classes_], index=original_index)

    return df


def join_campaigns_info(
        offers: pd.DataFrame, 
        customers: pd.DataFrame,
        transactions: pd.DataFrame, 
    ) -> pd.DataFrame:
    """
    """
    data = (
        transactions
        .merge(offers.rename(columns={"id": "offer_id"}), how="left", on="offer_id")
        .merge(customers.rename(columns={"id": "account_id"}), how="left", on="account_id")
    )
    # Reordenando colunas para facilitar leitura dos eventos
    columns = [
        "account_id",
        "age",
        "gender",
        "credit_card_limit",
        "registered_on",
        "offer_id",
        "offer_type",
        "duration",
        "min_value",
        "discount_value",
        "time_since_test_start",
        "event",
        "reward",
        "amount",
        "channels"
    ]
    data = data[columns]
    # Renomeando eventos para ordenação posterior por etapas
    map_events = {
        "offer received": "1-offer received",
        "offer viewed": "2-offer viewed",
        "offer completed": "3-offer completed",
        "transaction": "4-transaction"
    }
    data["event"] = data["event"].replace(map_events)
    data = data.sort_values(["account_id", "time_since_test_start", "event"]).reset_index(drop=True)

    return data


def enricher_transactions_information(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["account_id", "time_since_test_start"]).copy()
    oferta_cols = ["offer_id", "offer_type", "duration", "min_value", "discount_value", "channels"]
    resultados = []

    for account_id, grupo in df.groupby("account_id"):
        registros = []
        oferta = {}
        fim_validade = None
        permissoes = 0

        for _, row in grupo.iterrows():
            linha = row.copy()

            if row["event"] == "1-offer received":
                oferta = row[oferta_cols].to_dict()
                fim_validade = row["time_since_test_start"] + row["duration"]
                permissoes = 0  # reinicia permissões

            # Se estiver no prazo da oferta
            if fim_validade is not None and row["time_since_test_start"] <= fim_validade:
                if row["event"] == "3-offer completed":
                    permissoes += 1  # ganha nova permissão
                elif row["event"] == "4-transaction" and permissoes > 0:
                    for col in oferta_cols:
                        linha[col] = oferta[col]

                        if oferta.get("offer_type") in ["bogo", "discount"]:
                            linha["transaction_reward"] = oferta.get("discount_value", None)
                            
                    permissoes -= 1  # consome a permissão
                else:
                    for col in oferta_cols:
                        linha[col] = oferta[col]
            else:
                fim_validade = None
                permissoes = 0

            registros.append(linha)

        resultados.append(pd.DataFrame(registros))

    return pd.concat(resultados, ignore_index=True)
