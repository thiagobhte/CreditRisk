# /Dados

Camada de dados do projeto. Os arquivos CSV **não são versionados** no git
(são grandes demais — a ABT sozinha passa de 1 GB); esta pasta documenta a
estrutura esperada e é onde o pipeline lê e escreve.

## Estrutura esperada

```
Dados/
├── raw_data/                  # CSVs brutos do Kaggle (Home Credit Default Risk)
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── installments_payments.csv
│   └── credit_card_balance.csv
├── clean_data.csv             # gerado por DataPipeline/data_sanitization.py
└── abt.csv                    # gerado por DataPipeline/abt_transform.py
```

## Como obter / gerar

1. Baixe os CSVs brutos da competição e coloque-os em `Dados/raw_data/`:
   https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Gere os dados processados a partir da raiz do projeto:
   ```bash
   python -m DataPipeline.data_sanitization   # → Dados/clean_data.csv
   python -m DataPipeline.abt_transform       # → Dados/abt.csv
   ```

> O caminho desta pasta é controlado por `DATA_DIR` em [../config.py](../config.py)
> (default: `Dados/`). Para usar outro local, defina a variável de ambiente `DATA_DIR`.
