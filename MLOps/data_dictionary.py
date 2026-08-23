"""
data_dictionary.py — Gera o dicionário de dados a partir do banco.

O dicionário NÃO é escrito à mão. Ele é lido do catálogo do PostgreSQL: os
tipos, as chaves, os índices e as descrições que o `schema.sql` gravou com
`COMMENT ON`.

Por que gerar em vez de escrever:
    documentação escrita à mão envelhece em silêncio. Alguém adiciona uma
    coluna, esquece do documento, e a partir dali o dicionário mente — o que é
    pior do que não existir, porque ninguém desconfia. Gerando do catálogo, a
    única forma de o dicionário ficar errado é o comentário no `schema.sql`
    estar errado, e esse fica ao lado da definição da coluna.

Uso:
    python -m MLOps.data_dictionary                      # imprime no terminal
    python -m MLOps.data_dictionary --output DICIONARIO_DE_DADOS.md
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from MLOps.db import get_engine

SCHEMAS = ("staging", "feature_store", "serving", "mlops")

# Descrição de cada camada — é a única parte escrita à mão, porque explica a
# INTENÇÃO da arquitetura, que não cabe num COMMENT de tabela.
CAMADAS = {
    "staging":       ("Camada silver", "Dados limpos e padronizados, ainda no nível da tabela de origem."),
    "feature_store": ("Camada gold",   "A ABT: uma linha por cliente, pronta para o modelo. É daqui que a API lê as features na hora da decisão."),
    "serving":       ("Serviço",       "O que o modelo produziu em produção: toda decisão de crédito, com a versão do modelo que a tomou."),
    "mlops":         ("Governança",    "Registro de modelos e histórico de monitoramento — o que permite auditar e detectar degradação."),
}


def _consultar(sql: str, **parametros) -> list:
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), parametros).mappings()]


def tabelas() -> list:
    """Tabelas e visões dos schemas do projeto, com a descrição e o tamanho."""
    return _consultar("""
        SELECT c.relname                                   AS tabela,
               n.nspname                                   AS schema,
               CASE c.relkind WHEN 'r' THEN 'tabela' WHEN 'v' THEN 'visao' END AS tipo,
               obj_description(c.oid)                      AS descricao,
               pg_size_pretty(pg_total_relation_size(c.oid)) AS tamanho,
               COALESCE(s.n_live_tup, 0)                   AS linhas
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
        WHERE n.nspname = ANY(:schemas) AND c.relkind IN ('r', 'v')
        ORDER BY n.nspname, c.relkind, c.relname
    """, schemas=list(SCHEMAS))


def colunas(schema: str, tabela: str) -> list:
    """Colunas de uma tabela: tipo, obrigatoriedade, default e descrição."""
    return _consultar("""
        SELECT a.attname                                        AS coluna,
               format_type(a.atttypid, a.atttypmod)             AS tipo,
               a.attnotnull                                     AS obrigatoria,
               a.attgenerated <> ''                             AS gerada,
               pg_get_expr(d.adbin, d.adrelid)                  AS padrao,
               col_description(a.attrelid, a.attnum)            AS descricao
        FROM pg_attribute a
        JOIN pg_class c     ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE n.nspname = :schema AND c.relname = :tabela
          AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY a.attnum
    """, schema=schema, tabela=tabela)


def restricoes(schema: str, tabela: str) -> list:
    """Chaves primárias, estrangeiras e regras de validação (CHECK)."""
    return _consultar("""
        SELECT con.conname                     AS nome,
               CASE con.contype WHEN 'p' THEN 'PK' WHEN 'f' THEN 'FK'
                                WHEN 'c' THEN 'CHECK' WHEN 'u' THEN 'UNIQUE' END AS tipo,
               pg_get_constraintdef(con.oid)   AS definicao
        FROM pg_constraint con
        JOIN pg_class c     ON c.oid = con.conrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relname = :tabela
        ORDER BY con.contype
    """, schema=schema, tabela=tabela)


def indices(schema: str, tabela: str) -> list:
    """Índices — mostram por quais caminhos a tabela é consultada."""
    return _consultar("""
        SELECT indexname AS nome, indexdef AS definicao
        FROM pg_indexes
        WHERE schemaname = :schema AND tablename = :tabela
        ORDER BY indexname
    """, schema=schema, tabela=tabela)


# ============================================================
# GERAÇÃO DO MARKDOWN
# ============================================================

def _sim_nao(valor) -> str:
    return "sim" if valor else "—"


def gerar() -> str:
    """Monta o documento inteiro."""
    from datetime import datetime

    linhas = [
        "# Dicionário de Dados — Solução de Risco de Crédito",
        "",
        "> **Documento gerado**, não escrito à mão. Sai do catálogo do PostgreSQL",
        "> (tipos, chaves, índices e os `COMMENT ON` do `MLOps/sql/schema.sql`).",
        "> Para atualizar: `python -m MLOps.data_dictionary --output DICIONARIO_DE_DADOS.md`",
        "",
        f"Gerado em {datetime.now():%d/%m/%Y %H:%M}.",
        "",
        "---",
        "",
        "## Visão geral",
        "",
        "| Schema | Camada | Papel |",
        "|---|---|---|",
    ]
    for schema, (camada, papel) in CAMADAS.items():
        linhas.append(f"| `{schema}` | {camada} | {papel} |")

    todas = tabelas()

    linhas += ["", "### Objetos", "",
               "| Schema | Objeto | Tipo | Linhas (aprox.) | Tamanho | Descrição |",
               "|---|---|---|---:|---:|---|"]
    for t in todas:
        linhas.append(
            f"| `{t['schema']}` | `{t['tabela']}` | {t['tipo']} | "
            f"{t['linhas']:,} | {t['tamanho']} | {t['descricao'] or '—'} |".replace(",", ".")
        )

    # Detalhe de cada objeto
    schema_atual = None
    for t in todas:
        if t["schema"] != schema_atual:
            schema_atual = t["schema"]
            camada, papel = CAMADAS.get(schema_atual, ("", ""))
            linhas += ["", "---", "", f"## Schema `{schema_atual}` — {camada}", "", papel, ""]

        linhas += ["", f"### `{t['schema']}.{t['tabela']}`", ""]
        if t["descricao"]:
            linhas.append(f"{t['descricao']}")
            linhas.append("")
        if t["tipo"] == "tabela":
            linhas.append(f"*~{t['linhas']:,} linhas · {t['tamanho']}*".replace(",", "."))
            linhas.append("")

        linhas += ["| Coluna | Tipo | Obrigatória | Gerada | Descrição |",
                   "|---|---|---|---|---|"]
        for c in colunas(t["schema"], t["tabela"]):
            linhas.append(
                f"| `{c['coluna']}` | `{c['tipo']}` | {_sim_nao(c['obrigatoria'])} | "
                f"{_sim_nao(c['gerada'])} | {c['descricao'] or '—'} |"
            )

        regras = restricoes(t["schema"], t["tabela"])
        if regras:
            linhas += ["", "**Chaves e regras de validação**", ""]
            for r in regras:
                linhas.append(f"- **{r['tipo']}** `{r['nome']}` — `{r['definicao']}`")

        idx = [i for i in indices(t["schema"], t["tabela"]) if "_pkey" not in i["nome"]]
        if idx:
            linhas += ["", "**Índices**", ""]
            for i in idx:
                # Só a parte útil da definição (o "ON tabela USING ...")
                definicao = i["definicao"].split(" USING ", 1)[-1]
                linhas.append(f"- `{i['nome']}` — {definicao}")

    linhas.append("")
    return "\n".join(linhas)


def _run_cli() -> int:
    parser = argparse.ArgumentParser(description="Gera o dicionario de dados a partir do banco")
    parser.add_argument("--output", default=None,
                        help="arquivo .md de saida (padrao: imprime no terminal)")
    args = parser.parse_args()

    documento = gerar()

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(documento)
        print(f"Dicionario gerado: {args.output} ({len(documento.splitlines())} linhas)")
    else:
        print(documento)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
