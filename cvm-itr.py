"""
cvm-itr.py - Fetch CVM ITR quarterly data (BPP + DRE) for a list of CNPJs.

Usage:
    python cvm-itr.py                    # Uses tickers from local API
    python cvm-itr.py --csv tickers.csv  # From CSV with CNPJ column
    python cvm-itr.py --cnpjs 00000000000191  # Direct CNPJ list

Output: DataFrame with quarterly financials (2011-2026).
Note: Some companies (e.g. Itau) do NOT file ITR. For those, annual DFP
data from DFC/DRE files can be used as fallback.
"""
import io
import zipfile
import argparse
from pathlib import Path

import pandas as pd
import requests

ITR_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS"
DFP_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS"
YEARS = list(range(2011, 2027))

BPP_ACCOUNTS = {
    "1": "ATIVO TOTAL",
    "1.01": "ATIVO CIRCULANTE",
    "1.01.01": "CAIXA",
    "1.01.03": "TITULOS VALORES MOBILIARIOS",
    "1.01.04": "EMPRESTIMOS",
    "1.01.06": "CLIENTES",
    "1.01.07": "ESTOQUES",
    "1.02": "ATIVO NAO CIRCULANTE",
    "1.02.01": "INVESTIMENTOS",
    "1.02.03": "IMOVEIS MAQUINAS",
    "1.02.06": "INTANGIVEIS",
    "2": "PASSIVO TOTAL",
    "2.01": "PASSIVO CIRCULANTE",
    "2.01.01": "FORNECEDORES",
    "2.01.02": "EMPRESTIMOS CP",
    "2.01.04": "IMPOSTOS",
    "2.02": "PASSIVO NAO CIRCULANTE",
    "2.02.01": "EMPRESTIMOS LP",
    "2.02.03": "DEBITOS TRIBUTARIOS",
    "2.03": "PATRIMONIO LIQUIDO",
    "2.03.01": "CAPITAL SOCIAL",
    "2.03.03": "RESERVES",
    "2.03.04": "LUCROS ACUMULADOS",
}

DRE_ACCOUNTS = {
    "3.01": "RECEITA LIQUIDA",
    "3.02": "CUSTOS",
    "3.03": "RESULTADO BRUTO",
    "3.04": "DESPESAS OPERACIONAIS",
    "3.05": "RESULTADO ANTES TRIBUTOS",
    "3.06": "TRIBUTOS",
    "3.07": "RESULTADO OPERACIONAL",
    "3.09": "RESULTADO ANTES PARTICIPACOES",
    "3.10": "PARTICIPACOES",
    "3.11": "LUCRO LIQUIDO",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download_zip(url: str) -> pd.DataFrame | None:
    """Download a CVM zip and extract DRE_con + BPP_con files."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  ! {url}: {e}")
        return None

    frames = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for pattern, table in [("DRE_con", "DRE"), ("BPP_con", "BPP")]:
            names = [n for n in zf.namelist() if pattern in n]
            if names:
                df = pd.read_csv(zf.open(names[0]), sep=";", encoding="latin1", dtype=str)
                df["TABLE"] = table
                frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def download_itr(year: int) -> pd.DataFrame | None:
    """Download ITR zip for a year."""
    url = f"{ITR_BASE}/itr_cia_aberta_{year}.zip"
    return _download_zip(url)


def download_dfp(year: int) -> pd.DataFrame | None:
    """Download DFP (annual) zip for a year."""
    url = f"{DFP_BASE}/dfp_cia_aberta_{year}.zip"
    return _download_zip(url)


# ---------------------------------------------------------------------------
# Parse and pivot
# ---------------------------------------------------------------------------
def parse_itr(raw: pd.DataFrame, cnpjs: list[str]) -> pd.DataFrame:
    """Filter by CNPJs, parse dates, extract key accounts."""
    raw["CNPJ_CIA"] = raw["CNPJ_CIA"].str.replace(r"\D", "", regex=True).str.zfill(14)

    df = raw[raw["CNPJ_CIA"].isin(cnpjs)].copy()
    found = set(df["CNPJ_CIA"].unique()) if not df.empty else set()
    missing = set(cnpjs) - found
    if missing:
        print(f"  ! CNPJs not found: {missing}")
    if found:
        print(f"  Matched: {found}")

    if df.empty:
        return pd.DataFrame()

    df["DT_REFER"] = pd.to_datetime(df["DT_REFER"], errors="coerce")
    df["DT_INI_EXERC"] = pd.to_datetime(df["DT_INI_EXERC"], errors="coerce")
    df["DT_FIM_EXERC"] = pd.to_datetime(df["DT_FIM_EXERC"], errors="coerce")

    # Only latest exercise
    df = df[df["ORDEM_EXERC"].isin(["ÚLTIMO", "ULTIMO"])]
    df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"], errors="coerce")

    df["YEAR"] = df["DT_FIM_EXERC"].dt.year
    df["MONTH"] = df["DT_FIM_EXERC"].dt.month
    df["QUARTER"] = df["MONTH"].map({3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"})
    df["PERIOD"] = df["YEAR"].astype(str) + "-" + df["QUARTER"]

    return df


def pivot_to_wide(df: pd.DataFrame, accounts: dict, table_name: str) -> pd.DataFrame:
    """Pivot ITR data to wide format: rows=CNPJ+period, cols=account names."""
    subset = df[df["TABLE"] == table_name].copy()

    def match_account(code: str) -> str | None:
        for k, v in accounts.items():
            if code == k or code.startswith(k + "."):
                return v
        return None

    subset["ACCOUNT_NAME"] = subset["CD_CONTA"].apply(match_account)
    subset = subset.dropna(subset=["ACCOUNT_NAME"])

    pivot = subset.pivot_table(
        index=["CNPJ_CIA", "DENOM_CIA", "PERIOD", "YEAR", "QUARTER"],
        columns="ACCOUNT_NAME",
        values="VL_CONTA",
        aggfunc="first",
    ).reset_index()

    return pivot


def fill_missing_with_dfp(result: pd.DataFrame, cnpjs: list[str], years: list[int]) -> pd.DataFrame:
    """For companies with no ITR data in a given year, fill from DFP annual data.

    DFP annual data is placed into Q4 of each year as a fallback.
    """
    # Find which companies are missing data
    companies_in_result = set(result["CNPJ_CIA"].unique())
    missing_companies = [c for c in cnpjs if c not in companies_in_result]

    if not missing_companies:
        # Check per-year coverage too
        for cnpj in cnpjs:
            co = result[result["CNPJ_CIA"] == cnpj]
            if co.empty:
                missing_companies.append(cnpj)
        missing_companies = list(set(missing_companies))

    if not missing_companies:
        print("  All companies have ITR data, no DFP fallback needed")
        return result

    print(f"\n  Companies without ITR: {missing_companies}")
    print("  Fetching DFP annual data as fallback...")

    # Download DFP for all years
    dfp_frames = []
    for year in years:
        print(f"    DFP {year}...", end=" ")
        dfp = download_dfp(year)
        if dfp is not None:
            dfp_frames.append(dfp)
            print(f"OK ({len(dfp)} rows)")
        else:
            print("skip")

    if not dfp_frames:
        print("  ! No DFP data available")
        return result

    raw_dfp = pd.concat(dfp_frames, ignore_index=True)
    parsed_dfp = parse_itr(raw_dfp, missing_companies)  # same parser works for DFP

    if parsed_dfp.empty:
        print("  ! No matching DFP data for missing companies")
        return result

    # Pivot DFP
    bpp_dfp = pivot_to_wide(parsed_dfp, BPP_ACCOUNTS, "BPP")
    dre_dfp = pivot_to_wide(parsed_dfp, DRE_ACCOUNTS, "DRE")

    merge_keys = ["CNPJ_CIA", "DENOM_CIA", "PERIOD", "YEAR", "QUARTER"]
    dfp_wide = bpp_dfp.merge(dre_dfp, on=merge_keys, how="outer")

    # Append DFP data to result
    result = pd.concat([result, dfp_wide], ignore_index=True)
    result = result.sort_values(["CNPJ_CIA", "YEAR", "QUARTER"])

    print(f"  Added {len(dfp_wide)} DFP rows for {len(missing_companies)} companies")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch CVM ITR quarterly data")
    parser.add_argument("--csv", help="CSV file with TICKER and/or CNPJ columns")
    parser.add_argument("--cnpjs", nargs="+", help="Direct CNPJ list")
    parser.add_argument("--output", default="cvm_itr_data.parquet", help="Output file")
    parser.add_argument("--years", nargs=2, type=int, default=[2011, 2026], help="Year range")
    args = parser.parse_args()

    # Get CNPJs
    if args.cnpjs:
        cnpjs = [c.replace(r"\D", "").zfill(14) for c in args.cnpjs]
        tickers = None
    elif args.csv:
        df = pd.read_csv(args.csv, dtype=str)
        if "CNPJ" in df.columns:
            cnpjs = df["CNPJ"].str.replace(r"\D", "", regex=True).str.zfill(14).tolist()
        elif "TICKER" in df.columns:
            print("! CSV has TICKER but no CNPJ. Run cvm-cnpj.py first to get CNPJs.")
            return
        else:
            print("! CSV must have CNPJ or TICKER column")
            return
        tickers = df["TICKER"].tolist() if "TICKER" in df.columns else None
    else:
        try:
            data = requests.get("http://localhost:3200/stocks/fundamental?fields=DY&dates=2026-06-19").json()["data"]
            df = pd.DataFrame(data)
            tickers = df["TICKER"].tolist()
            print(f"! No CNPJs provided. Use --csv or --cnpjs. Got {len(tickers)} tickers from API.")
            return
        except Exception as e:
            print(f"! Could not fetch from API: {e}")
            return

    global YEARS
    YEARS = list(range(args.years[0], args.years[1] + 1))

    print(f"Fetching ITR data for {len(cnpjs)} companies, {YEARS[0]}-{YEARS[-1]}...")

    # Download all years
    raw_frames = []
    for year in YEARS:
        print(f"  {year}...", end=" ")
        result = download_itr(year)
        if result is not None:
            raw_frames.append(result)
            print(f"OK ({len(result)} rows)")
        else:
            print("skip")

    if not raw_frames:
        print("X No ITR data downloaded")
        return

    raw = pd.concat(raw_frames, ignore_index=True)
    print(f"\nTotal raw rows: {len(raw)}")

    # Parse and filter
    parsed = parse_itr(raw, cnpjs)
    print(f"After filtering: {len(parsed)} rows for {parsed['CNPJ_CIA'].nunique()} companies")

    # Pivot to wide format
    bpp = pivot_to_wide(parsed, BPP_ACCOUNTS, "BPP")
    dre = pivot_to_wide(parsed, DRE_ACCOUNTS, "DRE")

    merge_keys = ["CNPJ_CIA", "DENOM_CIA", "PERIOD", "YEAR", "QUARTER"]
    result = bpp.merge(dre, on=merge_keys, how="outer")

    # Fill missing companies from DFP annual data
    result = fill_missing_with_dfp(result, cnpjs, YEARS)

    result = result.sort_values(["CNPJ_CIA", "YEAR", "QUARTER"])

    out_path = Path(args.output)
    if out_path.suffix == ".parquet":
        result.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        result.to_csv(out_path, index=False)
    else:
        result.to_parquet(out_path.with_suffix(".parquet"), index=False)

    print(f"\nSaved {len(result)} rows x {len(result.columns)} columns to {out_path}")
    print(f"   Companies: {result['CNPJ_CIA'].nunique()}")
    print(f"   Periods: {result['PERIOD'].min()} to {result['PERIOD'].max()}")
    print(f"   Columns: {list(result.columns)}")


if __name__ == "__main__":
    main()
