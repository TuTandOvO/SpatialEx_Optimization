#!/usr/bin/env python3
"""
HumanBase 乳腺组织特异性 PPI 网络下载器
=========================================
从 HumanBase API 批量下载 mammary gland 组织的基因-基因功能关联网络，
构建 G×G 的 PPI 权重矩阵并保存为本地文件。

数据来源：
  HumanBase (Flatiron Institute) — GIANT tissue-specific functional networks
  API 文档：https://humanbase.net/api/docs/
  论文：Greene et al., Nature Genetics 2015

API Endpoint:
  GET https://humanbase.net/api/integrations/mammary-gland/network/
  参数：entrez=<id>&...&giant_version=v1&node_size=<n>&format=json

输出文件：
  ppi_matrix.npy           — [G, G] float32 PPI 权重矩阵
  ppi_gene_names.npy       — [G] 基因名称（与矩阵行列对齐）
  ppi_entrez_ids.npy       — [G] Entrez ID
  gene_symbol_to_entrez.json — symbol→entrez 映射

用法：
  # 方式1：从 adata 自动提取基因列表
  python download_ppi_network.py --adata /path/to/adata.h5ad

  # 方式2：手动指定基因列表文件（每行一个 gene symbol）
  python download_ppi_network.py --gene-list genes.txt

  # 方式3：在 HPC 上从 Xenium 数据提取（乳腺癌 → mammary-gland）
  python download_ppi_network.py --xenium-h5 /path/to/cell_feature_matrix.h5 --obs-csv /path/to/cells.csv

  # 方式4：Skin Melanoma 数据（使用 --tissue skin）
  python download_ppi_network.py \
    --xenium-h5 /gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/Human_Skin_Melanoma_Base_FFPE/cell_feature_matrix.h5 \
    --obs-csv /gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/Human_Skin_Melanoma_Base_FFPE/cells.csv \
    --tissue skin \
    --output-dir /gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/ppi_data_skin/

HumanBase 可用的组织 slug（已验证）：
  - mammary-gland  — 乳腺组织（用于 Breast Cancer 数据）
  - skin           — 皮肤组织（用于 Skin Melanoma 数据）
  - epidermis      — 表皮组织
  - keratinocyte   — 角质形成细胞
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================
# 1. Gene Symbol → Entrez ID 映射
# =============================================

def map_symbols_to_entrez_via_mygene(gene_symbols: list[str]) -> dict[str, int]:
    """使用 mygene 包将 gene symbols 映射到 Entrez ID。

    Args:
        gene_symbols: 基因 symbol 列表，如 ['ESR1', 'BRCA1', ...]

    Returns:
        dict: {symbol: entrez_id}，未找到的基因被跳过。
    """
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        results = mg.querymany(gene_symbols, scopes='symbol', fields='entrezgene',
                               species='human', returnall=True)
        mapping = {}
        for hit in results['out']:
            if 'entrezgene' in hit and 'query' in hit:
                mapping[hit['query']] = int(hit['entrezgene'])
        logger.info("mygene: mapped %d / %d symbols to Entrez IDs",
                     len(mapping), len(gene_symbols))
        return mapping
    except ImportError:
        logger.warning("mygene not installed, falling back to NCBI query")
        return {}


def map_symbols_to_entrez_via_ncbi(gene_symbols: list[str],
                                    batch_size: int = 5) -> dict[str, int]:
    """通过 NCBI E-utilities API 将 gene symbols 映射到 Entrez ID（无需额外包）。

    逐个基因查询（或小批次），避免 URL 过长导致 414 错误。

    Args:
        gene_symbols: 基因 symbol 列���。
        batch_size: 每次 API 请求的基因数（默认 5，避免 URL 过长）。

    Returns:
        dict: {symbol: entrez_id}
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    mapping = {}
    total = len(gene_symbols)

    for i in range(0, total, batch_size):
        batch = gene_symbols[i:i + batch_size]
        # 每个基因用 (symbol[Gene Name] AND Homo sapiens[Organism]) 包裹
        terms = [f'({s}[Gene Name] AND Homo sapiens[Organism])' for s in batch]
        query = ' OR '.join(terms)
        encoded_query = urllib.parse.quote(query)
        url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
               f"db=gene&term={encoded_query}&retmax={len(batch)*3}&retmode=xml")

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml_text = resp.read().decode()
            root = ET.fromstring(xml_text)
            ids = [id_elem.text for id_elem in root.findall('.//Id')]

            if ids:
                id_str = ','.join(ids[:50])  # 限制 summary 请求大小
                summary_url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
                               f"db=gene&id={id_str}&retmode=json")
                with urllib.request.urlopen(summary_url, timeout=30) as resp:
                    summary = json.loads(resp.read().decode())

                batch_set = set(batch)
                for gid, info in summary.get('result', {}).items():
                    if gid == 'uids':
                        continue
                    name = info.get('name', '')
                    if name in batch_set:
                        mapping[name] = int(gid)
        except Exception as e:
            logger.warning("NCBI batch %d-%d failed: %s", i, i + len(batch), e)

        # 进度日志（每 50 个基因打印一次）
        if (i + batch_size) % 50 < batch_size:
            logger.info("NCBI progress: %d / %d genes queried, %d mapped so far",
                         min(i + batch_size, total), total, len(mapping))

        time.sleep(0.35)  # NCBI rate limit: ~3 requests/sec without API key

    logger.info("NCBI: mapped %d / %d symbols to Entrez IDs",
                 len(mapping), len(gene_symbols))
    return mapping


def get_symbol_to_entrez(gene_symbols: list[str],
                          cache_path: str | None = None) -> dict[str, int]:
    """获取 gene symbol → Entrez ID 映射（带缓存）。

    优先从缓存加载，否则依次尝试 mygene → NCBI。

    Args:
        gene_symbols: 基因 symbol 列表。
        cache_path: 缓存文件路径。

    Returns:
        dict: {symbol: entrez_id}
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        cached_int = {k: int(v) for k, v in cached.items() if v and int(v) > 0}
        missing = [s for s in gene_symbols if s not in cached_int]
        if not missing and len(cached_int) > 0:
            logger.info("Loaded %d mappings from cache (all genes covered)", len(cached_int))
            return cached_int
        elif len(cached_int) > 0:
            logger.info("Cache has %d mappings, %d genes still missing", len(cached_int), len(missing))
    else:
        cached_int = {}
        missing = gene_symbols

    # Try mygene first
    mapping = map_symbols_to_entrez_via_mygene(missing)
    if len(mapping) < len(missing) * 0.5:
        # Fallback to NCBI
        ncbi_map = map_symbols_to_entrez_via_ncbi(
            [s for s in missing if s not in mapping]
        )
        mapping.update(ncbi_map)

    full_mapping = {**cached_int, **mapping}

    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(full_mapping, f, indent=2)
        logger.info("Saved %d mappings to cache: %s", len(full_mapping), cache_path)

    return full_mapping


# =============================================
# 2. HumanBase PPI 网络下载
# =============================================

def query_humanbase_network(
    entrez_ids: list[int],
    tissue_slug: str = 'mammary-gland',
    giant_version: str = 'v1',
    node_size: int | None = None,
    timeout: int = 120,
    max_retries: int = 3,
) -> tuple[list[dict], list[dict]]:
    """查询 HumanBase tissue-specific 功能网络。

    Args:
        entrez_ids: Entrez Gene ID 列表。
        tissue_slug: 组织 slug（如 'mammary-gland'）。
        giant_version: GIANT 版本（'v1' 返回 edges，'v3' 通常不返回）。
        node_size: 子网络大小。None 时取 len(entrez_ids)。
        timeout: 请求超时秒数。
        max_retries: 失败后最大重试次数。

    Returns:
        (genes_list, edges_list)
    """
    import urllib.request
    import urllib.parse

    if node_size is None:
        node_size = len(entrez_ids)

    params = [('giant_version', giant_version), ('node_size', str(node_size)),
              ('format', 'json')]
    for eid in entrez_ids:
        params.append(('entrez', str(eid)))

    url = (f"https://humanbase.net/api/integrations/{tissue_slug}/network/?"
           + urllib.parse.urlencode(params))

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            genes = data.get('genes', [])
            edges = data.get('edges', [])
            return genes, edges
        except Exception as e:
            logger.warning("HumanBase query attempt %d/%d failed: %s",
                           attempt, max_retries, e)
            if attempt < max_retries:
                wait = 5 * attempt  # 5s, 10s, 15s
                logger.info("  Retrying in %ds...", wait)
                time.sleep(wait)
            else:
                logger.error("HumanBase query failed after %d attempts", max_retries)
                return [], []


def download_ppi_matrix(
    gene_symbols: list[str],
    symbol_to_entrez: dict[str, int],
    tissue_slug: str = 'mammary-gland',
    batch_size: int = 80,
    output_dir: str = '.',
) -> np.ndarray:
    """分批从 HumanBase 下载 PPI 网络，合并为 G×G 权重矩阵。

    策略：
      - 将 G 个基因分成 batch_size 大小的批次
      - 对每个批次查询 API 获取 edges
      - 同时查询相邻批次的交叉 edges（overlapping batches）
      - 合并所有 edges 到 G×G 矩阵

    Args:
        gene_symbols: 全部基因 symbol（与表达矩阵列对齐）。
        symbol_to_entrez: symbol → Entrez ID 映射。
        tissue_slug: HumanBase 组织 slug。
        batch_size: 每批查询的基因数。
        output_dir: 输出目录。

    Returns:
        [G, G] float32 PPI 权重矩阵（对称，对角线为 0）。
    """
    G = len(gene_symbols)
    ppi_matrix = np.zeros((G, G), dtype=np.float32)

    # 只取有 Entrez ID 的基因
    valid_genes = [(i, sym, symbol_to_entrez[sym])
                   for i, sym in enumerate(gene_symbols)
                   if sym in symbol_to_entrez]
    logger.info("Valid genes with Entrez ID: %d / %d", len(valid_genes), G)

    if len(valid_genes) == 0:
        logger.error("No genes could be mapped to Entrez IDs!")
        return ppi_matrix

    # Entrez ID → 矩阵索引映射
    entrez_to_idx = {}
    for mat_idx, sym, eid in valid_genes:
        entrez_to_idx[eid] = mat_idx

    # 分批查询
    all_entrez = [eid for _, _, eid in valid_genes]
    n_batches = (len(all_entrez) + batch_size - 1) // batch_size
    total_edges = 0

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(all_entrez))
        batch_entrez = all_entrez[start:end]

        logger.info("Batch %d/%d: querying %d genes (entrez %d..%d)...",
                     b + 1, n_batches, len(batch_entrez),
                     batch_entrez[0], batch_entrez[-1])

        genes, edges = query_humanbase_network(
            batch_entrez, tissue_slug=tissue_slug,
            node_size=len(batch_entrez),
        )

        if not edges:
            logger.warning("  No edges returned for batch %d", b + 1)
            continue

        # 建立 API 返回基因索引 → Entrez ID 映射
        api_idx_to_entrez = {}
        for api_idx, g in enumerate(genes):
            api_idx_to_entrez[api_idx] = g['entrez']

        # 将 edges 填入矩阵
        batch_count = 0
        for e in edges:
            src_entrez = api_idx_to_entrez.get(e['source'])
            tgt_entrez = api_idx_to_entrez.get(e['target'])
            if src_entrez in entrez_to_idx and tgt_entrez in entrez_to_idx:
                i = entrez_to_idx[src_entrez]
                j = entrez_to_idx[tgt_entrez]
                w = float(e['weight'])
                ppi_matrix[i, j] = max(ppi_matrix[i, j], w)
                ppi_matrix[j, i] = max(ppi_matrix[j, i], w)
                batch_count += 1

        total_edges += batch_count
        logger.info("  Got %d genes, %d edges (%d mapped to matrix)",
                     len(genes), len(edges), batch_count)

        time.sleep(1.0)  # API 速率限制

    # 跨批次查询（相邻批次之间的 edges）
    logger.info("Querying cross-batch edges...")
    for b in range(n_batches - 1):
        s1 = b * batch_size
        e1 = min(s1 + batch_size, len(all_entrez))
        s2 = (b + 1) * batch_size
        e2 = min(s2 + batch_size, len(all_entrez))

        # 取两个批次的边界基因
        cross_genes = all_entrez[max(0, e1 - 20):e1] + all_entrez[s2:min(s2 + 20, e2)]
        if len(cross_genes) < 4:
            continue

        genes, edges = query_humanbase_network(
            cross_genes, tissue_slug=tissue_slug,
            node_size=len(cross_genes),
        )

        api_idx_to_entrez = {i: g['entrez'] for i, g in enumerate(genes)}
        for e in edges:
            src_entrez = api_idx_to_entrez.get(e['source'])
            tgt_entrez = api_idx_to_entrez.get(e['target'])
            if src_entrez in entrez_to_idx and tgt_entrez in entrez_to_idx:
                i = entrez_to_idx[src_entrez]
                j = entrez_to_idx[tgt_entrez]
                w = float(e['weight'])
                ppi_matrix[i, j] = max(ppi_matrix[i, j], w)
                ppi_matrix[j, i] = max(ppi_matrix[j, i], w)
                total_edges += 1

        time.sleep(1.0)

    np.fill_diagonal(ppi_matrix, 0)
    n_nonzero = np.count_nonzero(ppi_matrix)
    logger.info("PPI matrix: %d×%d, %d nonzero edges (density=%.4f)",
                 G, G, n_nonzero // 2, n_nonzero / (G * G))
    logger.info("Weight range: [%.4f, %.4f]",
                 ppi_matrix[ppi_matrix > 0].min() if n_nonzero > 0 else 0,
                 ppi_matrix.max())

    return ppi_matrix


# =============================================
# 3. 主函数
# =============================================

def main():
    parser = argparse.ArgumentParser(description="Download HumanBase tissue-specific PPI network")
    parser.add_argument('--adata', type=str, help='Path to .h5ad file (extract var_names)')
    parser.add_argument('--xenium-h5', type=str, help='Path to Xenium cell_feature_matrix.h5')
    parser.add_argument('--obs-csv', type=str, help='Path to Xenium cells.csv')
    parser.add_argument('--gene-list', type=str, help='Path to gene list file (one per line)')
    parser.add_argument('--tissue', type=str, default='mammary-gland',
                        help='HumanBase tissue slug (default: mammary-gland)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for PPI matrix files')
    parser.add_argument('--batch-size', type=int, default=30,
                        help='Genes per API batch (default: 30, reduce if API times out)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 获取基因列表
    gene_symbols = None

    if args.adata:
        import anndata as ad
        adata = ad.read_h5ad(args.adata)
        gene_symbols = list(adata.var_names)
        logger.info("Loaded %d genes from %s", len(gene_symbols), args.adata)

    elif args.xenium_h5:
        sys.path.insert(0, '/gpfsdata/home/renyixiang/YuanLab/SpatialEx')
        import SpatialEx as se
        adata = se.pp.Read_Xenium(args.xenium_h5, args.obs_csv)
        adata = se.pp.Preprocess_adata(adata)
        gene_symbols = list(adata.var_names)
        logger.info("Loaded %d genes from Xenium data", len(gene_symbols))

    elif args.gene_list:
        with open(args.gene_list) as f:
            gene_symbols = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d genes from %s", len(gene_symbols), args.gene_list)

    else:
        parser.error("Must specify one of: --adata, --xenium-h5, --gene-list")

    # Gene symbol → Entrez ID
    cache_path = os.path.join(args.output_dir, 'gene_symbol_to_entrez.json')
    symbol_to_entrez = get_symbol_to_entrez(gene_symbols, cache_path=cache_path)

    n_mapped = sum(1 for s in gene_symbols if s in symbol_to_entrez)
    logger.info("Mapped %d / %d gene symbols to Entrez IDs", n_mapped, len(gene_symbols))

    # Download PPI
    ppi_matrix = download_ppi_matrix(
        gene_symbols, symbol_to_entrez,
        tissue_slug=args.tissue,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    # Save
    np.save(os.path.join(args.output_dir, 'ppi_matrix.npy'), ppi_matrix)
    np.save(os.path.join(args.output_dir, 'ppi_gene_names.npy'),
            np.array(gene_symbols))

    entrez_arr = np.array([symbol_to_entrez.get(s, -1) for s in gene_symbols])
    np.save(os.path.join(args.output_dir, 'ppi_entrez_ids.npy'), entrez_arr)

    logger.info("Saved PPI data to %s", args.output_dir)
    logger.info("  ppi_matrix.npy       — [%d, %d]", *ppi_matrix.shape)
    logger.info("  ppi_gene_names.npy   — [%d]", len(gene_symbols))
    logger.info("  ppi_entrez_ids.npy   — [%d]", len(entrez_arr))
    logger.info("  gene_symbol_to_entrez.json")


if __name__ == '__main__':
    main()
