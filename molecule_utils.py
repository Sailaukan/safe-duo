import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch


QED_THRESHOLD = 0.6
SA_THRESHOLD = 4.0
SAFE_GPT_V1_REVISION = 'b83175cd7394'


def _require_rdkit():
  try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, QED
    from rdkit.Contrib.SA_Score import sascorer
  except ImportError as exc:
    raise ImportError(
      'Molecule metrics require rdkit. Install it with '
      '`pip install rdkit`.') from exc
  RDLogger.DisableLog('rdApp.*')
  return Chem, DataStructs, AllChem, QED, sascorer


def _require_safe():
  try:
    import safe
  except ImportError as exc:
    raise ImportError(
      'SAFE decoding requires safe-mol. Install it with '
      '`pip install safe-mol`.') from exc
  return safe


def load_safe_tokenizer(revision: str = SAFE_GPT_V1_REVISION):
  try:
    from safe.tokenizer import SAFETokenizer
  except ImportError as exc:
    raise ImportError(
      'SAFE tokenization requires safe-mol. Install it with '
      '`pip install safe-mol`.') from exc
  return SAFETokenizer.from_pretrained(
    'datamol-io/safe-gpt',
    revision=revision).get_pretrained()


def _to_id_list(token_ids):
  if isinstance(token_ids, torch.Tensor):
    token_ids = token_ids.detach().cpu().tolist()
  return [int(token_id) for token_id in token_ids]


def _tokenizer_id(tokenizer, name: str) -> Optional[int]:
  token_id = getattr(tokenizer, name, None)
  if token_id is None:
    return None
  return int(token_id)


def trim_token_ids_for_molecule_decode(token_ids, tokenizer) -> list[int]:
  """Keep only the generated molecule span before EOS/padding.

  SAFE-GPT V1 training examples are `[CLS] safe [SEP] [PAD]...`, and pad
  positions are usually loss-masked. Decoding the full fixed-length sample can
  append arbitrary post-EOS tokens to an otherwise valid molecule.
  """
  ids = _to_id_list(token_ids)
  stop_ids = {
    token_id for token_id in (
      _tokenizer_id(tokenizer, 'eos_token_id'),
      _tokenizer_id(tokenizer, 'sep_token_id'),
      _tokenizer_id(tokenizer, 'pad_token_id'),
    )
    if token_id is not None
  }
  for idx, token_id in enumerate(ids):
    if token_id in stop_ids:
      return ids[:idx]
  return ids


def token_decode_metadata(token_ids, tokenizer) -> dict:
  ids = _to_id_list(token_ids)

  def first_position(*names):
    wanted = {
      token_id for token_id in (
        _tokenizer_id(tokenizer, name) for name in names)
      if token_id is not None
    }
    for idx, token_id in enumerate(ids):
      if token_id in wanted:
        return idx
    return None

  return {
    'raw_length': len(ids),
    'first_token_id': ids[0] if ids else None,
    'first_token_is_bos': (
      ids[0] == _tokenizer_id(tokenizer, 'bos_token_id')
      if ids and _tokenizer_id(tokenizer, 'bos_token_id') is not None
      else False),
    'decoded_length': len(trim_token_ids_for_molecule_decode(ids, tokenizer)),
    'eos_position': first_position('eos_token_id', 'sep_token_id'),
    'pad_position': first_position('pad_token_id'),
  }


def decode_token_ids_to_safe(token_ids, tokenizer) -> str:
  return tokenizer.decode(
    trim_token_ids_for_molecule_decode(token_ids, tokenizer),
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False)


def decode_token_batch_to_safe(token_ids, tokenizer) -> list[str]:
  if isinstance(token_ids, torch.Tensor):
    token_ids = token_ids.detach().cpu().tolist()
  return [
    decode_token_ids_to_safe(ids, tokenizer)
    for ids in token_ids
  ]


def safe_to_smiles(safe_string: str,
                   canonical: bool = True) -> Optional[str]:
  if not isinstance(safe_string, str) or safe_string == '':
    return None
  safe = _require_safe()
  try:
    smiles = safe.decode(
      safe_string,
      canonical=canonical,
      ignore_errors=True)
  except Exception:
    return None
  if not smiles:
    return None
  return canonicalize_smiles(smiles)


def canonicalize_smiles(smiles: str) -> Optional[str]:
  if not isinstance(smiles, str) or smiles == '':
    return None
  Chem, _, _, _, _ = _require_rdkit()
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return None
  try:
    Chem.SanitizeMol(mol)
  except Exception:
    return None
  return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def decode_token_ids_to_smiles(token_ids, tokenizer) -> Optional[str]:
  return safe_to_smiles(decode_token_ids_to_safe(token_ids, tokenizer))


def _mol_from_smiles(smiles: str):
  Chem, _, _, _, _ = _require_rdkit()
  if not isinstance(smiles, str) or smiles == '':
    return None
  return Chem.MolFromSmiles(smiles)


def _fingerprint(smiles: str):
  _, _, AllChem, _, _ = _require_rdkit()
  mol = _mol_from_smiles(smiles)
  if mol is None:
    return None
  return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> Optional[float]:
  _, DataStructs, _, _, _ = _require_rdkit()
  fp_a = _fingerprint(smiles_a)
  fp_b = _fingerprint(smiles_b)
  if fp_a is None or fp_b is None:
    return None
  return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def internal_diversity(smiles_list: Iterable[str]) -> float:
  _, DataStructs, _, _, _ = _require_rdkit()
  fingerprints = [
    fp for fp in (_fingerprint(smiles) for smiles in smiles_list)
    if fp is not None
  ]
  if len(fingerprints) < 2:
    return 0.0
  distances = []
  for i, fp in enumerate(fingerprints[:-1]):
    similarities = DataStructs.BulkTanimotoSimilarity(
      fp, fingerprints[i + 1:])
    distances.extend(1.0 - sim for sim in similarities)
  return float(np.mean(distances)) if distances else 0.0


def qed_score(smiles: str) -> Optional[float]:
  _, _, _, QED, _ = _require_rdkit()
  mol = _mol_from_smiles(smiles)
  if mol is None:
    return None
  return float(QED.qed(mol))


def sa_score(smiles: str) -> Optional[float]:
  _, _, _, _, sascorer = _require_rdkit()
  mol = _mol_from_smiles(smiles)
  if mol is None:
    return None
  return float(sascorer.calculateScore(mol))


@dataclass
class MoleculeEvaluation:
  source: Optional[str]
  smiles: Optional[str]
  canonical_smiles: Optional[str]
  valid: bool
  qed: Optional[float]
  sa: Optional[float]
  quality: bool

  def to_dict(self):
    return {
      'source': self.source,
      'smiles': self.smiles,
      'canonical_smiles': self.canonical_smiles,
      'valid': self.valid,
      'qed': self.qed,
      'sa': self.sa,
      'quality': self.quality,
    }


def evaluate_molecules(safe_strings: Optional[Iterable[str]] = None,
                       smiles: Optional[Iterable[str]] = None,
                       qed_threshold: float = QED_THRESHOLD,
                       sa_threshold: float = SA_THRESHOLD
                       ) -> list[MoleculeEvaluation]:
  if safe_strings is None and smiles is None:
    raise ValueError('Provide safe_strings or smiles.')
  if safe_strings is not None and smiles is not None:
    raise ValueError('Provide only one of safe_strings or smiles.')

  records = []
  source_values = safe_strings if safe_strings is not None else smiles
  for source in source_values:
    if safe_strings is not None:
      canonical = safe_to_smiles(source)
      raw_smiles = canonical
    else:
      raw_smiles = source
      canonical = canonicalize_smiles(source)
    qed = qed_score(canonical) if canonical is not None else None
    sa = sa_score(canonical) if canonical is not None else None
    quality = (
      canonical is not None
      and qed is not None
      and sa is not None
      and qed >= qed_threshold
      and sa <= sa_threshold)
    records.append(MoleculeEvaluation(
      source=source,
      smiles=raw_smiles,
      canonical_smiles=canonical,
      valid=canonical is not None,
      qed=qed,
      sa=sa,
      quality=quality))
  return records


def generation_metrics(records: Iterable[MoleculeEvaluation]) -> dict:
  records = list(records)
  total = len(records)
  valid = [record for record in records if record.valid]
  canonical = [record.canonical_smiles for record in valid]
  unique = sorted(set(canonical))
  qed_values = [record.qed for record in valid if record.qed is not None]
  sa_values = [record.sa for record in valid if record.sa is not None]
  quality_count = sum(record.quality for record in records)
  return {
    'num_samples': total,
    'num_valid': len(valid),
    'num_unique': len(unique),
    'validity': _safe_div(len(valid), total),
    'uniqueness': _safe_div(len(unique), len(valid)),
    'diversity': internal_diversity(canonical),
    'quality': _safe_div(quality_count, total),
    'quality_valid': _safe_div(quality_count, len(valid)),
    'qed_mean': _mean(qed_values),
    'sa_mean': _mean(sa_values),
  }


def fragment_metrics(rows: Iterable[dict],
                     task_col: str = 'task',
                     smiles_col: str = 'smiles',
                     reference_col: str = 'reference_smiles') -> dict:
  grouped = {}
  for row in rows:
    grouped.setdefault(row.get(task_col, 'all'), []).append(row)

  out = {}
  for task, task_rows in grouped.items():
    smiles = [row.get(smiles_col) for row in task_rows]
    records = evaluate_molecules(smiles=smiles)
    metrics = generation_metrics(records)
    refs = [row.get(reference_col) for row in task_rows]
    if any(refs):
      distances = []
      for record, ref in zip(records, refs):
        if not record.valid or not ref:
          continue
        sim = tanimoto_similarity(record.canonical_smiles, ref)
        if sim is not None:
          distances.append(1.0 - sim)
      metrics['distance'] = _mean(distances)
    else:
      metrics['distance'] = math.nan
    out[task] = metrics
  return out


def pmo_metrics(rows: Iterable[dict],
                score_col: str = 'score',
                higher_is_better: bool = True) -> dict:
  scores = [
    float(row[score_col])
    for row in rows
    if row.get(score_col) not in (None, '')
  ]
  scores = [score for score in scores if not math.isnan(score)]
  if not scores:
    return {
      'num_samples': 0,
      'top_1': math.nan,
      'top_10': math.nan,
      'top_100': math.nan,
      'auc': math.nan,
    }
  ordered = sorted(scores, reverse=higher_is_better)
  eval_order = scores if higher_is_better else [-score for score in scores]
  best_so_far = np.maximum.accumulate(eval_order)
  if len(best_so_far) == 1:
    auc = float(best_so_far[0])
  else:
    auc = float(np.trapz(best_so_far) / (len(best_so_far) - 1))
  if not higher_is_better:
    auc = -auc
  return {
    'num_samples': len(scores),
    'top_1': _mean(ordered[:1]),
    'top_10': _mean(ordered[:10]),
    'top_100': _mean(ordered[:100]),
    'auc': auc,
  }


def lead_optimization_metrics(rows: Iterable[dict],
                              lead_col: str = 'lead_smiles',
                              smiles_col: str = 'smiles',
                              score_col: str = 'score',
                              similarity_threshold: float = 0.4,
                              qed_threshold: float = QED_THRESHOLD,
                              sa_threshold: float = SA_THRESHOLD) -> dict:
  records = []
  for row in rows:
    canonical = canonicalize_smiles(row.get(smiles_col))
    lead = canonicalize_smiles(row.get(lead_col))
    if canonical is None:
      continue
    qed = qed_score(canonical)
    sa = sa_score(canonical)
    similarity = (
      tanimoto_similarity(canonical, lead)
      if lead is not None else None)
    score = row.get(score_col)
    score = float(score) if score not in (None, '') else math.nan
    passed = (
      similarity is not None
      and similarity >= similarity_threshold
      and qed is not None
      and sa is not None
      and qed >= qed_threshold
      and sa <= sa_threshold)
    records.append({
      'smiles': canonical,
      'lead_smiles': lead,
      'score': score,
      'similarity': similarity,
      'qed': qed,
      'sa': sa,
      'passed_filters': passed,
    })

  passed_records = [record for record in records
                    if record['passed_filters']]
  unique = sorted(set(record['smiles'] for record in records))
  scored = [record for record in records
            if not math.isnan(record['score'])]
  top = max(scored, key=lambda row: row['score']) if scored else None
  return {
    'num_samples': len(records),
    'num_passed_filters': len(passed_records),
    'uniqueness': _safe_div(len(unique), len(records)),
    'success_rate': _safe_div(len(passed_records), len(records)),
    'top_score': top['score'] if top else math.nan,
    'top_molecule': top['smiles'] if top else None,
    'records': records,
  }


def token_decode_summary(token_ids_batch, tokenizer,
                         records: Optional[Iterable[
                           MoleculeEvaluation]] = None) -> dict:
  if isinstance(token_ids_batch, torch.Tensor):
    token_ids_batch = token_ids_batch.detach().cpu().tolist()
  token_ids_batch = list(token_ids_batch)
  metadata = [
    token_decode_metadata(token_ids, tokenizer)
    for token_ids in token_ids_batch
  ]
  decoded_lengths = [item['decoded_length'] for item in metadata]
  eos_found = [item['eos_position'] is not None for item in metadata]
  pad_found = [item['pad_position'] is not None for item in metadata]
  first_is_bos = [item['first_token_is_bos'] for item in metadata]

  summary = {
    'num_samples': len(metadata),
    'first_token_is_bos_rate': _safe_div(sum(first_is_bos), len(metadata)),
    'eos_found_rate': _safe_div(sum(eos_found), len(metadata)),
    'pad_found_rate': _safe_div(sum(pad_found), len(metadata)),
    'decoded_length_mean': _mean(decoded_lengths),
    'decoded_length_median': _percentile(decoded_lengths, 50),
    'decoded_length_p90': _percentile(decoded_lengths, 90),
  }
  if records is not None:
    records = list(records)
    summary['validity_by_decoded_length_bucket'] = (
      _validity_by_decoded_length_bucket(decoded_lengths, records))
  return summary


def _validity_by_decoded_length_bucket(decoded_lengths, records):
  buckets = {
    '0': lambda value: value == 0,
    '1-32': lambda value: 1 <= value <= 32,
    '33-64': lambda value: 33 <= value <= 64,
    '65-128': lambda value: 65 <= value <= 128,
    '129-192': lambda value: 129 <= value <= 192,
    '193-256': lambda value: 193 <= value <= 256,
    '>256': lambda value: value > 256,
  }
  out = {}
  for name, predicate in buckets.items():
    bucket_records = [
      record for length, record in zip(decoded_lengths, records)
      if predicate(length)
    ]
    out[name] = {
      'num_samples': len(bucket_records),
      'num_valid': sum(record.valid for record in bucket_records),
      'validity': _safe_div(
        sum(record.valid for record in bucket_records),
        len(bucket_records)),
    }
  return out


def write_json(data, path: str) -> None:
  parent = os.path.dirname(path)
  if parent:
    os.makedirs(parent, exist_ok=True)
  with open(path, 'w') as f:
    json.dump(_json_sanitize(data), f, indent=2)


def _safe_div(num, denom):
  return float(num / denom) if denom else 0.0


def _mean(values):
  values = [value for value in values if value is not None]
  return float(np.mean(values)) if values else math.nan


def _percentile(values, percentile):
  values = [value for value in values if value is not None]
  return float(np.percentile(values, percentile)) if values else math.nan


def _json_sanitize(value):
  if isinstance(value, dict):
    return {k: _json_sanitize(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_json_sanitize(v) for v in value]
  if isinstance(value, float) and math.isnan(value):
    return None
  return value
