import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch

import molecule_utils


def _read_rows(path):
  return pd.read_csv(path).to_dict('records')


def _is_missing(value):
  if value is None or value == '':
    return True
  try:
    return bool(pd.isna(value))
  except TypeError:
    return False


def _write_records_csv(records, path):
  records = list(records)
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  base_fieldnames = [
    'safe',
    'source',
    'smiles',
    'canonical_smiles',
    'valid',
    'qed',
    'sa',
    'quality',
  ]
  extras = sorted({
    key
    for record in records
    for key in record.keys()
    if key not in base_fieldnames
  })
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(
      f,
      fieldnames=base_fieldnames + extras)
    writer.writeheader()
    for record in records:
      writer.writerow(record)


def _denovo_score(args):
  rows = _read_rows(args.input)
  if args.safe_col and args.safe_col in rows[0]:
    safe_strings = [row.get(args.safe_col) for row in rows]
    records = molecule_utils.evaluate_molecules(
      safe_strings=safe_strings,
      qed_threshold=args.qed_threshold,
      sa_threshold=args.sa_threshold)
  elif args.smiles_col and args.smiles_col in rows[0]:
    smiles = [row.get(args.smiles_col) for row in rows]
    records = molecule_utils.evaluate_molecules(
      smiles=smiles,
      qed_threshold=args.qed_threshold,
      sa_threshold=args.sa_threshold)
  else:
    raise ValueError(
      f'Input must contain {args.safe_col!r} or {args.smiles_col!r}.')
  summary = {
    'task': 'denovo',
    'metrics': molecule_utils.generation_metrics(records),
  }
  molecule_utils.write_json(summary, args.output)
  if args.records_output:
    _write_records_csv([record.to_dict() for record in records],
                       args.records_output)


def _fragment_score(args):
  rows = _read_rows(args.input)
  summary = {
    'task': 'fragment',
    'metrics_by_task': molecule_utils.fragment_metrics(
      rows,
      task_col=args.task_col,
      smiles_col=args.smiles_col,
      reference_col=args.reference_col),
  }
  molecule_utils.write_json(summary, args.output)


def _pmo_score(args):
  rows = _read_rows(args.input)
  summary = {
    'task': 'pmo',
    'metrics': molecule_utils.pmo_metrics(
      rows,
      score_col=args.score_col,
      higher_is_better=not args.lower_is_better),
  }
  molecule_utils.write_json(summary, args.output)


def _lead_score(args):
  rows = _read_rows(args.input)
  summary = {
    'task': 'lead_optimization',
    'metrics': molecule_utils.lead_optimization_metrics(
      rows,
      lead_col=args.lead_col,
      smiles_col=args.smiles_col,
      score_col=args.score_col,
      similarity_threshold=args.similarity_threshold,
      qed_threshold=args.qed_threshold,
      sa_threshold=args.sa_threshold),
  }
  molecule_utils.write_json(summary, args.output)


def _parse_array(value, dtype=int):
  if isinstance(value, str):
    value = value.strip()
    if value.startswith('['):
      value = json.loads(value)
    else:
      value = [item for item in value.replace(';', ',').split(',')
               if item != '']
  return [dtype(item) for item in value]


def _template_from_row(row, tokenizer, length, args):
  if (args.template_ids_col in row
      and args.known_mask_col in row
      and not _is_missing(row.get(args.template_ids_col))):
    template_ids = _parse_array(row[args.template_ids_col], dtype=int)
    known_mask = _parse_array(row[args.known_mask_col], dtype=int)
    if len(template_ids) != length:
      raise ValueError(
        f'{args.template_ids_col} must have {length} token ids.')
    if len(known_mask) != length:
      raise ValueError(
        f'{args.known_mask_col} must have {length} entries.')
    return template_ids, [bool(x) for x in known_mask]

  if (args.template_safe_col not in row
      or _is_missing(row.get(args.template_safe_col))):
    raise ValueError(
      f'Input must contain {args.template_ids_col}/{args.known_mask_col} '
      f'or {args.template_safe_col}.')
  encoded = tokenizer(
    row[args.template_safe_col],
    max_length=length,
    padding='max_length',
    truncation=True,
    add_special_tokens=True,
    return_attention_mask=True,
    return_token_type_ids=False)
  template_ids = encoded['input_ids']
  attention_mask = encoded['attention_mask']
  mask_id = tokenizer.mask_token_id
  known_mask = [
    bool(attn) and token_id != mask_id
    for token_id, attn in zip(template_ids, attention_mask)
  ]
  return template_ids, known_mask


def _load_model_from_checkpoint(checkpoint_path, device):
  import algo
  import dataloader
  from omegaconf import OmegaConf

  checkpoint = torch.load(
    checkpoint_path,
    map_location='cpu',
    weights_only=False)
  config = checkpoint['hyper_parameters']['config']
  OmegaConf.set_struct(config, False)
  if not hasattr(config.training, 'class_dropout_p'):
    config.training.class_dropout_p = 0.0
  if not hasattr(config.eval, 'gen_ppl_eval_model_name_or_path'):
    config.eval.gen_ppl_eval_model_name_or_path = None
  OmegaConf.set_struct(config, True)

  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    model_cls = algo.AR
  elif config.algo.name == 'mdlm':
    model_cls = algo.MDLM
  elif config.algo.name in {
      'duo', 'duo_base', 'distillation', 'ot-finetune'}:
    model_cls = algo.DUO_BASE
  else:
    raise ValueError(f'Unsupported algorithm: {config.algo.name}')

  model = model_cls(config, tokenizer)
  state_dict = {
    key: value
    for key, value in checkpoint['state_dict'].items()
    if not key.startswith('teacher')
  }
  model.load_state_dict(state_dict)
  if model.ema is not None and 'ema' in checkpoint:
    model.ema.load_state_dict(checkpoint['ema'])
  model.to(device)
  model.eval()
  model._eval_mode()
  return model, tokenizer


def _denovo_template(tokenizer, length):
  template = torch.full((length,), tokenizer.pad_token_id, dtype=torch.long)
  known_mask = torch.zeros((length,), dtype=torch.bool)
  template[0] = tokenizer.bos_token_id
  known_mask[0] = True
  return template, known_mask


def _record_generated_tokens(rows, token_ids, tokenizer):
  for row, ids in zip(rows, token_ids):
    raw_ids = ids.detach().cpu().tolist()
    row['raw_token_ids'] = json.dumps(raw_ids)
    row.update(molecule_utils.token_decode_metadata(raw_ids, tokenizer))


@torch.no_grad()
def _sample_denovo(args):
  device = args.device
  if device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model, tokenizer = _load_model_from_checkpoint(
    args.checkpoint,
    device=device)

  safe_strings = []
  raw_batches = []
  token_template, known_token_mask = _denovo_template(
    tokenizer, model.num_tokens)
  remaining = args.num_samples
  while remaining > 0:
    batch_size = min(args.batch_size, remaining)
    samples = model.generate_samples(
      num_samples=batch_size,
      num_steps=args.steps,
      eps=args.eps,
      token_template=token_template,
      known_token_mask=known_token_mask)
    raw_batches.append(samples.detach().cpu())
    safe_strings.extend(
      molecule_utils.decode_token_batch_to_safe(samples, tokenizer))
    remaining -= batch_size

  all_tokens = torch.cat(raw_batches, dim=0)
  records = molecule_utils.evaluate_molecules(
    safe_strings=safe_strings,
    qed_threshold=args.qed_threshold,
    sa_threshold=args.sa_threshold)
  token_diagnostics = molecule_utils.token_decode_summary(
    all_tokens, tokenizer, records)
  if token_diagnostics['first_token_is_bos_rate'] != 1.0:
    print(
      'warning: de novo sampling produced non-BOS first tokens',
      file=sys.stderr)
  summary = {
    'task': 'denovo',
    'checkpoint': args.checkpoint,
    'metrics': molecule_utils.generation_metrics(records),
    'token_diagnostics': token_diagnostics,
  }
  molecule_utils.write_json(summary, args.output)
  if args.records_output:
    rows = []
    for safe_string, record in zip(safe_strings, records):
      row = record.to_dict()
      row['safe'] = safe_string
      rows.append(row)
    _record_generated_tokens(rows, all_tokens, tokenizer)
    _write_records_csv(rows, args.records_output)


@torch.no_grad()
def _sample_template(args):
  device = args.device
  if device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model, tokenizer = _load_model_from_checkpoint(
    args.checkpoint,
    device=device)
  prompt_rows = _read_rows(args.input)

  all_rows = []
  all_record_objs = []
  all_token_batches = []
  for prompt_index, prompt_row in enumerate(prompt_rows):
    template_ids, known_mask = _template_from_row(
      prompt_row,
      tokenizer,
      model.num_tokens,
      args)
    safe_strings = []
    raw_batches = []
    remaining = args.num_samples_per_template
    while remaining > 0:
      batch_size = min(args.batch_size, remaining)
      samples = model.generate_samples(
        num_samples=batch_size,
        num_steps=args.steps,
        eps=args.eps,
        token_template=torch.tensor(template_ids),
        known_token_mask=torch.tensor(known_mask))
      raw_batches.append(samples.detach().cpu())
      safe_strings.extend(
        molecule_utils.decode_token_batch_to_safe(samples, tokenizer))
      remaining -= batch_size

    prompt_tokens = torch.cat(raw_batches, dim=0)
    all_token_batches.append(prompt_tokens)
    records = molecule_utils.evaluate_molecules(
      safe_strings=safe_strings,
      qed_threshold=args.qed_threshold,
      sa_threshold=args.sa_threshold)
    all_record_objs.extend(records)
    for safe_string, record in zip(safe_strings, records):
      row = record.to_dict()
      row['safe'] = safe_string
      row['task'] = prompt_row.get(args.task_col, 'fragment')
      row['prompt_index'] = prompt_index
      if args.reference_col in prompt_row:
        row['reference_smiles'] = prompt_row[args.reference_col]
      all_rows.append(row)
    _record_generated_tokens(all_rows[-len(records):], prompt_tokens,
                             tokenizer)

  metrics_rows = [
    {
      'task': row['task'],
      'smiles': row['canonical_smiles'],
      'reference_smiles': row.get('reference_smiles'),
    }
    for row in all_rows
  ]
  summary = {
    'task': 'fragment',
    'checkpoint': args.checkpoint,
    'metrics_by_task': molecule_utils.fragment_metrics(metrics_rows),
    'token_diagnostics': molecule_utils.token_decode_summary(
      torch.cat(all_token_batches, dim=0), tokenizer, all_record_objs),
  }
  molecule_utils.write_json(summary, args.output)
  if args.records_output:
    _write_records_csv(all_rows, args.records_output)


def _add_quality_args(parser):
  parser.add_argument('--qed-threshold', type=float, default=0.6)
  parser.add_argument('--sa-threshold', type=float, default=4.0)


def _build_parser():
  parser = argparse.ArgumentParser(
    description='SAFE/SMILES benchmark utilities for Duo molecules.')
  subparsers = parser.add_subparsers(dest='command', required=True)

  denovo = subparsers.add_parser(
    'score-denovo',
    help='Score existing SAFE/SMILES generations.')
  denovo.add_argument('--input', required=True)
  denovo.add_argument('--output', required=True)
  denovo.add_argument('--records-output')
  denovo.add_argument('--safe-col', default='safe')
  denovo.add_argument('--smiles-col', default='smiles')
  _add_quality_args(denovo)
  denovo.set_defaults(func=_denovo_score)

  fragment = subparsers.add_parser(
    'score-fragment',
    help='Score fragment-constrained generation CSVs by task.')
  fragment.add_argument('--input', required=True)
  fragment.add_argument('--output', required=True)
  fragment.add_argument('--task-col', default='task')
  fragment.add_argument('--smiles-col', default='smiles')
  fragment.add_argument('--reference-col', default='reference_smiles')
  fragment.set_defaults(func=_fragment_score)

  pmo = subparsers.add_parser(
    'score-pmo',
    help='Score PMO hit-generation CSVs.')
  pmo.add_argument('--input', required=True)
  pmo.add_argument('--output', required=True)
  pmo.add_argument('--score-col', default='score')
  pmo.add_argument('--lower-is-better', action='store_true')
  pmo.set_defaults(func=_pmo_score)

  lead = subparsers.add_parser(
    'score-lead',
    help='Score lead-optimization CSVs.')
  lead.add_argument('--input', required=True)
  lead.add_argument('--output', required=True)
  lead.add_argument('--lead-col', default='lead_smiles')
  lead.add_argument('--smiles-col', default='smiles')
  lead.add_argument('--score-col', default='score')
  lead.add_argument('--similarity-threshold', type=float, default=0.4)
  _add_quality_args(lead)
  lead.set_defaults(func=_lead_score)

  sample = subparsers.add_parser(
    'sample-denovo',
    help='Sample de novo molecules from a Duo checkpoint and score them.')
  sample.add_argument('--checkpoint', required=True)
  sample.add_argument('--output', required=True)
  sample.add_argument('--records-output')
  sample.add_argument('--num-samples', type=int, default=100)
  sample.add_argument('--batch-size', type=int, default=32)
  sample.add_argument('--steps', type=int, default=1000)
  sample.add_argument('--eps', type=float, default=1e-5)
  sample.add_argument('--device', default='auto')
  _add_quality_args(sample)
  sample.set_defaults(func=_sample_denovo)

  sample_template = subparsers.add_parser(
    'sample-template',
    help='Sample from token templates while preserving known tokens.')
  sample_template.add_argument('--checkpoint', required=True)
  sample_template.add_argument('--input', required=True)
  sample_template.add_argument('--output', required=True)
  sample_template.add_argument('--records-output')
  sample_template.add_argument('--num-samples-per-template',
                               type=int,
                               default=100)
  sample_template.add_argument('--batch-size', type=int, default=32)
  sample_template.add_argument('--steps', type=int, default=1000)
  sample_template.add_argument('--eps', type=float, default=1e-5)
  sample_template.add_argument('--device', default='auto')
  sample_template.add_argument('--template-safe-col',
                               default='template_safe')
  sample_template.add_argument('--template-ids-col',
                               default='template_ids')
  sample_template.add_argument('--known-mask-col',
                               default='known_token_mask')
  sample_template.add_argument('--task-col', default='task')
  sample_template.add_argument('--reference-col',
                               default='reference_smiles')
  _add_quality_args(sample_template)
  sample_template.set_defaults(func=_sample_template)

  return parser


def main():
  args = _build_parser().parse_args()
  args.func(args)


if __name__ == '__main__':
  main()
