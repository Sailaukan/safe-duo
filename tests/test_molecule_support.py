import os

import numpy as np
import pytest
import torch

OmegaConf = pytest.importorskip('omegaconf').OmegaConf

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


class _Section:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


def _tokenizer_config():
  return _Section(
    data=_Section(tokenizer_name_or_path='safe-gpt'),
    algo=_Section(name='duo'))


def _tiny_config(tmp_path, algo_name='duo_base'):
  algo_cfg = {
    'name': algo_name,
    'backbone': 'dit',
    'parameterization': 'mean',
    'time_conditioning': True,
    'T': 0,
    'subs_masking': False,
    'causal_attention': False,
    'ignore_bos': False,
    'loss_type': 'elbo',
  }
  if algo_name == 'duo':
    np.save(tmp_path / 'poly3.npy', np.array([1.0, -0.99, 0.0, 0.0]))
    algo_cfg['curriculum'] = {
      'mode': 'poly3',
      'gumbel_tau_log10_start': -1.0,
      'gumbel_tau_log10_end': -2.0,
      'start': 100000,
      'end': 200000,
      'gamma_min': -3.5,
      'gamma_max': -1.75,
      'integral_cache_path': '',
      'top_k': 8,
      'cache_dir': str(tmp_path),
      'n_series_terms': 4,
    }
  return OmegaConf.create({
    'algo': algo_cfg,
    'model': {
      'hidden_size': 16,
      'cond_dim': 8,
      'length': 16,
      'n_blocks': 1,
      'n_heads': 2,
      'scale_by_sigma': True,
      'dropout': 0.0,
    },
    'sampling': {
      'predictor': 'ancestral',
      'p_nucleus': 1.0,
      'use_float64': False,
      'noise_removal': 'none',
      'steps': 1,
      'guid_weight': None,
    },
    'training': {
      'antithetic_sampling': False,
      'sampling_eps': 1e-3,
      'class_dropout_p': 0.0,
      'ema': 0,
    },
    'noise': {'type': 'log-linear', 'eps': 1e-3},
    'prior': {'type': 'none'},
    'eval': {
      'gen_ppl_eval_model_name_or_path': None,
      'perplexity_batch_size': 1,
    },
    'data': {'tokenizer_name_or_path': 'safe-gpt',
             'modality': 'molecule'},
    'optim': {
      'lr': 1e-4,
      'beta1': 0.9,
      'beta2': 0.999,
      'eps': 1e-8,
      'weight_decay': 0,
    },
  })


def test_safe_tokenizer_special_ids_and_roundtrip():
  pytest.importorskip('safe')
  import dataloader
  import molecule_utils

  tokenizer = dataloader.get_tokenizer(_tokenizer_config())
  assert tokenizer.vocab_size == 1880
  assert tokenizer.bos_token_id == 1
  assert tokenizer.eos_token_id == 2
  assert tokenizer.pad_token_id == 3
  assert tokenizer.mask_token_id == 4

  encoded = tokenizer(
    'CCO',
    max_length=16,
    padding='max_length',
    truncation=True,
    return_token_type_ids=False)
  decoded = tokenizer.decode(
    encoded['input_ids'],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False)
  assert decoded == 'CCO'

  eos_position = encoded['input_ids'].index(tokenizer.eos_token_id)
  generated_ids = (
    encoded['input_ids'][:eos_position + 1]
    + [27, 27, 27]
    + encoded['input_ids'][eos_position + 4:])
  assert molecule_utils.decode_token_ids_to_safe(
    generated_ids, tokenizer) == 'CCO'
  assert molecule_utils.token_decode_metadata(
    generated_ids, tokenizer)['decoded_length'] == eos_position


def test_safe_gpt_v1_streaming_batch_shape():
  if os.environ.get('RUN_SAFE_GPT_STREAMING_TEST') != '1':
    pytest.skip('Set RUN_SAFE_GPT_STREAMING_TEST=1 for HF streaming smoke.')
  pytest.importorskip('safe')
  import dataloader

  tokenizer = dataloader.get_tokenizer(_tokenizer_config())
  try:
    dataset = dataloader.get_dataset(
      'safe-gpt-v1',
      tokenizer,
      wrap=False,
      mode='validation',
      cache_dir='/tmp/data',
      insert_eos=False,
      block_size=256,
      num_proc=1,
      streaming=True)
    batch = next(iter(torch.utils.data.DataLoader(
      dataset,
      batch_size=2,
      num_workers=0)))
  except Exception as exc:
    pytest.skip(f'SAFE-GPT V1 streaming dataset unavailable: {exc}')

  assert batch['input_ids'].shape == (2, 256)
  assert batch['attention_mask'].shape == (2, 256)


def test_generation_metrics_diversity_keeps_valid_duplicates():
  pytest.importorskip('rdkit')
  import molecule_utils

  records = molecule_utils.evaluate_molecules(
    smiles=['CCO', 'CCO', 'c1ccccc1'])
  metrics = molecule_utils.generation_metrics(records)

  diversity_with_duplicates = molecule_utils.internal_diversity(
    ['CCO', 'CCO', 'c1ccccc1'])
  diversity_unique_only = molecule_utils.internal_diversity(
    ['CCO', 'c1ccccc1'])
  assert metrics['num_unique'] == 2
  assert metrics['uniqueness'] == pytest.approx(2 / 3)
  assert metrics['diversity'] == pytest.approx(diversity_with_duplicates)
  assert metrics['diversity'] < diversity_unique_only


def test_duo_loss_runs_on_safe_batch(tmp_path):
  pytest.importorskip('safe')
  import algo
  import dataloader

  tokenizer = dataloader.get_tokenizer(_tokenizer_config())
  model = algo.DUO(_tiny_config(tmp_path, algo_name='duo'), tokenizer)
  batch = tokenizer(
    ['CCO', 'c1ccccc1'],
    max_length=16,
    padding='max_length',
    truncation=True,
    return_tensors='pt',
    return_token_type_ids=False)
  loss = model._loss(
    batch['input_ids'],
    None,
    batch['attention_mask'],
    train_mode=True)
  assert torch.isfinite(loss.loss)
  assert loss.num_tokens.item() == batch['input_ids'].numel()


def test_conditional_sampler_preserves_known_tokens(tmp_path):
  pytest.importorskip('safe')
  import algo
  import dataloader

  tokenizer = dataloader.get_tokenizer(_tokenizer_config())
  model = algo.DUO_BASE(
    _tiny_config(tmp_path, algo_name='duo_base'),
    tokenizer)
  template = torch.full((16,), tokenizer.pad_token_id)
  template[:4] = torch.tensor([
    tokenizer.bos_token_id,
    27,
    27,
    tokenizer.eos_token_id])
  known_mask = torch.zeros(16, dtype=torch.bool)
  known_mask[:4] = True

  samples = model.generate_samples(
    2,
    num_steps=1,
    eps=1e-3,
    token_template=template,
    known_token_mask=known_mask)
  assert torch.equal(samples[:, :4], template[:4].expand(2, -1))
  assert torch.equal(
    samples[:, 4:],
    torch.full_like(samples[:, 4:], tokenizer.pad_token_id))


def test_molecule_sampler_forces_boundaries_and_forbidden_tokens(tmp_path):
  pytest.importorskip('safe')
  import algo
  import dataloader
  import molecule_utils

  tokenizer = dataloader.get_tokenizer(_tokenizer_config())
  model = algo.DUO_BASE(
    _tiny_config(tmp_path, algo_name='duo_base'),
    tokenizer)

  samples = model.generate_samples(8, num_steps=1, eps=1e-3)
  forbidden = {
    tokenizer.bos_token_id,
    tokenizer.unk_token_id,
    tokenizer.mask_token_id,
  }
  assert torch.equal(
    samples[:, 0],
    torch.full_like(samples[:, 0], tokenizer.bos_token_id))
  for token_id in forbidden:
    assert not torch.any(samples[:, 1:] == token_id)

  diagnostics = molecule_utils.token_decode_summary(
    samples, tokenizer)
  assert diagnostics['first_token_is_bos_rate'] == 1.0
  assert 'decoded_length_median' in diagnostics
