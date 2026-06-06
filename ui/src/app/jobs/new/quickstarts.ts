'use client';

import { JobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';

export interface QuickstartTemplate {
  id: string;
  label: string;
  description: string;
  apply: (current: JobConfig) => JobConfig;
}

type Process = JobConfig['config']['process'][number];

const baseSettingsOnlyConfig = (current: JobConfig): JobConfig => {
  const next = objectCopy(current);
  const process = next.config.process[0] as Process & { network: NonNullable<Process['network']> };

  process.type = 'diffusion_trainer';
  process.device = process.device || 'cuda';
  process.performance_log_every = process.performance_log_every ?? 10;

  process.network = {
    type: 'lora',
    linear: 16,
    linear_alpha: 16,
    conv: 0,
    conv_alpha: 0,
    lokr_full_rank: true,
    lokr_factor: -1,
    network_kwargs: {
      ignore_if_contains: [],
    },
  };

  process.save = {
    ...process.save,
    dtype: 'bf16',
    save_every: 250,
    max_step_saves_to_keep: 4,
    save_format: 'diffusers',
    push_to_hub: false,
  };

  process.train = {
    ...process.train,
    batch_size: 1,
    bypass_guidance_embedding: true,
    steps: 3000,
    gradient_accumulation: 1,
    train_unet: true,
    train_text_encoder: false,
    gradient_checkpointing: true,
    noise_scheduler: 'flowmatch',
    timestep_type: 'weighted',
    content_or_style: 'balanced',
    optimizer: 'adamw8bit',
    lr: 0.0001,
    dtype: 'bf16',
    unload_text_encoder: false,
    cache_text_embeddings: true,
    skip_first_sample: false,
    force_first_sample: false,
    disable_sampling: false,
    diff_output_preservation: false,
    diff_output_preservation_multiplier: 1.0,
    diff_output_preservation_class: 'person',
    switch_boundary_every: 1,
    loss_type: 'mse',
    do_differential_guidance: undefined,
    differential_guidance_scale: undefined,
    ema_config: {
      use_ema: false,
      ema_decay: 0.99,
    },
    optimizer_params: {
      ...(process.train.optimizer_params || {}),
      weight_decay: 0.0001,
    },
  };

  process.model = {
    ...process.model,
    name_or_path: 'Tongyi-MAI/Z-Image-Turbo',
    arch: 'zimage:turbo',
    quantize: true,
    quantize_te: true,
    qtype: 'qfloat8',
    qtype_te: 'qfloat8',
    low_vram: false,
    layer_offloading: false,
    layer_offloading_transformer_percent: 0,
    layer_offloading_text_encoder_percent: 0,
    model_kwargs: {},
    assistant_lora_path: 'ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors',
  };
  delete (process.model as any).extras_name_or_path;

  process.sample = {
    ...process.sample,
    sampler: 'flowmatch',
    sample_every: 250,
    width: process.sample.width || 1024,
    height: process.sample.height || 1024,
    guidance_scale: 1,
    sample_steps: 8,
  };

  return next;
};

export const QUICKSTARTS: QuickstartTemplate[] = [
  {
    id: 'passi-zimage-turbo-lora16-settings',
    label: 'Passi - Z-Image Turbo LoRA16 Character (settings only)',
    description:
      'Modern Z-Image Turbo adapter baseline. Preserves training name, trigger, all dataset config, and sample prompts.',
    apply: baseSettingsOnlyConfig,
  },
  {
    id: 'passi-zimage-deturb-lokr128-settings',
    label: 'Passi - Z-Image De-Turbo LoKr128 Character (settings only)',
    description:
      'Heavier De-Turbo likeness route. Preserves training name, trigger, all dataset config, and sample prompts.',
    apply: (current: JobConfig): JobConfig => {
      const next = baseSettingsOnlyConfig(current);
      const process = next.config.process[0] as Process & { network: NonNullable<Process['network']> };

      process.network = {
        ...process.network,
        type: 'lokr',
        linear: 128,
        linear_alpha: 64,
        conv: 32,
        conv_alpha: 16,
        lokr_full_rank: true,
        lokr_factor: 4,
      };

      process.model = {
        ...process.model,
        name_or_path: 'ostris/Z-Image-De-Turbo',
        arch: 'zimage:deturbo',
        assistant_lora_path: undefined,
      };
      (process.model as any).extras_name_or_path = 'Tongyi-MAI/Z-Image-Turbo';

      process.train = {
        ...process.train,
        cache_text_embeddings: false,
        steps: 3000,
        gradient_checkpointing: true,
      };

      process.sample = {
        ...process.sample,
        guidance_scale: 3,
        sample_steps: 25,
      };

      return next;
    },
  },
];
