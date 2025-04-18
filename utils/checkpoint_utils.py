import re
import torch
import torch.nn.functional as F


def remap_dino_checkpoint(state_dict):
    keys_to_remove = [k for k in state_dict if k.startswith("dino_head.")]
    for k in keys_to_remove:
        del state_dict[k]

    for key in list(state_dict.keys()):
        m = re.match(r"backbone\.blocks\.(\d+)\.attn\.qkv\.weight", key)
        if m:
            block_idx = int(m.group(1))
            qkv_weight = state_dict.pop(f"backbone.blocks.{block_idx}.attn.qkv.weight")
            qkv_bias = state_dict.pop(f"backbone.blocks.{block_idx}.attn.qkv.bias")
            hidden_dim = qkv_bias.shape[0] // 3

            state_dict[f"encoder.layer.{block_idx}.attention.attention.query.weight"] = qkv_weight[:hidden_dim, :]
            state_dict[f"encoder.layer.{block_idx}.attention.attention.query.bias"] = qkv_bias[:hidden_dim]
            state_dict[f"encoder.layer.{block_idx}.attention.attention.key.weight"] = qkv_weight[hidden_dim:2*hidden_dim, :]
            state_dict[f"encoder.layer.{block_idx}.attention.attention.key.bias"] = qkv_bias[hidden_dim:2*hidden_dim]
            state_dict[f"encoder.layer.{block_idx}.attention.attention.value.weight"] = qkv_weight[2*hidden_dim:, :]
            state_dict[f"encoder.layer.{block_idx}.attention.attention.value.bias"] = qkv_bias[2*hidden_dim:]

    block_matches = [
        int(m.group(1))
        for m in (re.match(r"backbone\.blocks\.(\d+)\.", k) for k in state_dict.keys())
        if m
    ]

    if not block_matches:
        raise ValueError("No transformer blocks found in checkpoint. Is this a valid DINO checkpoint?")

    num_blocks = max(block_matches) + 1

    rename_map = [
        ("backbone.cls_token", "embeddings.cls_token"),
        ("backbone.mask_token", "embeddings.mask_token"),
        ("backbone.pos_embed", "embeddings.position_embeddings"),
        ("backbone.patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"),
        ("backbone.patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"),
    ]
    for i in range(num_blocks):
        rename_map += [
            (f"backbone.blocks.{i}.norm1.weight", f"encoder.layer.{i}.norm1.weight"),
            (f"backbone.blocks.{i}.norm1.bias", f"encoder.layer.{i}.norm1.bias"),
            (f"backbone.blocks.{i}.norm2.weight", f"encoder.layer.{i}.norm2.weight"),
            (f"backbone.blocks.{i}.norm2.bias", f"encoder.layer.{i}.norm2.bias"),
            (f"backbone.blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.mlp.fc1.weight"),
            (f"backbone.blocks.{i}.mlp.fc1.bias", f"encoder.layer.{i}.mlp.fc1.bias"),
            (f"backbone.blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.mlp.fc2.weight"),
            (f"backbone.blocks.{i}.mlp.fc2.bias", f"encoder.layer.{i}.mlp.fc2.bias"),
            (f"backbone.blocks.{i}.ls1.gamma", f"encoder.layer.{i}.layer_scale1.lambda1"),
            (f"backbone.blocks.{i}.ls2.gamma", f"encoder.layer.{i}.layer_scale2.lambda1"),
            (f"backbone.blocks.{i}.attn.proj.weight", f"encoder.layer.{i}.attention.output.dense.weight"),
            (f"backbone.blocks.{i}.attn.proj.bias", f"encoder.layer.{i}.attention.output.dense.bias"),
        ]

    for old_key, new_key in rename_map:
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)

    if "backbone.norm.weight" in state_dict:
        state_dict["layernorm.weight"] = state_dict.pop("backbone.norm.weight")
    if "backbone.norm.bias" in state_dict:
        state_dict["layernorm.bias"] = state_dict.pop("backbone.norm.bias")

    return state_dict


def interpolate_pos_embed(pos_embed_checkpoint, pos_embed_model, num_extra_tokens=1):
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    old_grid_size = int(pos_tokens.shape[1] ** 0.5)
    new_grid_size = int((pos_embed_model.shape[1] - num_extra_tokens) ** 0.5)

    pos_tokens = pos_tokens.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, -1)

    return torch.cat((extra_tokens, pos_tokens), dim=1)
