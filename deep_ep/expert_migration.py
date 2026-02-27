# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

ExpertParams = Dict[str, Any]
CompressionMetadata = Dict[str, Any]
CompressionFn = Callable[[ExpertParams], Tuple[ExpertParams, CompressionMetadata]]
DecompressionFn = Callable[[ExpertParams, CompressionMetadata, Any], ExpertParams]
MigrationPolicyFn = Callable[[Iterable[int], Any], Mapping[int, int]]


@dataclass
class ExpertMigrationPayload:
    expert_idx: int
    destination_rank: int
    params: ExpertParams
    metadata: CompressionMetadata


def default_compress_expert_params(params: ExpertParams) -> Tuple[ExpertParams, CompressionMetadata]:
    torch = __import__("torch")
    cpu_params = {name: tensor.detach().cpu() for name, tensor in params.items()}
    return cpu_params, {"compression": "identity"}


def default_decompress_expert_params(
    params: ExpertParams, metadata: CompressionMetadata, device: Any
) -> ExpertParams:
    _ = metadata
    torch = __import__("torch")
    return {name: tensor.to(device=device) for name, tensor in params.items()}


def default_expert_migration_policy(
    local_expert_indices: Iterable[int], group: Any
) -> Mapping[int, int]:
    rank = group.rank()
    return {expert_idx: rank for expert_idx in local_expert_indices}


def gather_expert_migration_payloads(
    group: Any,
    outgoing_payloads: List[ExpertMigrationPayload],
) -> List[List[ExpertMigrationPayload]]:
    dist = __import__("torch").distributed

    gathered: List[List[ExpertMigrationPayload]] = [
        [] for _ in range(group.size())
    ]
    dist.all_gather_object(gathered, outgoing_payloads, group=group)
    return gathered
