"""Adapters and portable fixtures for the input attribution package."""

from .fake import FakeAdapter, FakePolicyAdapter, FakeVectorEnv, create_adapter
from .fixtures import (
    assert_schema_resolved,
    custom_262_schema_template,
    custom_schema_template,
    fake_262_resolved_schema,
    official_schema,
    official_schema_259,
    unresolved_indices,
)
from .metadrive import (
    AdapterError,
    MetaDriveAdapter,
    PolicyAdapter,
    PolicyOutput,
    TensorPolicyOutput,
    create_adapter as create_metadrive_adapter,
)
from .port_template import PortAdapterTemplate, create_adapter as create_port_adapter

__all__ = [
    "AdapterError",
    "FakeAdapter",
    "FakePolicyAdapter",
    "FakeVectorEnv",
    "MetaDriveAdapter",
    "PolicyAdapter",
    "PolicyOutput",
    "PortAdapterTemplate",
    "TensorPolicyOutput",
    "assert_schema_resolved",
    "custom_262_schema_template",
    "custom_schema_template",
    "create_adapter",
    "fake_262_resolved_schema",
    "create_metadrive_adapter",
    "create_port_adapter",
    "official_schema",
    "official_schema_259",
    "unresolved_indices",
]
