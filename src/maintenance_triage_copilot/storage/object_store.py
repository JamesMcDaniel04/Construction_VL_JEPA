"""Object storage abstractions for persisted media and audit evidence."""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from maintenance_triage_copilot.config import ObjectStoreConfig


@dataclass
class StoredObject:
    object_uri: str
    storage_key: str
    byte_size: int
    sha256: str
    content_type: str
    created_at: datetime


class ObjectStore(Protocol):
    def put_bytes(
        self,
        *,
        asset_type: str,
        filename: str,
        data: bytes,
        content_type: str,
    ) -> StoredObject: ...

    def presigned_url(self, object_uri: str, expires_seconds: int = 3600) -> str: ...

    def status(self) -> dict[str, str]: ...


class MemoryObjectStore:
    def __init__(self) -> None:
        self._items: dict[str, bytes] = {}

    def put_bytes(
        self,
        *,
        asset_type: str,
        filename: str,
        data: bytes,
        content_type: str,
    ) -> StoredObject:
        sha256 = hashlib.sha256(data).hexdigest()
        suffix = Path(filename).suffix.lower()
        storage_key = f"{asset_type}/{sha256[:2]}/{sha256}{suffix}"
        uri = f"memory://{storage_key}"
        self._items[storage_key] = data
        return StoredObject(
            object_uri=uri,
            storage_key=storage_key,
            byte_size=len(data),
            sha256=sha256,
            content_type=content_type,
            created_at=datetime.now(tz=UTC),
        )

    def presigned_url(self, object_uri: str, expires_seconds: int = 3600) -> str:
        return object_uri

    def status(self) -> dict[str, str]:
        return {"object_store": "memory", "objects": str(len(self._items))}


class S3ObjectStore:
    def __init__(self, cfg: ObjectStoreConfig, *, required: bool = False):
        if not cfg.endpoint_url or not cfg.bucket:
            if required:
                raise RuntimeError("Production mode requires object store endpoint_url and bucket")
            raise ValueError("Object store endpoint_url and bucket must be configured")

        import boto3
        from botocore.client import Config as BotoConfig
        from botocore.exceptions import ClientError

        self.bucket = cfg.bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            region_name=cfg.region,
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            config=BotoConfig(s3={"addressing_style": "path" if cfg.force_path_style else "auto"}),
        )
        self._endpoint_url = cfg.endpoint_url.rstrip("/")
        self._client_error = ClientError
        self._ensure_bucket(create_if_missing=cfg.create_bucket_if_missing)

    def put_bytes(
        self,
        *,
        asset_type: str,
        filename: str,
        data: bytes,
        content_type: str,
    ) -> StoredObject:
        sha256 = hashlib.sha256(data).hexdigest()
        suffix = Path(filename).suffix.lower()
        storage_key = f"{asset_type}/{sha256[:2]}/{sha256}{suffix}"
        self._client.upload_fileobj(
            Fileobj=io.BytesIO(data),
            Bucket=self.bucket,
            Key=storage_key,
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": {"sha256": sha256, "asset-type": asset_type},
            },
        )
        return StoredObject(
            object_uri=f"s3://{self.bucket}/{storage_key}",
            storage_key=storage_key,
            byte_size=len(data),
            sha256=sha256,
            content_type=content_type,
            created_at=datetime.now(tz=UTC),
        )

    def presigned_url(self, object_uri: str, expires_seconds: int = 3600) -> str:
        prefix = f"s3://{self.bucket}/"
        if not object_uri.startswith(prefix):
            raise ValueError(f"Unexpected object URI for bucket {self.bucket}: {object_uri}")
        storage_key = object_uri.removeprefix(prefix)
        return str(
            self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": storage_key},
                ExpiresIn=expires_seconds,
            )
        )

    def status(self) -> dict[str, str]:
        return {
            "object_store": "s3",
            "bucket": self.bucket,
            "endpoint_url": self._endpoint_url,
        }

    def _ensure_bucket(self, *, create_if_missing: bool) -> None:
        deadline = time.time() + 20
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                self._client.head_bucket(Bucket=self.bucket)
                return
            except Exception as exc:
                last_error = exc
                if not create_if_missing:
                    time.sleep(1)
                    continue
                try:
                    self._client.create_bucket(Bucket=self.bucket)
                    return
                except Exception as create_exc:
                    last_error = create_exc
                    time.sleep(1)
        if last_error is not None:
            raise last_error
