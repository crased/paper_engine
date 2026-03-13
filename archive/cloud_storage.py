"""
Cloud Storage providers for Paper Engine.

Supports: Google Drive, AWS S3, Dropbox, Azure Blob Storage, Hugging Face Hub.

Each provider implements upload/download/list_remote for dataset and model syncing.
Non-sensitive config (bucket names, regions, paths) comes from conf/main_conf.ini.
Sensitive credentials (tokens, keys, connection strings) come from the OS keyring
via core.functions.store_secret / get_secret.
"""

import configparser
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fields that hold secrets — must be read from keyring, not INI.
from tools.functions import get_secret, is_sensitive_field


class CloudProvider:
    """Abstract base for cloud storage providers."""

    name = "base"

    def upload(self, local_path, remote_path=None, progress_cb=None):
        """Upload a file or directory to cloud storage.

        Args:
            local_path: Path to local file or directory.
            remote_path: Destination path on remote (provider-specific).
            progress_cb: Optional callback(current, total, filename).
        """
        raise NotImplementedError

    def download(self, remote_path, local_path, progress_cb=None):
        """Download a file or directory from cloud storage.

        Args:
            remote_path: Source path on remote.
            local_path: Local destination path.
            progress_cb: Optional callback(current, total, filename).
        """
        raise NotImplementedError

    def list_remote(self, remote_path="/"):
        """List files at the given remote path.

        Returns:
            list[dict]: Each dict has 'name', 'size' (bytes), 'type' ('file'|'dir').
        """
        raise NotImplementedError

    def test_connection(self):
        """Test that credentials are valid. Returns True or raises."""
        raise NotImplementedError


# ======================================================================
# Google Drive
# ======================================================================


class GoogleDriveProvider(CloudProvider):
    name = "gdrive"

    def __init__(self, folder_id, credentials_path):
        self.folder_id = folder_id
        self.credentials_path = credentials_path

    def _build_service(self):
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path
        )
        return build("drive", "v3", credentials=creds)

    def test_connection(self):
        service = self._build_service()
        service.files().list(q=f"'{self.folder_id}' in parents", pageSize=1).execute()
        return True

    def upload(self, local_path, remote_path=None, progress_cb=None):
        from googleapiclient.http import MediaFileUpload

        service = self._build_service()
        local_path = Path(local_path)

        files = [local_path] if local_path.is_file() else sorted(local_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        total = len(files)

        for i, fpath in enumerate(files):
            rel = fpath.relative_to(local_path) if local_path.is_dir() else fpath.name
            media = MediaFileUpload(str(fpath), resumable=True)
            meta = {
                "name": str(rel),
                "parents": [self.folder_id],
            }
            service.files().create(body=meta, media_body=media).execute()
            if progress_cb:
                progress_cb(i + 1, total, str(rel))

    def download(self, remote_path, local_path, progress_cb=None):
        import io
        from googleapiclient.http import MediaIoBaseDownload

        service = self._build_service()
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        results = (
            service.files()
            .list(q=f"'{self.folder_id}' in parents", pageSize=1000)
            .execute()
        )
        items = results.get("files", [])
        total = len(items)

        for i, item in enumerate(items):
            dest = local_path / item["name"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            request = service.files().get_media(fileId=item["id"])
            with open(dest, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            if progress_cb:
                progress_cb(i + 1, total, item["name"])

    def list_remote(self, remote_path="/"):
        service = self._build_service()
        results = (
            service.files()
            .list(
                q=f"'{self.folder_id}' in parents",
                pageSize=1000,
                fields="files(id,name,size,mimeType)",
            )
            .execute()
        )
        out = []
        for item in results.get("files", []):
            is_dir = item["mimeType"] == "application/vnd.google-apps.folder"
            out.append(
                {
                    "name": item["name"],
                    "size": int(item.get("size", 0)),
                    "type": "dir" if is_dir else "file",
                }
            )
        return out


# ======================================================================
# AWS S3
# ======================================================================


class S3Provider(CloudProvider):
    name = "s3"

    def __init__(self, bucket, region, access_key, secret_key):
        self.bucket = bucket
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key

    def _client(self):
        import boto3

        return boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    def test_connection(self):
        self._client().head_bucket(Bucket=self.bucket)
        return True

    def upload(self, local_path, remote_path=None, progress_cb=None):
        s3 = self._client()
        local_path = Path(local_path)
        prefix = remote_path or ""

        files = [local_path] if local_path.is_file() else sorted(local_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        total = len(files)

        for i, fpath in enumerate(files):
            if local_path.is_dir():
                key = (
                    f"{prefix}/{fpath.relative_to(local_path)}"
                    if prefix
                    else str(fpath.relative_to(local_path))
                )
            else:
                key = f"{prefix}/{fpath.name}" if prefix else fpath.name
            s3.upload_file(str(fpath), self.bucket, key)
            if progress_cb:
                progress_cb(i + 1, total, key)

    def download(self, remote_path, local_path, progress_cb=None):
        s3 = self._client()
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        paginator = s3.get_paginator("list_objects_v2")
        all_keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=remote_path or ""):
            for obj in page.get("Contents", []):
                all_keys.append(obj["Key"])

        total = len(all_keys)
        for i, key in enumerate(all_keys):
            dest = local_path / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(self.bucket, key, str(dest))
            if progress_cb:
                progress_cb(i + 1, total, key)

    def list_remote(self, remote_path="/"):
        s3 = self._client()
        prefix = "" if remote_path == "/" else remote_path
        resp = s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1000)
        out = []
        for obj in resp.get("Contents", []):
            out.append(
                {
                    "name": obj["Key"],
                    "size": obj["Size"],
                    "type": "file",
                }
            )
        return out


# ======================================================================
# Dropbox
# ======================================================================


class DropboxProvider(CloudProvider):
    name = "dropbox"

    def __init__(self, token, path="/paper_engine"):
        self.token = token
        self.path = path

    def _client(self):
        import dropbox

        return dropbox.Dropbox(self.token)

    def test_connection(self):
        self._client().users_get_current_account()
        return True

    def upload(self, local_path, remote_path=None, progress_cb=None):
        import dropbox as dbx_mod

        dbx = self._client()
        local_path = Path(local_path)
        dest_root = remote_path or self.path

        files = [local_path] if local_path.is_file() else sorted(local_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        total = len(files)

        for i, fpath in enumerate(files):
            if local_path.is_dir():
                rel = str(fpath.relative_to(local_path))
            else:
                rel = fpath.name
            dest = f"{dest_root}/{rel}"
            with open(fpath, "rb") as f:
                dbx.files_upload(
                    f.read(),
                    dest,
                    mode=dbx_mod.files.WriteMode.overwrite,
                )
            if progress_cb:
                progress_cb(i + 1, total, rel)

    def download(self, remote_path, local_path, progress_cb=None):
        dbx = self._client()
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        result = dbx.files_list_folder(remote_path or self.path, recursive=True)
        entries = [e for e in result.entries if hasattr(e, "size")]
        total = len(entries)

        for i, entry in enumerate(entries):
            rel = entry.path_display.replace(remote_path or self.path, "").lstrip("/")
            dest = local_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dbx.files_download_to_file(str(dest), entry.path_display)
            if progress_cb:
                progress_cb(i + 1, total, rel)

    def list_remote(self, remote_path="/"):
        dbx = self._client()
        result = dbx.files_list_folder(remote_path if remote_path != "/" else self.path)
        out = []
        for entry in result.entries:
            is_dir = not hasattr(entry, "size")
            out.append(
                {
                    "name": entry.name,
                    "size": getattr(entry, "size", 0),
                    "type": "dir" if is_dir else "file",
                }
            )
        return out


# ======================================================================
# Azure Blob Storage
# ======================================================================


class AzureBlobProvider(CloudProvider):
    name = "azure"

    def __init__(self, connection_string, container):
        self.connection_string = connection_string
        self.container = container

    def _container_client(self):
        from azure.storage.blob import BlobServiceClient

        client = BlobServiceClient.from_connection_string(self.connection_string)
        return client.get_container_client(self.container)

    def test_connection(self):
        self._container_client().get_container_properties()
        return True

    def upload(self, local_path, remote_path=None, progress_cb=None):
        cc = self._container_client()
        local_path = Path(local_path)
        prefix = remote_path or ""

        files = [local_path] if local_path.is_file() else sorted(local_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        total = len(files)

        for i, fpath in enumerate(files):
            if local_path.is_dir():
                blob_name = (
                    f"{prefix}/{fpath.relative_to(local_path)}"
                    if prefix
                    else str(fpath.relative_to(local_path))
                )
            else:
                blob_name = f"{prefix}/{fpath.name}" if prefix else fpath.name
            with open(fpath, "rb") as data:
                cc.upload_blob(name=blob_name, data=data, overwrite=True)
            if progress_cb:
                progress_cb(i + 1, total, blob_name)

    def download(self, remote_path, local_path, progress_cb=None):
        cc = self._container_client()
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        blobs = list(cc.list_blobs(name_starts_with=remote_path or ""))
        total = len(blobs)

        for i, blob in enumerate(blobs):
            dest = local_path / blob.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                data = cc.download_blob(blob.name)
                f.write(data.readall())
            if progress_cb:
                progress_cb(i + 1, total, blob.name)

    def list_remote(self, remote_path="/"):
        cc = self._container_client()
        prefix = "" if remote_path == "/" else remote_path
        out = []
        for blob in cc.list_blobs(name_starts_with=prefix):
            out.append(
                {
                    "name": blob.name,
                    "size": blob.size,
                    "type": "file",
                }
            )
        return out


# ======================================================================
# Hugging Face Hub
# ======================================================================


class HuggingFaceProvider(CloudProvider):
    name = "hf"

    def __init__(self, repo_id, token):
        self.repo_id = repo_id
        self.token = token

    def _api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def test_connection(self):
        self._api().repo_info(self.repo_id)
        return True

    def upload(self, local_path, remote_path=None, progress_cb=None):
        from huggingface_hub import HfApi

        api = self._api()
        local_path = Path(local_path)

        if local_path.is_dir():
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=self.repo_id,
                path_in_repo=remote_path or "",
                repo_type="dataset",
            )
            if progress_cb:
                progress_cb(1, 1, str(local_path.name))
        else:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path or local_path.name,
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            if progress_cb:
                progress_cb(1, 1, local_path.name)

    def download(self, remote_path, local_path, progress_cb=None):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=str(local_path),
            token=self.token,
        )
        if progress_cb:
            progress_cb(1, 1, self.repo_id)

    def list_remote(self, remote_path="/"):
        api = self._api()
        info = api.repo_info(self.repo_id, repo_type="dataset")
        out = []
        for sibling in info.siblings or []:
            out.append(
                {
                    "name": sibling.rfilename,
                    "size": getattr(sibling, "size", 0) or 0,
                    "type": "file",
                }
            )
        return out


# ======================================================================
# Config reader + factory
# ======================================================================


def _read_cloud_config():
    """Read [CloudStorage] from INI + merge sensitive fields from keyring."""
    ini = PROJECT_ROOT / "conf" / "main_conf.ini"
    p = configparser.ConfigParser()
    p.read(ini)
    if not p.has_section("CloudStorage"):
        return {"provider": "none"}

    cfg = dict(p.items("CloudStorage"))

    # Override sensitive fields with keyring values
    for key in list(cfg.keys()):
        if is_sensitive_field(key):
            secret = get_secret(key)
            if secret:
                cfg[key] = secret
            else:
                # INI might still hold a leftover value from before keyring migration.
                # Keep it so existing setups don't break immediately.
                pass
    return cfg


def get_provider(cfg=None):
    """Factory: return a CloudProvider based on config, or None if disabled."""
    if cfg is None:
        cfg = _read_cloud_config()

    pid = cfg.get("provider", "none")
    if pid == "none":
        return None

    if pid == "gdrive":
        return GoogleDriveProvider(
            cfg.get("gdrive_folder_id", ""),
            cfg.get("gdrive_credentials", ""),
        )
    if pid == "s3":
        return S3Provider(
            cfg.get("s3_bucket", ""),
            cfg.get("s3_region", "us-east-1"),
            cfg.get("s3_access_key", ""),
            cfg.get("s3_secret_key", ""),
        )
    if pid == "dropbox":
        return DropboxProvider(
            cfg.get("dropbox_token", ""),
            cfg.get("dropbox_path", "/paper_engine"),
        )
    if pid == "azure":
        return AzureBlobProvider(
            cfg.get("azure_connection_string", ""),
            cfg.get("azure_container", "paper-engine"),
        )
    if pid == "hf":
        return HuggingFaceProvider(
            cfg.get("hf_repo_id", ""),
            cfg.get("hf_token", ""),
        )
    return None
