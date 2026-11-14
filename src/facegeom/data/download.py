"""Téléchargement de CelebA depuis Hugging Face, avec vérification MD5.

Chaque fichier est téléchargé en local, vérifié, puis copié vers la destination
(Drive sur Colab). Un fichier déjà présent et correct n'est pas retéléchargé.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from facegeom.utils import file_hash


class ChecksumError(RuntimeError):
    """Empreinte différente de celle attendue."""


def fetch_verified_file(
    repo_id: str, filename: str, expected_md5: str, dest_dir: Path, revision: str, repo_type: str
) -> Path:
    """Télécharge ``filename`` s'il manque ou est faux ; lève ChecksumError sinon."""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists():
        if file_hash(dest) == expected_md5:
            print(f"[ok]     {filename} déjà présent et vérifié")
            return dest
        print(f"[!]      {filename} présent mais empreinte fausse : retéléchargement")

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[...]    téléchargement de {filename}")
        local = hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision, local_dir=tmp)
        got = file_hash(local)
        if got != expected_md5:
            raise ChecksumError(f"{filename} : MD5 {got}, attendu {expected_md5}. Fichier refusé.")
        shutil.copyfile(local, dest)

    if file_hash(dest) != expected_md5:
        raise ChecksumError(f"{filename} : copie vers {dest} corrompue.")
    print(f"[ok]     {filename} téléchargé et vérifié")
    return dest


def download_celeba(celeba_cfg: dict, drive_root: str | Path) -> Path:
    """Télécharge tous les fichiers de ``configs/data.yaml`` et écrit un manifeste."""
    src = celeba_cfg["source"]
    dest_dir = Path(drive_root) / celeba_cfg["dest_dir"]
    for filename, md5 in celeba_cfg["files"].items():
        fetch_verified_file(src["repo_id"], filename, md5, dest_dir, src["revision"], src["repo_type"])
    manifest = {
        "source": src,
        "files_md5": celeba_cfg["files"],
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return dest_dir
