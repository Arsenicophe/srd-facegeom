"""Téléchargement de CelebA avec vérification d'intégrité.

Les fichiers sont téléchargés dans un dossier temporaire local, vérifiés par
empreinte MD5, puis seulement copiés vers la destination (Google Drive sur
Colab). Un fichier déjà présent avec la bonne empreinte n'est pas retéléchargé.
Un manifeste JSON consigne la source, la révision et les empreintes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class ChecksumError(RuntimeError):
    """Empreinte MD5 différente de l'empreinte attendue."""


def md5sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Calcule l'empreinte MD5 d'un fichier, par morceaux."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_verified_file(
    repo_id: str,
    filename: str,
    expected_md5: str,
    dest_dir: str | Path,
    revision: str,
    repo_type: str = "dataset",
) -> Path:
    """Télécharge un fichier depuis le Hub Hugging Face s'il manque, et le vérifie.

    Renvoie le chemin du fichier vérifié dans ``dest_dir``. Lève
    ``ChecksumError`` si l'empreinte ne correspond pas : rien n'est alors écrit
    dans la destination.
    """
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    if dest.exists():
        if md5sum(dest) == expected_md5:
            print(f"[ok]     {filename} déjà présent et vérifié")
            return dest
        print(f"[!]      {filename} présent mais empreinte fausse : retéléchargement")

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[...]    téléchargement de {filename}")
        local = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                revision=revision,
                local_dir=tmp,
            )
        )
        got = md5sum(local)
        if got != expected_md5:
            raise ChecksumError(
                f"{filename} : MD5 {got}, attendu {expected_md5}. Fichier refusé."
            )
        shutil.copyfile(local, dest)

    if md5sum(dest) != expected_md5:
        raise ChecksumError(f"{filename} : copie vers {dest} corrompue.")
    print(f"[ok]     {filename} téléchargé et vérifié")
    return dest


def download_celeba(celeba_cfg: dict, drive_root: str | Path) -> Path:
    """Télécharge et vérifie tous les fichiers CelebA décrits dans la configuration.

    ``celeba_cfg`` est la section ``celeba`` de ``configs/data.yaml``. Écrit un
    manifeste ``manifest.json`` à côté des fichiers et renvoie le dossier.
    """
    src = celeba_cfg["source"]
    dest_dir = Path(drive_root) / celeba_cfg["dest_dir"]

    for filename, md5 in celeba_cfg["files"].items():
        fetch_verified_file(
            repo_id=src["repo_id"],
            filename=filename,
            expected_md5=md5,
            dest_dir=dest_dir,
            revision=src["revision"],
            repo_type=src.get("repo_type", "dataset"),
        )

    manifest = {
        "source": src,
        "files_md5": celeba_cfg["files"],
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dest_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dest_dir
