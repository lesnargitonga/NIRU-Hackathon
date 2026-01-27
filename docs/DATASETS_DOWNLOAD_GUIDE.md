# Downloading Remaining Datasets (Beginner-Friendly)

This guide shows step-by-step how to finish downloading everything that’s still missing without touching what you’ve already downloaded.

- Destination folder: `D:\datasets`
- Downloader script: `datasets\download_datasets.py`
- Python env: `d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe`

> Safety: The downloader is non-destructive. It never deletes extracted folders. It uses resume for partial downloads and writes a marker after successful extraction to avoid re-extracting files.

---

## 1) Audit what’s present vs missing

This prints a list of datasets and their status.

```powershell
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" datasets\download_datasets.py --profile mega --dest D:\datasets --audit
```

### Quick plan (fast)

If the audit is slow on a huge drive, run the fast planner to print exact commands based on a quick peek:

```powershell
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" datasets\plan_datasets.py --root D:\datasets --dest D:\datasets --skip-existing
```

This avoids heavy scans and suggests two commands: one for scrape-able datasets and one for protected ones.

---

## 2) Generate AirSim synthetic data (no website needed)

1) Launch Unreal with the AirSim plugin and start the simulation.
2) Run the capture script to save images/masks to `D:\datasets\airsim_synth`:

```powershell
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" training\collect_airsim_dataset.py --out D:\datasets\airsim_synth
```

You can run this any time to add more data. The downloader won’t attempt to fetch this one over the network.

---

## 3) Auto-download what can be scraped (safe defaults)

This attempts to fetch datasets marked for scraping (e.g., SDD, DOTA, iSAID, TartanAir, VisDrone, UAVDT). It won’t re-extract what’s already unpacked and will skip very large files by default.

```powershell
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" datasets\download_datasets.py \
  --keys stanford_drone visdrone uavdt dota isaid tartanair \
  --dest D:\datasets \
  --skip-errors \
  --allow-scrape \
  --scrape-depth 1 \
  --scrape-max-mb 200 \
  --no-reextract \
  --retries 3
```

Tips:
- If a site is slow/flaky, just re-run. Resume is supported for partial `.part` files.
- To allow larger files, increase `--scrape-max-mb` (e.g., `500`).
- Avoid `--insecure` unless you trust the host and need to bypass a TLS issue.

---

## 4) Download protected datasets using your account (cookies/headers)

Some datasets require logins and agreements: Cityscapes, Mapillary Vistas, BDD100K, KITTI (Raw/360/Odometry), nuScenes, Waymo, Argoverse, Habitat.

- Export cookies from your browser in Netscape format (e.g., using the “Get cookies.txt” extension) after logging in.
- Save to `C:\path\cookies.txt` and run:

```powershell
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" datasets\download_datasets.py \
  --profile mega \
  --dest D:\datasets \
  --allow-scrape \
  --cookie-file C:\path\cookies.txt \
  --skip-errors \
  --no-reextract \
  --retries 3
```

- If a site provides an API token, pass it as a header:

```powershell
--header "Authorization: Bearer YOUR_TOKEN"
```

---

## 5) Manual download path (fastest for stubborn sites)

You can download archives yourself from the official sites and let the tool extract them locally.

- Put downloaded archives into either (preferred first):
  - `D:\datasets\_archives\<dataset_key>\`
  - Or directly into the dataset’s target folder shown below
- Supported for auto-extract: `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.bag`
- For `.7z`/`.rar`, extract them manually, then place the extracted folders into the target directory.

Then run extract-only (no network):

```powershell
# Example: Manual extraction for SDD and DOTA
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" datasets\download_datasets.py \
  --keys stanford_drone dota \
  --dest D:\datasets \
  --extract-only \
  --no-reextract
```

---

## 6) Full pass to pick up anything still missing

You can run this at any time; it will skip already downloaded/extracted content.

```powershell
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" datasets\download_datasets.py \
  --profile mega \
  --dest D:\datasets \
  --skip-errors \
  --allow-scrape \
  --scrape-depth 1 \
  --scrape-max-mb 200 \
  --no-reextract \
  --retries 3
```

---

## Dataset keys → target folders

Place archives under `D:\datasets\_archives\<key>\` (preferred), or use the target folders below.

- `stanford_drone` → `D:\datasets\sdd`
- `visdrone` → `D:\datasets\visdrone`
- `uavdt` → `D:\datasets\uavdt`
- `dota` → `D:\datasets\dota`
- `isaid` → `D:\datasets\isaid`
- `tartanair` → `D:\datasets\tartanair`
- `cityscapes` → `D:\datasets\cityscapes`
- `mapillary_vistas` → `D:\datasets\mapillary_vistas`
- `bdd100k` → `D:\datasets\bdd100k`
- `semantic_kitti` → `D:\datasets\semantic_kitti`
- `nuscenes` → `D:\datasets\nuscenes`
- `kitti_raw` → `D:\datasets\kitti\raw`
- `kitti_360` → `D:\datasets\kitti\360`
- `waymo_open` → `D:\datasets\waymo_open`
- `argoverse` → `D:\datasets\argoverse`
- `habitat_datasets` → `D:\datasets\habitat`
- `airsim_synthetic` (generated) → `D:\datasets\airsim_synth`

---

## Troubleshooting

- “getaddrinfo failed” or repeated timeouts: the datasets’ servers can be slow. Re-run the same command; resume will continue where it left off.
- TLS errors: some older hosts have misconfigured certificates. Only if you trust the site, add `--insecure` for that run.
- FTP links: blocked by default. Enable with `--scrape-allow-ftp` if needed.
- Already extracted: with `--no-reextract`, the tool skips re-extraction if the folder has content or a `.extract_complete` marker.
- Manual archives weren’t found: make sure they’re under either `D:\datasets\_archives\<key>\` or the exact target folder, with supported extensions.

---

## Need help?

If you want, share a `cookies.txt` file and which datasets to prioritize, and we can run a credentials-enabled pass for you. Or we can start the AirSim capture now to fill `airsim_synthetic` without any website.

---

## Manual downloads reference (official links and placement)

If scraping is blocked or slow, download archives in your browser from the official pages and place them under `D:\datasets\_archives\<dataset_key>\`. Then run the extract-only commands below.

Placement rule
- Put archives under: `D:\datasets\_archives\<dataset_key>\`
- Supported for auto-extract: `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.bag`
- For `.7z`/`.rar`: extract manually, then move the extracted folder(s) to the final target (see keys table above).

### Scrape-able datasets (manual path recommended)

- DOTA (key: `dota`)
  - Official: https://captain-whu.github.io/DOTA/
  - Get: DOTA v1.x train/val images and labelTxt (annotations)
  - Drop here: `D:\datasets\_archives\dota\`

- iSAID (key: `isaid`)
  - Official: https://captain-whu.github.io/iSAID/
  - Get: iSAID images + annotations
  - Drop here: `D:\datasets\_archives\isaid\`

- TartanAir (key: `tartanair`)
  - Official: https://theairlab.org/tartanair-dataset/
  - Get: a few sequences (e.g., AbandonedFactory*, Carwelding*, Office*)
  - Drop here: `D:\datasets\_archives\tartanair\`

- VisDrone (key: `visdrone`)
  - Official: http://www.aiskyeye.com/
  - Get first: VisDrone2019-DET-train.zip, VisDrone2019-DET-val.zip
  - Drop here: `D:\datasets\_archives\visdrone\`

- UAVDT (key: `uavdt`)
  - Official: https://sites.google.com/site/daviddo0323/projects/uavdt
  - Get: UAV-benchmark zips (M/S)
  - Drop here: `D:\datasets\_archives\uavdt\`

### Protected/credentialed datasets (manual download needed)

- Cityscapes (key: `cityscapes`) → https://www.cityscapes-dataset.com/downloads/ → `D:\datasets\_archives\cityscapes\`
- Mapillary Vistas (key: `mapillary_vistas`) → https://www.mapillary.com/dataset/vistas → `D:\datasets\_archives\mapillary_vistas\`
- BDD100K (key: `bdd100k`) → https://www.bdd100k.com/download/ → `D:\datasets\_archives\bdd100k\`
- KITTI Raw (key: `kitti_raw`) → http://www.cvlibs.net/datasets/kitti/ → `D:\datasets\_archives\kitti_raw\`
- KITTI-360 (key: `kitti_360`) → http://www.cvlibs.net/datasets/kitti-360/ → `D:\datasets\_archives\kitti_360\`
- SemanticKITTI (key: `semantic_kitti`) → http://www.semantic-kitti.org/dataset.html → `D:\datasets\_archives\semantic_kitti\`
- nuScenes (key: `nuscenes`) → https://www.nuscenes.org/download → `D:\datasets\_archives\nuscenes\`
- Waymo Open (key: `waymo_open`) → https://waymo.com/open/ → `D:\datasets\_archives\waymo_open\`
- Argoverse (key: `argoverse`) → https://www.argoverse.org/ → `D:\datasets\_archives\argoverse\`
- Habitat datasets (key: `habitat_datasets`) → https://aihabitat.org/datasets/ → `D:\datasets\_archives\habitat_datasets\`

### Extract-only (no network) commands

- DOTA + iSAID + TartanAir
```powershell
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" datasets\download_datasets.py `
  --keys dota isaid tartanair `
  --dest D:\datasets `
  --extract-only `
  --no-reextract
```

- VisDrone + UAVDT
```powershell
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" datasets\download_datasets.py `
  --keys visdrone uavdt `
  --dest D:\datasets `
  --extract-only `
  --no-reextract
```

- Protected set example (adjust keys to what you downloaded)
```powershell
& "d:\docs\lesnar\Lesnar AI\airsim-env\Scripts\python.exe" datasets\download_datasets.py `
  --keys cityscapes mapillary_vistas bdd100k semantic_kitti nuscenes waymo_open argoverse habitat_datasets kitti_raw kitti_360 `
  --dest D:\datasets `
  --extract-only `
  --no-reextract
```
