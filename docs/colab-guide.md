# Running on Google Colab

The three notebooks in `notebooks/` cover the whole pipeline on Colab. They can be opened in the browser or from VS Code with the Colab extension.

## Setup

1. Zip the repository as `windsurrogate_code.zip` and upload it into `MyDrive/windsurrogate/`.
   - Drag the file itself into the folder. Uploading a folder can create a second folder called `windsurrogate (1)`.
2. The first cell of every notebook:
   - mounts Drive only if it is not mounted yet;
   - stops if the zip is missing;
   - unzips the code into `/content/windsurrogate`.
3. OpenFOAM is installed from conda-forge with micromamba the first time, then cached on Drive (`MyDrive/windsurrogate/_cache/foam-env-2412.tar.gz`). Later sessions only untar it, which takes about a minute.

## Generating the dataset (notebook 01)

- **Runtime:** CPU with High-RAM. It has 8 vCPUs, which are 4 physical cores. The config cell sets `CONCURRENT = ncpu // 2`, one case per physical core; running 8 cases on 4 cores is not faster.
- **Prepare once:** `prepare` downloads the DEMs and writes `cases.jsonl`. Run it in one session only, before starting other shards. Re-running it is harmless.
- **Shards:** each server runs one shard.
  - With N servers, set `NUM_SHARDS = N` and give each server a different `SHARD` from 0 to N−1, all running at the same time.
  - `NUM_SHARDS` may change between batches, as long as no worker from the previous batch is still running.
  - Running only some of the shards leaves the cases of the missing ones undone.
- **One server per notebook:** use *Select Kernel → Colab → New Colab Server* for each shard. Two notebooks attached to the same server compete for its 4 cores. A lock refuses to start the same shard twice on one machine.
- **Session length:** Colab sessions end after roughly 12 hours, so `HOURS = 11` stops starting new cases in time.
- **Resuming:** finished cases are detected by their `.npz` on Drive. Just run the cells again on a new server.
- **Failed cases:**
  - A case is retried at most twice.
  - Stopping a cell (■) marks the cases that were running as failed.
  - To give them another try, run the optional cleanup cell in one session before starting the shards. It deletes only `failed` status files, never results.
- **Progress:** the progress cell counts finished cases per split and lists status counts.
- **`conv=False` in the log:** the case oscillated and was labelled with the iteration average. This is expected on very steep terrain.

## Training (notebook 02)

- **Runtime:** GPU. A100 or G4 are fastest; L4 is fine; T4 works but is slower. Standard RAM is enough.
- **Data copy:** the dataset is copied to local disk first, because Drive is slow with thousands of small files. For ~1,200 cases (~1.8 GB) the copy takes about half an hour.
- **Resuming:** training resumes from `checkpoints/last.pt`. The best epoch on the validation split is saved as `checkpoints/best.pt`.
- **Evaluation:** the last cells evaluate `best.pt` on the test split and show a few CFD-vs-U-Net figures.

## Maps (notebook 03)

- **Output:** one GeoTIFF per region and wind direction (66 bands at 50 m, roughly 0.5–1 GB each).
- **Resuming:** existing maps are skipped, and each map is written under a temporary name and renamed only when complete. After a disconnect, just re-run the cell.

## Drive: things that go wrong

- **Never run `rm -rf /content/drive`.** If Drive is mounted, it deletes the contents of your whole Google Drive.
  - Files deleted from Colab go to the Drive trash, one item per file. To recover them, restore everything from the trash (select all, *Restore*), not just the folders.
  - To check whether Drive is mounted: `!mount | grep /content/drive || echo "NOT MOUNTED"`.
- **"Mountpoint must not already contain files":** something wrote into `/content/drive` before Drive was mounted. Do not delete anything by hand. Start a new server and run the first cell, which mounts before writing.
- **`OSError: [Errno 107] Transport endpoint is not connected`:** the Drive mount dropped, usually after heavy small-file I/O. Run `drive.mount('/content/drive', force_remount=True)`. If that hangs, restart the session and run the setup cells again. Nothing is lost.
- **A folder shows up as `name (1)`** on a server but not in the Drive web UI: the mount is caching a deleted folder with the same name. A new server sees the right name.
- **The account matters:** `drive.mount` and `auth.authenticate_user()` may pick the account already signed in to Colab without asking. Check with `!ls /content/drive/MyDrive` that you are looking at the right Drive before running anything.

## Notebook gotcha

In `!` shell lines IPython replaces `$name` and `{expr}` with Python variables. Shell loops that use `$var` should go in a `%%bash` cell instead.

## Cost

- One CPU High-RAM session costs about 0.5 compute units per hour and produces about 5–7 cases per hour.
- A GPU session costs several units per hour. End GPU sessions as soon as training or prediction is done.
- Colab Pro limits how many sessions can run at the same time. *Runtime → Manage sessions* shows what is running.
