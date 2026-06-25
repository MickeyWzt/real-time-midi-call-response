# AI MIDI A/B Listening Test

Static 12-pair blind listening page for the MFP call-and-response paper.
The current sample set is stratified from Call100: six POP909-derived calls, four real public dataset calls, and two artificial stress / hard-case calls.

## What Is Included

- `index.html`, `styles.css`, `app.js`: the deployable listening page.
- `public/study-pairs.json`: public anonymous pair list.
- `public/midi/*.mid`: anonymous A/B MIDI clips.
- `public/config.js`: endpoint configuration.
- `google_apps_script/Code.gs`: Google Sheets receiver.
- `private/answer_key.csv`: local-only mapping from pair/side to candidate strategy.
- `tools/build_study_pairs.py`: reproducible sampler for the public MIDI clips and private answer key.
- `tools/analyze_results.py`: joins exported Sheet CSV with the private answer key and prints preference counts.

Do not publish `private/answer_key.csv` with participant-facing materials.

## Google Sheet Setup

1. Create a Google Sheet.
2. Open **Extensions -> Apps Script**.
3. Paste `google_apps_script/Code.gs`.
4. Run `setupSheet()` once and approve permissions.
5. Click **Deploy -> New deployment -> Web app**.
6. Set **Execute as** to yourself.
7. Set **Who has access** to anyone with the link.
8. Copy the Web App URL ending in `/exec`.
9. Paste it into `public/config.js` as `sheetEndpoint`.

The browser submits one row per pair. A 12-pair completed test creates 12 rows. Each row includes the A/B choice, per-clip play counts, `pair_started_at`, `pair_choice_at`, `pair_duration_seconds`, and optional recruitment-source fields from URL parameters such as `source`, `utm_source`, `utm_medium`, and `utm_campaign`. Participants must choose A or B for each pair; there is no tie option.

## Local Test

From this folder:

```powershell
python -m http.server 4173
```

Then open:

```text
http://127.0.0.1:4173
```

The page works without a Google endpoint, but it will only save a local browser backup.

## Deploy

Deploy the `online_blind_listening` folder as a static site with Vercel, Netlify, Cloudflare Pages, or GitHub Pages.

## Rebuild The Sample Set

The study clips are generated from Call100 using this rule:

- 12 calls total: 6 POP909, 4 real public dataset, 2 artificial stress / hard cases.
- All responses use the `pentatonic_conservative` preset.
- For each candidate and call, choose a no-fallback trial whose `objective_score` is closest to that candidate's 15-trial median.
- 8 pairs compare `amt_small_controlled` against `amt_small_raw`.
- 4 pairs compare `amt_small_controlled` against `motif_transform_baseline`.
- A/B side order is deterministically randomized and model identity is hidden from participants.
- The participant-facing UI forces a binary A/B choice and immediately advances after each choice.
- The participant-facing UI does not show MIDI piano-rolls, waveforms, note counts, or other visual cues about the clips.
- Each pair presents one separate Call clip, then two response-only A/B clips.

Run:

```powershell
python tools/build_study_pairs.py
```

For Reddit, use a short title such as:

```text
[Academic] 5-minute A/B music listening test for AI MIDI call-and-response
```

## Data Analysis

Export the Google Sheet as CSV. Join by `pair_id` and chosen side against `private/answer_key.csv` to recover whether the listener preferred `amt_small_controlled`, `amt_small_raw`, or `motif_transform_baseline`.

You can run:

```powershell
python tools/analyze_results.py path\to\BlindListeningResponses.csv
```

The script prints overall candidate preference counts and a pair-level summary.
