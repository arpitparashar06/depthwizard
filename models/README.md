# models/

Put the trained HeightNet checkpoint here as **`heightnet.pt`** (the
`heightnet_best.pt` that `training/train.py` writes - rename it). The app,
the CLI and the benchmark find it automatically; delete it to go back to the
zero-shot engine. A different location works too:
`DEPTHWIZARD_HEIGHTNET=/path/to/file.pt`.

Check it loaded: `python mathsandml/heightnet.py some_image.tif` prints the
model name and the height statistics, and `/api/health` reports
`"engine": "heightnet"`.
