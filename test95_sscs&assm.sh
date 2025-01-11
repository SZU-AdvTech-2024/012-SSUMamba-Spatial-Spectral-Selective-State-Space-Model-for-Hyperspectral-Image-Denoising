python main.py gpu_ids=\'7\' data.bs=1 mode=test model=ssumamba_sscs \
noise.params.sigma_max=95 test=icvl95 test.b_size=1 \
ckpt_path=SSCSepoch44-val_mpsnr48.16.ckpt \
test.save_raw=true test=icvl95