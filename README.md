# factai

1. Install environment through this jobfile: /jobfiles/install_env.job 
We added two packages for the embeddings: 1. 2. 

2. Training is possible, here is an example (as a warning, you might have to download the used reference data sets)
```
# Run your code
srun python -u /home/scur1031/ITI-GEN/train_iti_gen.py \
    --prompt='a headshot of a person' \
    --attr-list='Male' \
    --epochs=30 \
    --save-ckpt-per-epochs=10 \
    --data-path='/home/scur1031/ITI-GEN/data' \
    --ckpt-path='/home/scur1031/ITI-GEN/ckpts/test'
```


3. Generation
    Generate ITI-GEN images
    Generate baseline images

4. Embeddings

