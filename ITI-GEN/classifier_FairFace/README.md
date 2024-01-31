# FairFace classifier:
The FairFace classifier is used to classify the multi-category attributes 'age' and 'skin tone'.

The job file to run the code of the FairFace classifier is called classifier_fairface.job:
```
srun python -u FairFace/predict.py \
    --input_csv "FairFace/img_paths.csv" \
    --output_csv "FairFace/outputs.csv" \
    --image_path "ckpts/a_headshot_of_a_person_Male_Skin_tone/original_prompt_embedding/sample_results"
```
- '--input_csv': csv file of image paths where col name for image path is "img_path'.
- '--output_csv': csv file for the output of the classifier.
- '--image_path': directory where the images are stored.

The github page from the paper with the original code can be found here: https://github.com/dchen236/FairFace
