## Training
- To train the point-cloud auto-encoder, run:
```shell
bash scripts/train_ae.py
```
Note that the auto-encoder needs to be trained before training either of the latent matching setups.

- To train the latent matching (lm) setup, run:
```shell
bash scripts/train_lm.py
```

- To train the probabilistic latent matching (plm) setup, run:
```shell
bash scripts/train_plm.py
```

## Trained Models
Create a folder called 'trained_models' inside the project folder:
```
mkdir trained_models
```
- Download the trained model for latent matching (lm) here:<br>
[https://drive.google.com/open?id=1nl30z1CJL5WZn8svFllHkVLvS4GWiAxf](https://drive.google.com/open?id=1nl30z1CJL5WZn8svFllHkVLvS4GWiAxf) <br>
Extract it and move it into *trained_models/*

- Download the trained model for probabilistic latent matching (plm) here:<br>
[https://drive.google.com/open?id=1iYUOPTrhwAIwubihrBhLGG-KVD4Qr7is](https://drive.google.com/open?id=1iYUOPTrhwAIwubihrBhLGG-KVD4Qr7is) <br>
Extract it and move it into *trained_models/*

## Evaluation
Follow the steps detailed above to download the dataset and pre-trained models.

- For computing the Chamfer and EMD metrics reported in the paper (all 13 categories), run:
```shell
bash scripts/metrics_lm.sh
```
The computed metrics will be saved inside *trained_models/lm/metrics/*

- For the plm setup (chair category), run:
```shell
bash scripts/metrics_plm.sh
```
The computed metrics will be saved inside *trained_models/plm/metrics/*

## Demo
Follow the steps detailed above to download the dataset and pre-trained models.

- Run the following to visualize the results for latent matching (lm):
```shell
bash scripts/demo_lm.sh
```
You can navigate to the next visualization by pressing 'q'.

- Run the following to visualize the results for probabilistic latent matching (plm):
```shell
bash scripts/demo_plm.sh
```

## Sample Results
Below are a few sample reconstructions from our trained model.
![3D-LMNet_sample_results](images/sample_results.png)

