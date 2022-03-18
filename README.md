# DentNet
Pytorch __model only__ of U-Net, as used in the paper:

> Pasquale Lafiosca et al., *Automatic Segmentation of Aircraft Dents in Point Clouds*, SAE Technical Paper, 2022, https://doi.org/10.4271/2022-01-0022

If you find this useful, please cite as
```
@inproceedings{dentNet,
  title={Automatic segmentation of aircraft dents in point clouds},
  author={Lafiosca, Pasquale and Fan, Ip-Shing and Avdelidis, Nicolas P},
  year={2022},
  institution={SAE Technical Paper}
  doi={10.4271/2022-01-0022}
}
```

## Other info
In the paper, the loss function used was `nn.BCEWithLogitsLoss` and the optimizer was `optim.Adam`.
Dataset not shared.

## Notes
Please note: this repository is __not__ maintained.
