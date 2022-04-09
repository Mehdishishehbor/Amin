# LMGP-PyTorch

LMGP-Gpytorch is implementation of Latent map Gaussian process (LMGP) for modeling data with qualitative and quantititave variables. Currently the code uses the many 



## License

Please contact raminb@uci.edu for further info.


## NN-latent
- Adding NN with custom architecture to the LMGP code
- Using dimensionality reduction and clustering techniques to find latent map variables


## Note 1
There is a paramter in gpytorch kernels called active_dims which specied what dimension if the input should be used for that kernel.
This line is defined twice in lmgp.py function. The dimenions has nothing to do with the input and it depends on how we are feeding the x
in the forward method in lmgp. Cuarrently, the latent map dimensions are the first d dimeniosn and then we have other inputs. So, that's how I have adefined the active_dims.