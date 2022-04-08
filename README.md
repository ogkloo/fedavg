# fedavg
Pytorch fedavg implementation.

# Nix
This project uses nix to manage system dependencies. After installing nix, 
you can activate a shell by using:

```
$ nix-shell --pure
[nix-shell] $ virtualenv torch-env  
[nix-shell] $ source torch-env/bin/activate
[nix-shell] $ pip install requirements.txt
```

After that, you can use the project as intended.