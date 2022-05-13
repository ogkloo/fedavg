with import <nixpkgs> {};
mkShell {
  name = "fedavg-pytorch";

  buildInputs = with python3.pkgs; 
  let
    python-with-packages = python.buildEnv.override {
      extraLibs = with pythonPackages; [ numpy ];
    };
  in
  [
    git
    pip
    python-with-packages
    cudaPackages.cudatoolkit_11_3
    cudnn_cudatoolkit_11_3
  ];

  # Explanation of shell hook variables:
  # 
  # LD_LIBRARY_PATH: allows the shell to find the stdenv and CUDA/cudnn. 
  # Without these lines, you'll get errors related to missing *.so, related to
  # CUDA stuff. 
  #
  # CUDA_PATH: Basically the same as above, but will give somewhat different 
  # errors. Have not tested what these are out of fear.
  #
  # TMPDIR: This is to make sure that when you install torch it has sufficient
  # space to install.
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit_11_3}/lib:${pkgs.cudaPackages.cudatoolkit_11_3}/lib64::${pkgs.cudnn_cudatoolkit_11_3}/lib:${pkgs.cudaPackages.cudatoolkit_11_3.lib}/lib:/run/opengl-driver/lib:/run/opengl-driver-32/lib:/lib:${pkgs.cudaPackages.cudatoolkit_11_3}/extras/CUPTI/lib64:$LD_LIBRARY_PATH
    export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit_11_3}"
    export TMPDIR="$(pwd)/tmp"
  '';
}
