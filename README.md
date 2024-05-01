# bspline-jax
Simple implementation of B-Splines, M-Splines and I-Splines in jax, supporting jitting for fast and precomputation of the basis for even faster evaluation. 
It also supports setting of values (including values of the derivatives) on the boundary of the definition interval.

## Usage
  For example usage look in the 
  
   ```
    if __name__ == "__main__":
    ...
  ```

section of each file. If you execute the file, it will first precompute and cache the basis splines and then generate plots of random splines. 
